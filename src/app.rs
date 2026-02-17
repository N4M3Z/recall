use crate::index::{discover_and_sort_files, index_files, IndexProgress, IndexState, SessionIndex};
use crate::parser;
use crate::session::{SearchResult, Session};
use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Debounce delay for search (avoid searching on every keystroke during fast typing/paste)
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(50);

/// Messages from the indexing thread
pub enum IndexMsg {
    Progress { indexed: usize, total: usize },
    Done { total_sessions: usize },
    NeedsReload,
    Error(String),
}

/// Search scope
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchScope {
    /// Search all conversations
    Everything,
    /// Search only conversations from a specific folder
    Folder(String),
}

pub struct App {
    /// Current search query
    pub query: String,
    /// Cursor position in query (char index)
    pub cursor: usize,
    /// Search results
    pub results: Vec<SearchResult>,
    /// Selected result index
    pub selected: usize,
    /// Results list scroll offset
    pub list_scroll: usize,
    /// Preview scroll offset
    pub preview_scroll: usize,
    /// Currently focused message index in preview (None = auto-focus on matched message)
    pub focused_message: Option<usize>,
    /// Set of expanded message indices (shown in full, not truncated)
    pub expanded_messages: HashSet<usize>,
    /// Total message count in current preview (for navigation bounds)
    pub preview_message_count: usize,
    /// Whether the focused message can be expanded/collapsed
    pub focused_message_expandable: bool,
    /// Line ranges for each message in preview (start_line, end_line) for mouse click mapping
    pub message_line_ranges: Vec<(usize, usize)>,
    /// Preview area bounds (x, y, width, height) for mouse hit testing
    pub preview_area: (u16, u16, u16, u16),
    /// Whether to auto-scroll preview to matched message
    pub pending_auto_scroll: bool,
    /// Whether preview has more content than visible (for scroll hint)
    pub preview_scrollable: bool,
    /// Should quit
    pub should_quit: bool,
    /// Should execute resume (set on Enter)
    pub should_resume: Option<Session>,
    /// Session ID to copy (set on Tab)
    pub should_copy: Option<String>,
    /// Index for searching
    index: SessionIndex,
    /// Status message (for indexing progress, etc.)
    pub status: Option<String>,
    /// Total sessions indexed
    pub total_sessions: usize,
    /// Channel to receive indexing updates
    index_rx: Option<Receiver<IndexMsg>>,
    /// Is indexing in progress
    pub indexing: bool,
    /// Current search scope
    pub search_scope: SearchScope,
    /// Launch directory (for folder-scoped search)
    pub launch_cwd: String,
    /// Whether a search is pending (for debouncing)
    search_pending: bool,
    /// When the last input occurred (for debouncing)
    last_input: Instant,
    /// Error from indexing thread (shown on exit)
    pub index_error: Option<String>,
    /// Extra arguments to append to the resume command (from -- passthrough)
    pub resume_args: Vec<String>,
}

impl App {
    pub fn new(initial_query: String) -> Result<Self> {
        // Allow override for testing
        let cache_dir = std::env::var("RECALL_HOME_OVERRIDE")
            .map(|h| PathBuf::from(h).join(".cache").join("recall"))
            .unwrap_or_else(|_| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("recall")
            });

        let index_path = cache_dir.join("index");
        let state_path = cache_dir.join("state.json");

        let index = SessionIndex::open_or_create(&index_path)?;

        // Get launch directory (override for tests)
        let launch_cwd = std::env::var("RECALL_CWD_OVERRIDE").unwrap_or_else(|_| {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
        });

        // Start background indexing
        let (tx, rx) = mpsc::channel();
        let index_path_clone = index_path.clone();
        thread::spawn(move || {
            background_index(index_path_clone, state_path, tx);
        });

        let initial_cursor = initial_query.chars().count();
        let mut app = Self {
            query: initial_query,
            cursor: initial_cursor,
            results: Vec::new(),
            selected: 0,
            list_scroll: 0,
            preview_scroll: 0,
            focused_message: None,
            expanded_messages: HashSet::new(),
            preview_message_count: 0,
            focused_message_expandable: false,
            message_line_ranges: Vec::new(),
            preview_area: (0, 0, 0, 0),
            pending_auto_scroll: false,
            preview_scrollable: false,
            should_quit: false,
            should_resume: None,
            should_copy: None,
            index,
            status: None,
            total_sessions: 0,
            index_rx: Some(rx),
            indexing: true,
            search_scope: SearchScope::Folder(launch_cwd.clone()),
            launch_cwd,
            search_pending: false,
            last_input: Instant::now(),
            index_error: None,
            resume_args: Vec::new(),
        };

        // If there's an initial query, run the search immediately
        if !app.query.is_empty() {
            let _ = app.search();
        }

        Ok(app)
    }

    /// Check for indexing updates (call this in the main loop)
    pub fn poll_index_updates(&mut self) {
        use std::sync::mpsc::TryRecvError;

        let Some(rx) = &self.index_rx else {
            return;
        };

        // Collect messages, tracking if channel was disconnected
        let mut messages = Vec::new();
        let mut channel_disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(msg) => messages.push(msg),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    channel_disconnected = true;
                    break;
                }
            }
        }

        let mut should_close_rx = false;
        let mut needs_reload = false;
        let mut needs_search = false;

        for msg in messages {
            match msg {
                IndexMsg::Progress { indexed, total } => {
                    self.status = Some(format!("Indexing {}/{}...", indexed, total));
                    self.total_sessions = indexed;
                }
                IndexMsg::NeedsReload => {
                    needs_reload = true;
                    needs_search = true;
                }
                IndexMsg::Done { total_sessions } => {
                    self.total_sessions = total_sessions;
                    self.status = None;
                    self.indexing = false;
                    should_close_rx = true;
                    needs_reload = true;
                    needs_search = true;
                }
                IndexMsg::Error(err) => {
                    self.index_error = Some(err);
                    self.status = Some("Index error • Ctrl+C for details".to_string());
                    self.indexing = false;
                    should_close_rx = true;
                }
            }
        }

        // Detect unexpected indexer death (channel closed without Done/Error)
        if channel_disconnected && self.indexing {
            self.index_error = Some("Indexer stopped unexpectedly (possible crash)".to_string());
            self.status = Some("Index error • Ctrl+C for details".to_string());
            self.indexing = false;
            should_close_rx = true;
        }

        if needs_reload {
            let _ = self.index.reload();
        }
        if needs_search {
            let _ = self.search();
        }
        if should_close_rx {
            self.index_rx = None;
        }
    }

    /// Perform a search (or show recent sessions if query is empty)
    pub fn search(&mut self) -> Result<()> {
        // Remember currently selected session to preserve selection
        let selected_session_id = self.results.get(self.selected).map(|r| r.session.id.clone());

        let mut results = if self.query.is_empty() {
            self.index.recent(50)?
        } else {
            self.index.search(&self.query, 50)?
        };

        // Filter by scope if searching within a folder
        if let SearchScope::Folder(ref cwd) = self.search_scope {
            results.retain(|r| r.session.cwd == *cwd);
        }

        self.results = results;

        // Try to preserve selection on the same session
        if let Some(ref id) = selected_session_id {
            if let Some(pos) = self.results.iter().position(|r| &r.session.id == id) {
                self.selected = pos;
                // Scroll to keep selection visible (at top of list area)
                self.list_scroll = pos;
            } else {
                self.selected = 0;
                self.list_scroll = 0;
            }
        } else {
            self.selected = 0;
            self.list_scroll = 0;
        }
        self.update_preview_scroll();

        Ok(())
    }

    /// Toggle search scope between everything and current folder
    pub fn toggle_scope(&mut self) {
        self.search_scope = match self.search_scope {
            SearchScope::Everything => SearchScope::Folder(self.launch_cwd.clone()),
            SearchScope::Folder(_) => SearchScope::Everything,
        };
        let _ = self.search();
    }

    /// Get the folder name for display (last component of path)
    pub fn scope_folder_name(&self) -> Option<&str> {
        match &self.search_scope {
            SearchScope::Everything => None,
            SearchScope::Folder(path) => {
                path.rsplit(std::path::MAIN_SEPARATOR).next()
            }
        }
    }

    /// Get a compact display path for the scope
    /// - Replaces home dir with ~
    /// - If short enough, shows full path
    /// - Otherwise shows ~/.../<dir> or /.../<dir>
    pub fn scope_display_path(&self) -> Option<String> {
        let path = match &self.search_scope {
            SearchScope::Everything => return None,
            SearchScope::Folder(path) => path.as_str(),
        };

        // Replace home dir with ~ (HOME on Unix, USERPROFILE on Windows)
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        let display_path = if !home.is_empty() && path.starts_with(&home) {
            format!("~{}", &path[home.len()..])
        } else {
            path.to_string()
        };

        // If short enough, show full path
        const MAX_LEN: usize = 25;
        if display_path.len() <= MAX_LEN {
            return Some(display_path);
        }

        // Otherwise show prefix/.../<last_dir>
        let last_component = path.rsplit(std::path::MAIN_SEPARATOR).next().unwrap_or(path);
        let prefix = if display_path.starts_with('~') { "~" } else { "" };
        Some(format!("{}/.../{}", prefix, last_component))
    }

    /// Handle character input
    pub fn on_char(&mut self, c: char) {
        // Insert at cursor position
        let byte_pos = self.cursor_byte_pos();
        self.query.insert(byte_pos, c);
        self.cursor += 1;
        self.mark_search_pending();
    }

    /// Handle backspace
    pub fn on_backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let byte_pos = self.cursor_byte_pos();
            self.query.remove(byte_pos);
            self.mark_search_pending();
        }
    }

    /// Handle delete key
    pub fn on_delete(&mut self) {
        let char_count = self.query.chars().count();
        if self.cursor < char_count {
            let byte_pos = self.cursor_byte_pos();
            self.query.remove(byte_pos);
            self.mark_search_pending();
        }
    }

    /// Clear search
    pub fn on_escape(&mut self) {
        if self.query.is_empty() {
            self.should_quit = true;
        } else {
            self.query.clear();
            self.cursor = 0;
            self.mark_search_pending();
        }
    }

    /// Move cursor left
    pub fn on_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    /// Move cursor right
    pub fn on_right(&mut self) {
        let char_count = self.query.chars().count();
        if self.cursor < char_count {
            self.cursor += 1;
        }
    }

    /// Move cursor to start
    pub fn on_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end
    pub fn on_end(&mut self) {
        self.cursor = self.query.chars().count();
    }

    /// Convert cursor (char index) to byte position
    fn cursor_byte_pos(&self) -> usize {
        self.query.char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.query.len())
    }

    /// Mark that a search is needed (debounced)
    fn mark_search_pending(&mut self) {
        self.search_pending = true;
        self.last_input = Instant::now();
    }

    /// Check if debounce period has elapsed and trigger search if needed
    pub fn maybe_search(&mut self) {
        if self.search_pending && self.last_input.elapsed() >= SEARCH_DEBOUNCE {
            self.search_pending = false;
            let _ = self.search();
        }
    }

    /// Force any pending search to run immediately (for tests)
    pub fn flush_pending_search(&mut self) {
        if self.search_pending {
            self.search_pending = false;
            let _ = self.search();
        }
    }

    /// Move selection up
    pub fn on_up(&mut self) {
        if !self.results.is_empty() {
            self.selected = self.selected.saturating_sub(1);
            self.update_preview_scroll();
        }
    }

    /// Move selection down
    pub fn on_down(&mut self) {
        if !self.results.is_empty() {
            self.selected = (self.selected + 1).min(self.results.len() - 1);
            self.update_preview_scroll();
        }
    }

    /// Handle Tab key - copy session ID
    pub fn on_tab(&mut self) {
        if let Some(result) = self.results.get(self.selected) {
            self.should_copy = Some(result.session.id.clone());
        }
    }

    /// Handle Enter key - open conversation
    pub fn on_enter(&mut self) {
        if let Some(result) = self.results.get(self.selected) {
            if let Ok(session) = parser::parse_session_file(&result.session.file_path) {
                self.should_resume = Some(session);
            }
        }
    }

    /// Update preview scroll to show the matched message
    fn update_preview_scroll(&mut self) {
        // Signal that we need to auto-scroll to the matched message
        // The actual scroll position is calculated in render_preview
        // since it depends on wrapped line counts
        self.pending_auto_scroll = true;
        self.preview_scroll = 0;
        // Reset focus and expansions when switching sessions
        self.focused_message = None;
        self.expanded_messages.clear();
    }

    /// Scroll preview up
    pub fn scroll_preview_up(&mut self, lines: usize) {
        self.preview_scroll = self.preview_scroll.saturating_sub(lines);
    }

    /// Scroll preview down
    pub fn scroll_preview_down(&mut self, lines: usize) {
        self.preview_scroll = self.preview_scroll.saturating_add(lines);
    }

    /// Navigate to previous message in preview
    pub fn focus_prev_message(&mut self) {
        if self.preview_message_count == 0 {
            return;
        }
        let matched_idx = self
            .selected_result()
            .map(|r| r.matched_message_index)
            .unwrap_or(0);
        let current = self.focused_message.unwrap_or(matched_idx);
        if current > 0 {
            self.focused_message = Some(current - 1);
            self.pending_auto_scroll = true;
        }
    }

    /// Navigate to next message in preview
    pub fn focus_next_message(&mut self) {
        if self.preview_message_count == 0 {
            return;
        }
        let matched_idx = self
            .selected_result()
            .map(|r| r.matched_message_index)
            .unwrap_or(0);
        let current = self.focused_message.unwrap_or(matched_idx);
        if current + 1 < self.preview_message_count {
            self.focused_message = Some(current + 1);
            self.pending_auto_scroll = true;
        }
    }

    /// Toggle expansion of the focused message
    pub fn toggle_focused_expansion(&mut self) {
        if self.preview_message_count == 0 {
            return;
        }
        let matched_idx = self
            .selected_result()
            .map(|r| r.matched_message_index)
            .unwrap_or(0);
        let focused = self.focused_message.unwrap_or(matched_idx);
        if self.expanded_messages.contains(&focused) {
            self.expanded_messages.remove(&focused);
        } else {
            self.expanded_messages.insert(focused);
        }
    }

    /// Get the currently selected result
    pub fn selected_result(&self) -> Option<&SearchResult> {
        self.results.get(self.selected)
    }

    /// Handle mouse click in preview area - returns true if a message was clicked
    pub fn click_preview_message(&mut self, x: u16, y: u16) -> bool {
        let (px, py, pw, ph) = self.preview_area;

        // Check if click is within preview bounds
        if x < px || x >= px + pw || y < py || y >= py + ph {
            return false;
        }

        // Calculate which line was clicked (accounting for scroll)
        let clicked_line = (y - py) as usize + self.preview_scroll;

        // Find which message contains this line
        for (msg_idx, &(start, end)) in self.message_line_ranges.iter().enumerate() {
            if clicked_line >= start && clicked_line < end {
                self.focused_message = Some(msg_idx);
                return true;
            }
        }

        false
    }
}

/// Background indexing function
fn background_index(index_path: PathBuf, state_path: PathBuf, tx: Sender<IndexMsg>) {
    let index = match SessionIndex::open_or_create(&index_path) {
        Ok(idx) => idx,
        Err(e) => {
            let _ = tx.send(IndexMsg::Error(format!("Failed to open index: {}", e)));
            return;
        }
    };
    let mut state = match IndexState::load(&state_path) {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(IndexMsg::Error(format!("Failed to load index state: {}", e)));
            return;
        }
    };

    // Discover and sort files by mtime (most recent first)
    let files = discover_and_sort_files();

    let files_to_index: Vec<_> = files
        .iter()
        .filter(|f| state.needs_reindex(f))
        .cloned()
        .collect();

    if files_to_index.is_empty() {
        let _ = tx.send(IndexMsg::Done {
            total_sessions: files.len(),
        });
        return;
    }

    let mut writer = match index.writer() {
        Ok(w) => w,
        Err(e) => {
            let _ = tx.send(IndexMsg::Error(format!("Failed to create index writer: {}", e)));
            return;
        }
    };

    // Progress callback sends to channel
    let tx_progress = tx.clone();
    let on_progress = Box::new(move |p: IndexProgress| {
        let _ = tx_progress.send(IndexMsg::Progress {
            indexed: p.indexed,
            total: p.total,
        });
    });

    // Reload callback sends to channel
    let tx_reload = tx.clone();
    let on_reload = Box::new(move || {
        let _ = tx_reload.send(IndexMsg::NeedsReload);
    });

    let result = index_files(
        &index,
        &mut writer,
        &mut state,
        &files_to_index,
        Some(on_progress),
        Some(on_reload),
    );

    if let Err(e) = result {
        let _ = tx.send(IndexMsg::Error(format!("Indexing failed: {}", e)));
        return;
    }

    let _ = state.save(&state_path);

    let _ = tx.send(IndexMsg::Done {
        total_sessions: files.len(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal App for testing navigation/expansion features
    /// This bypasses the index initialization for unit tests
    fn test_app() -> App {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let index_path = std::env::temp_dir().join(format!("recall_test_index_{}", test_id));

        App {
            query: String::new(),
            cursor: 0,
            results: Vec::new(),
            selected: 0,
            list_scroll: 0,
            preview_scroll: 0,
            focused_message: None,
            expanded_messages: HashSet::new(),
            preview_message_count: 0,
            focused_message_expandable: false,
            message_line_ranges: Vec::new(),
            preview_area: (0, 0, 0, 0),
            pending_auto_scroll: false,
            preview_scrollable: false,
            should_quit: false,
            should_resume: None,
            should_copy: None,
            index: SessionIndex::open_or_create(&index_path).unwrap(),
            status: None,
            total_sessions: 0,
            index_rx: None,
            indexing: false,
            search_scope: SearchScope::Everything,
            launch_cwd: String::new(),
            search_pending: false,
            last_input: Instant::now(),
            index_error: None,
            resume_args: Vec::new(),
        }
    }

    // ==================== focus_prev_message tests ====================

    #[test]
    fn test_focus_prev_at_first_message_stays() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(0);

        app.focus_prev_message();

        assert_eq!(app.focused_message, Some(0));
    }

    #[test]
    fn test_focus_prev_moves_up() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(3);

        app.focus_prev_message();

        assert_eq!(app.focused_message, Some(2));
    }

    #[test]
    fn test_focus_prev_from_none_at_zero_stays_none() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = None;
        // When focused_message is None and no result, defaults to 0
        // Moving prev from 0 does nothing (already at first)

        app.focus_prev_message();

        // Stays None because we couldn't move (already at first message)
        assert_eq!(app.focused_message, None);
    }

    #[test]
    fn test_focus_prev_no_messages_noop() {
        let mut app = test_app();
        app.preview_message_count = 0;
        app.focused_message = Some(2);

        app.focus_prev_message();

        // Should not change when no messages
        assert_eq!(app.focused_message, Some(2));
    }

    // ==================== focus_next_message tests ====================

    #[test]
    fn test_focus_next_at_last_message_stays() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(4);

        app.focus_next_message();

        assert_eq!(app.focused_message, Some(4));
    }

    #[test]
    fn test_focus_next_moves_down() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(2);

        app.focus_next_message();

        assert_eq!(app.focused_message, Some(3));
    }

    #[test]
    fn test_focus_next_from_none_uses_matched_index() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = None;
        // When focused_message is None and no result, defaults to 0
        // So moving next from 0 goes to 1

        app.focus_next_message();

        assert_eq!(app.focused_message, Some(1));
    }

    #[test]
    fn test_focus_next_no_messages_noop() {
        let mut app = test_app();
        app.preview_message_count = 0;
        app.focused_message = Some(2);

        app.focus_next_message();

        // Should not change when no messages
        assert_eq!(app.focused_message, Some(2));
    }

    // ==================== toggle_focused_expansion tests ====================

    #[test]
    fn test_toggle_expands_collapsed_message() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(2);

        app.toggle_focused_expansion();

        assert!(app.expanded_messages.contains(&2));
    }

    #[test]
    fn test_toggle_collapses_expanded_message() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(2);
        app.expanded_messages.insert(2);

        app.toggle_focused_expansion();

        assert!(!app.expanded_messages.contains(&2));
    }

    #[test]
    fn test_toggle_no_messages_noop() {
        let mut app = test_app();
        app.preview_message_count = 0;
        app.focused_message = Some(2);

        app.toggle_focused_expansion();

        assert!(app.expanded_messages.is_empty());
    }

    #[test]
    fn test_multiple_messages_can_be_expanded() {
        let mut app = test_app();
        app.preview_message_count = 5;

        app.focused_message = Some(1);
        app.toggle_focused_expansion();

        app.focused_message = Some(3);
        app.toggle_focused_expansion();

        assert!(app.expanded_messages.contains(&1));
        assert!(app.expanded_messages.contains(&3));
        assert_eq!(app.expanded_messages.len(), 2);
    }

    // ==================== click_preview_message tests ====================

    #[test]
    fn test_click_inside_preview_selects_message() {
        let mut app = test_app();
        app.preview_area = (50, 5, 60, 20); // x, y, width, height
        app.preview_scroll = 0;
        app.message_line_ranges = vec![
            (0, 5),   // Message 0: lines 0-4
            (5, 12),  // Message 1: lines 5-11
            (12, 18), // Message 2: lines 12-17
        ];

        // Click on line 7 (y=5+7=12), which is in message 1
        let clicked = app.click_preview_message(55, 12);

        assert!(clicked);
        assert_eq!(app.focused_message, Some(1));
    }

    #[test]
    fn test_click_outside_preview_returns_false() {
        let mut app = test_app();
        app.preview_area = (50, 5, 60, 20);
        app.message_line_ranges = vec![(0, 5), (5, 12)];

        // Click outside preview area (x too small)
        let clicked = app.click_preview_message(10, 10);

        assert!(!clicked);
        assert_eq!(app.focused_message, None);
    }

    #[test]
    fn test_click_accounts_for_scroll() {
        let mut app = test_app();
        app.preview_area = (50, 5, 60, 20);
        app.preview_scroll = 10; // Scrolled down 10 lines
        app.message_line_ranges = vec![
            (0, 5),   // Message 0: lines 0-4
            (5, 12),  // Message 1: lines 5-11
            (12, 25), // Message 2: lines 12-24
        ];

        // Click at y=5 (top of preview), with scroll=10, actual line = 0 + 10 = 10
        // Line 10 is in message 1 (lines 5-11)
        let clicked = app.click_preview_message(55, 5);

        assert!(clicked);
        assert_eq!(app.focused_message, Some(1));
    }

    #[test]
    fn test_click_on_empty_area_returns_false() {
        let mut app = test_app();
        app.preview_area = (50, 5, 60, 20);
        app.preview_scroll = 0;
        app.message_line_ranges = vec![
            (0, 3),  // Message 0: lines 0-2
            (4, 8),  // Message 1: lines 4-7 (gap at line 3)
        ];

        // Click on line 3 which is between messages
        let clicked = app.click_preview_message(55, 8); // y=5+3=8 -> line 3

        assert!(!clicked);
    }

    // ==================== State reset tests ====================

    #[test]
    fn test_navigation_sets_pending_auto_scroll() {
        let mut app = test_app();
        app.preview_message_count = 5;
        app.focused_message = Some(2);
        app.pending_auto_scroll = false;

        app.focus_next_message();

        assert!(app.pending_auto_scroll);
    }
}
