//! Presentation state for the jobs list and log viewer.
use super::SessionPane;
use ratatui::layout::Rect;
use ratatui::widgets::ListState;

pub struct SessionViewState {
    pub pane: SessionPane,
    pub detail_pane_visible: bool,
    pub selection: ListState,
    pub log_lines: Vec<String>,
    pub log_follow: bool,
    pub log_scroll: u16,
    /// How many rows the Log pane beside the list is scrolled back from the
    /// end of the log; 0 follows the tail.
    pub pane_log_back: usize,
    /// Where the list and the column beside it were last drawn, so a mouse
    /// wheel can be routed to whichever the pointer is over.
    pub list_area: Option<Rect>,
    pub detail_area: Option<Rect>,
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            pane: SessionPane::Monitor,
            detail_pane_visible: true,
            selection: ListState::default(),
            log_lines: vec![],
            log_follow: true,
            log_scroll: 0,
            pane_log_back: 0,
            list_area: None,
            detail_area: None,
        }
    }
}

impl SessionViewState {
    pub(super) fn select_last(&mut self, count: usize) {
        self.selection.select(count.checked_sub(1));
    }

    pub(super) fn sync_selection(&mut self, count: usize) {
        self.selection.select(if count == 0 {
            None
        } else {
            Some(self.selection.selected().unwrap_or(0).min(count - 1))
        });
    }

    pub(super) fn move_selection(&mut self, delta: isize, count: usize) {
        if count == 0 {
            self.selection.select(None);
            return;
        }
        let current = self.selection.selected().unwrap_or(0);
        self.selection.select(Some(current.saturating_add_signed(delta).min(count - 1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_jobs_clamps_selection_and_empty_lists_clear_it() {
        let mut view = SessionViewState::default();
        view.selection.select(Some(4));
        view.sync_selection(2);
        assert_eq!(view.selection.selected(), Some(1));
        view.move_selection(isize::MIN, 2);
        assert_eq!(view.selection.selected(), Some(0));
        view.sync_selection(0);
        assert_eq!(view.selection.selected(), None);
    }

    #[test]
    fn selecting_last_on_an_empty_list_clears_the_selection() {
        let mut view = SessionViewState::default();
        view.selection.select(Some(2));
        view.select_last(0);
        assert_eq!(view.selection.selected(), None);

        view.select_last(3);
        assert_eq!(view.selection.selected(), Some(2));
    }
}
