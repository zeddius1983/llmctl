//! Presentation state for the jobs list and log viewer.
use super::SessionPane;
use ratatui::widgets::ListState;

pub struct SessionViewState {
    pub pane: SessionPane,
    pub detail_pane_visible: bool,
    pub selection: ListState,
    pub log_lines: Vec<String>,
    pub log_follow: bool,
    pub log_scroll: u16,
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            pane: SessionPane::Detail,
            detail_pane_visible: true,
            selection: ListState::default(),
            log_lines: vec![],
            log_follow: true,
            log_scroll: 0,
        }
    }
}

impl SessionViewState {
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
}
