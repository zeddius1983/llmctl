//! Browser selection and navigation invariants.
use super::{Pane, PaneList};
use crate::domain::{Model, OptionItem, Profile};
use crate::runtime::RuntimeBackend;

pub struct BrowserState {
    pub focus: Pane,
    pub runtimes: PaneList<Box<dyn RuntimeBackend>>,
    pub models: PaneList<Model>,
    pub catalog_preview: Vec<Model>,
    pub profiles: PaneList<Profile>,
    pub options: PaneList<OptionItem>,
    pub(super) catalog_prefix: Vec<String>,
    pub(super) catalog_history: Vec<(Vec<Model>, Option<usize>, Vec<String>)>,
}

impl BrowserState {
    pub(super) fn new(runtimes: Vec<Box<dyn RuntimeBackend>>) -> Self {
        Self {
            focus: Pane::Runtime,
            runtimes: PaneList::new(runtimes),
            models: PaneList::new(vec![]),
            catalog_preview: vec![],
            profiles: PaneList::new(vec![]),
            options: PaneList::new(vec![]),
            catalog_prefix: vec![],
            catalog_history: vec![],
        }
    }

    pub(super) fn enter_directory(&mut self) -> bool {
        let Some(selected) = self.models.selected() else { return false };
        if !selected.is_catalog_dir() || self.catalog_preview.is_empty() {
            return false;
        }
        self.catalog_history.push((
            self.models.items.clone(),
            self.models.state.selected(),
            self.catalog_prefix.clone(),
        ));
        self.catalog_prefix = selected.catalog_path.clone();
        self.models.replace(self.catalog_preview.clone());
        true
    }

    pub(super) fn back_directory(&mut self) -> bool {
        let Some((items, selected, prefix)) = self.catalog_history.pop() else { return false };
        self.catalog_prefix = prefix;
        self.models.replace(items);
        self.models.state.select(selected);
        true
    }

    pub(super) fn move_selection(&mut self, delta: isize) {
        match self.focus {
            Pane::Runtime => self.runtimes.move_by(delta),
            Pane::Model => self.models.move_by(delta),
            Pane::Profile => self.profiles.move_by(delta),
            Pane::Options => self.options.move_by(delta),
        }
    }
    pub(super) fn select_first(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_first(),
            Pane::Model => self.models.select_first(),
            Pane::Profile => self.profiles.select_first(),
            Pane::Options => self.options.select_first(),
        }
    }
    pub(super) fn select_last(&mut self) {
        match self.focus {
            Pane::Runtime => self.runtimes.select_last(),
            Pane::Model => self.models.select_last(),
            Pane::Profile => self.profiles.select_last(),
            Pane::Options => self.options.select_last(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory(name: &str) -> Model {
        serde_json::from_value(serde_json::json!({"name": name, "path": "", "size_bytes": 0, "has_chat_template": false, "catalog_path": [name]})).unwrap()
    }

    #[test]
    fn entering_and_leaving_a_directory_restores_its_parent_selection() {
        let mut browser = BrowserState::new(vec![]);
        browser.focus = Pane::Model;
        browser.models.replace(vec![directory("first"), directory("second")]);
        browser.models.state.select(Some(1));
        assert!(!browser.enter_directory(), "an empty preview cannot be entered");
        browser.catalog_preview = vec![directory("child")];
        assert!(browser.enter_directory());
        assert_eq!(browser.catalog_prefix, ["second"]);
        assert_eq!(browser.models.state.selected(), Some(0));
        assert!(browser.back_directory());
        assert_eq!(browser.models.state.selected(), Some(1));
        assert!(browser.catalog_prefix.is_empty());
        assert!(!browser.back_directory());
    }
}
