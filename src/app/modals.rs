//! One input-owning modal, with a separate dismissible notification layer.
use super::{Confirm, Message, ModelSearch, Prompt, Selector};

pub enum Modal {
    Help,
    Prompt(Prompt),
    Selector(Selector),
    Search(ModelSearch),
    Confirm(Confirm),
}

#[derive(Default)]
pub struct Modals {
    active: Option<Modal>,
    pub message: Option<Message>,
}

impl Modals {
    pub fn help(&self) -> bool {
        matches!(self.active, Some(Modal::Help))
    }
    pub(super) fn set_help(&mut self, open: bool) {
        if open {
            self.active = Some(Modal::Help);
        } else if self.help() {
            self.active = None;
        }
    }
    pub fn prompt(&self) -> Option<&Prompt> {
        match &self.active {
            Some(Modal::Prompt(value)) => Some(value),
            _ => None,
        }
    }
    pub(super) fn set_prompt(&mut self, value: Option<Prompt>) {
        if let Some(value) = value {
            self.active = Some(Modal::Prompt(value));
        } else if self.prompt().is_some() {
            self.active = None;
        }
    }
    pub(super) fn prompt_mut(&mut self) -> Option<&mut Prompt> {
        match &mut self.active {
            Some(Modal::Prompt(value)) => Some(value),
            _ => None,
        }
    }
    pub fn selector(&self) -> Option<&Selector> {
        match &self.active {
            Some(Modal::Selector(value)) => Some(value),
            _ => None,
        }
    }
    pub(super) fn set_selector(&mut self, value: Option<Selector>) {
        if let Some(value) = value {
            self.active = Some(Modal::Selector(value));
        } else if self.selector().is_some() {
            self.active = None;
        }
    }
    pub(super) fn selector_mut(&mut self) -> Option<&mut Selector> {
        match &mut self.active {
            Some(Modal::Selector(value)) => Some(value),
            _ => None,
        }
    }
    pub(super) fn take_selector(&mut self) -> Option<Selector> {
        self.selector()?;
        match self.active.take() {
            Some(Modal::Selector(value)) => Some(value),
            _ => None,
        }
    }
    pub fn search(&self) -> Option<&ModelSearch> {
        match &self.active {
            Some(Modal::Search(value)) => Some(value),
            _ => None,
        }
    }
    pub(super) fn set_search(&mut self, value: Option<ModelSearch>) {
        if let Some(value) = value {
            self.active = Some(Modal::Search(value));
        } else if self.search().is_some() {
            self.active = None;
        }
    }
    pub(super) fn search_mut(&mut self) -> Option<&mut ModelSearch> {
        match &mut self.active {
            Some(Modal::Search(value)) => Some(value),
            _ => None,
        }
    }
    pub fn confirm(&self) -> Option<&Confirm> {
        match &self.active {
            Some(Modal::Confirm(value)) => Some(value),
            _ => None,
        }
    }
    pub(super) fn set_confirm(&mut self, value: Option<Confirm>) {
        if let Some(value) = value {
            self.active = Some(Modal::Confirm(value));
        } else if self.confirm().is_some() {
            self.active = None;
        }
    }
    pub(super) fn take_confirm(&mut self) -> Option<Confirm> {
        self.confirm()?;
        match self.active.take() {
            Some(Modal::Confirm(value)) => Some(value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacing_a_dialog_changes_the_input_owner_but_preserves_notifications() {
        let mut modals = Modals::default();
        modals.set_help(true);
        modals.set_prompt(Some(Prompt {
            kind: super::super::PromptKind::NewProfile,
            title: "name".into(),
            buffer: String::new(),
            error: None,
        }));
        assert!(!modals.help());
        modals.set_search(None);
        assert!(modals.prompt().is_some(), "closing an absent search must not close an editor");
        modals.message = Some(Message { title: "save failed".into(), lines: vec![] });
        modals.set_prompt(None);
        assert!(modals.message.is_some(), "finishing an edit must not hide a save error");
        assert!(modals.active.is_none());
    }
}
