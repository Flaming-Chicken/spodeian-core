//! Generic bounded undo/redo state history stack.

use std::collections::VecDeque;

/// Bounded undo/redo snapshot manager for reversible editor and questionnaire actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndoStack<T: Clone> {
    max_history: usize,
    undo_stack: VecDeque<T>,
    redo_stack: Vec<T>,
}

impl<T: Clone> Default for UndoStack<T> {
    fn default() -> Self {
        Self::new(64)
    }
}

impl<T: Clone> UndoStack<T> {
    /// Creates an empty undo/redo stack keeping up to `max_history` undo states (at least 1).
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history: max_history.max(1),
            undo_stack: VecDeque::with_capacity(max_history.clamp(1, 128)),
            redo_stack: Vec::new(),
        }
    }

    /// Records `current` state before applying a mutation and clears any pending redo branch.
    pub fn push(&mut self, current: T) {
        if self.undo_stack.len() >= self.max_history {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(current);
        self.redo_stack.clear();
    }

    /// Reverts to the most recent undo snapshot, pushing `current` onto the redo stack.
    pub fn undo(&mut self, current: T) -> Option<T> {
        let prev = self.undo_stack.pop_back()?;
        self.redo_stack.push(current);
        Some(prev)
    }

    /// Reapplies the most recent undone snapshot, pushing `current` back onto the undo stack.
    pub fn redo(&mut self, current: T) -> Option<T> {
        let next = self.redo_stack.pop()?;
        if self.undo_stack.len() >= self.max_history {
            self.undo_stack.pop_front();
        }
        self.undo_stack.push_back(current);
        Some(next)
    }

    /// Returns `true` if at least one state is available to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns `true` if at least one state is available to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Number of stored undo steps.
    pub fn undo_len(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of stored redo steps.
    pub fn redo_len(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clears both undo and redo histories.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}
