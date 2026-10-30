//! A result that another thread hands over, such as a completion handler on
//! a queue of ScreenCaptureKit's, waited for with a deadline.
//!
//! The handler holds its own reference to the state ([`Pending::into_context`],
//! then [`Pending::from_context`] once in the handler). So when a wait times
//! out and its caller moves on, a late result still lands in state that
//! lives, never in freed memory, and is dropped with it.

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

pub(crate) struct Pending<T> {
  slot: Mutex<Option<T>>,
  ready: Condvar,
}

impl<T> Pending<T> {
  pub(crate) fn new() -> Arc<Pending<T>> {
    Arc::new(Pending {
      slot: Mutex::new(None),
      ready: Condvar::new(),
    })
  }

  /// A reference for a callback, as a pointer: the callback takes it back
  /// with [`Pending::from_context`], once.
  pub(crate) fn into_context(this: &Arc<Pending<T>>) -> *const c_void {
    Arc::into_raw(Arc::clone(this)).cast()
  }

  /// Takes back a reference that [`Pending::into_context`] gave away.
  ///
  /// # Safety
  /// `context` came from `into_context` for the same `T`, and is taken back
  /// only once.
  pub(crate) unsafe fn from_context(context: *const c_void) -> Arc<Pending<T>> {
    Arc::from_raw(context.cast())
  }

  /// Hands the result over. It never panics, so a callback that a panic
  /// can't unwind out of may call it.
  pub(crate) fn finish(&self, value: T) {
    let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
    *slot = Some(value);
    drop(slot);
    self.ready.notify_all();
  }

  /// The result once it's handed over, or None if `timeout` passes first.
  pub(crate) fn wait(&self, timeout: Duration) -> Option<T> {
    let deadline = Instant::now() + timeout;
    let mut slot = self.slot.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
      if let Some(value) = slot.take() {
        return Some(value);
      }
      let left = deadline.saturating_duration_since(Instant::now());
      if left.is_zero() {
        return None;
      }
      slot = self
        .ready
        .wait_timeout(slot, left)
        .unwrap_or_else(PoisonError::into_inner)
        .0;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::atomic::{AtomicUsize, Ordering};
  use std::thread;

  /// Counts its drops.
  struct Counted(Arc<AtomicUsize>);

  impl Drop for Counted {
    fn drop(&mut self) {
      self.0.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn a_result_handed_over_before_the_wait_is_taken() {
    let pending = Pending::new();
    pending.finish(7);
    assert_eq!(pending.wait(Duration::ZERO), Some(7));
  }

  #[test]
  fn a_wait_gets_a_result_from_another_thread() {
    let pending = Pending::new();
    let context = Pending::into_context(&pending) as usize;
    let handler = thread::spawn(move || {
      thread::sleep(Duration::from_millis(50));
      let pending = unsafe { Pending::<u32>::from_context(context as *const c_void) };
      pending.finish(42);
    });
    assert_eq!(pending.wait(Duration::from_secs(5)), Some(42));
    handler.join().unwrap();
    assert_eq!(Arc::strong_count(&pending), 1);
  }

  #[test]
  fn a_wait_times_out() {
    let pending = Pending::<u32>::new();
    let started = Instant::now();
    assert_eq!(pending.wait(Duration::from_millis(30)), None);
    assert!(started.elapsed() >= Duration::from_millis(30));
  }

  #[test]
  fn a_late_result_lands_in_live_state_and_is_dropped_with_it() {
    let drops = Arc::new(AtomicUsize::new(0));
    let pending = Pending::<Counted>::new();
    let context = Pending::into_context(&pending) as usize;
    assert!(pending.wait(Duration::from_millis(10)).is_none());
    drop(pending); // the caller gave up
    let late = Counted(Arc::clone(&drops));
    thread::spawn(move || {
      let pending = unsafe { Pending::<Counted>::from_context(context as *const c_void) };
      pending.finish(late);
      // The handler's reference was the last: the result goes with it.
    })
    .join()
    .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
  }
}
