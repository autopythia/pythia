//! Objective-C blocks for completion handlers, laid out as clang lays out a
//! block literal (the Blocks ABI's `Block_literal_1`), with one captured
//! pointer: a reference to the handler's [`Pending`].
//!
//! A block is made on the stack, and passed to an API that calls it later;
//! such an API copies it to the heap (`Block_copy`) before it returns. The
//! copy is a plain memory copy, since the descriptor has no copy or dispose
//! helpers: what's captured is a pointer, not an object. The handler takes
//! the pointer out with an atomic swap, so that a block called twice would
//! find nothing the second time.
//!
//! A handler that's never called leaks its reference, rather than risk a late
//! call into freed memory.

use super::objc::Id;
use crate::pending::Pending;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

#[allow(non_upper_case_globals)]
extern "C" {
  /// The class of blocks on the stack, from libsystem_blocks through
  /// libSystem; Block.h declares it as `void *[32]`.
  static _NSConcreteStackBlock: [*const c_void; 32];
}

/// Set in a block's flags when its descriptor has a type encoding.
const BLOCK_HAS_SIGNATURE: i32 = 1 << 30;

/// A block's descriptor: static data, one per block signature.
#[repr(C)]
pub struct Descriptor {
  reserved: usize,
  size: usize,
  signature: *const c_char,
}

// A descriptor is never written once made, and blocks on any thread read it.
unsafe impl Sync for Descriptor {}

/// The descriptor of this module's blocks, with `signature`, the block's type
/// encoding, such as `v24@?0@8@16` for `void (^)(id, id)`.
pub const fn descriptor(signature: &'static CStr) -> Descriptor {
  Descriptor {
    reserved: 0,
    size: std::mem::size_of::<Block>(),
    signature: signature.as_ptr(),
  }
}

/// A block literal that captures one pointer, a [`Pending`] reference.
#[repr(C)]
pub struct Block {
  isa: *const c_void,
  flags: i32,
  reserved: i32,
  invoke: *const c_void,
  descriptor: *const Descriptor,
  context: AtomicPtr<c_void>,
}

// The layout that the Blocks ABI gives such a block on arm64.
const _: () = assert!(std::mem::size_of::<Block>() == 40);
const _: () = assert!(std::mem::offset_of!(Block, context) == 32);

impl Block {
  /// A block on the stack that calls `invoke`, an
  /// `unsafe extern "C" fn(*mut Block, ...)` that matches `descriptor`'s
  /// signature, with a reference to `pending`.
  pub fn new<T>(
    invoke: *const c_void,
    descriptor: &'static Descriptor,
    pending: &Arc<Pending<T>>,
  ) -> Block {
    // Taking an extern static's address needed `unsafe` before Rust 1.82.
    #[allow(unused_unsafe)]
    let isa = unsafe { std::ptr::addr_of!(_NSConcreteStackBlock) };
    Block {
      isa: isa.cast(),
      flags: BLOCK_HAS_SIGNATURE,
      reserved: 0,
      invoke,
      descriptor,
      context: AtomicPtr::new(Pending::into_context(pending).cast_mut()),
    }
  }

  /// The block, as a message's argument.
  pub fn as_id(&mut self) -> Id {
    (self as *mut Block).cast()
  }

  /// Takes back the reference that [`Block::new`] captured: the first call
  /// gets it, and any later one None. It never panics, since invoke
  /// functions call it.
  ///
  /// # Safety
  /// `block` is a block that `new` made with a `Pending<T>`, or a heap copy
  /// of one.
  pub unsafe fn take<T>(block: *mut Block) -> Option<Arc<Pending<T>>> {
    let context: &AtomicPtr<c_void> = &(*block).context;
    let taken = context.swap(std::ptr::null_mut(), Ordering::AcqRel);
    if taken.is_null() {
      None
    } else {
      Some(Pending::from_context(taken.cast_const()))
    }
  }
}
