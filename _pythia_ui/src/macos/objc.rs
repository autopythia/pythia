//! Just enough of the Objective-C runtime (libobjc) to call ScreenCaptureKit
//! and Foundation.
//!
//! From research-gui (`_glomery/src/objc.rs`, commit a5099c9), changed for a
//! library that runs inside another program: a missing class is `None`
//! rather than an exit; no class definitions, since nothing here needs a
//! delegate; an autorelease pool that's drained after a panic too; and
//! [`Retained`], strings, and `u32` results (`CGDirectDisplayID`).
//!
//! Rules the compiler can't fully check:
//! - `msg_send!` calls `objc_msgSend` through a pointer cast to the method's
//!   exact C signature. arm64 passes variadic and fixed arguments
//!   differently, so a wrong signature is undefined behaviour, not just a
//!   wrong value: use the argument and return types from Apple's headers.
//!   `ObjcType` rejects the most common slip, a bare integer literal (`i32`).
//! - Memory is managed by hand (no ARC): a [`Retained`] releases what it owns.
//! - An Objective-C exception can't be caught here: it ends the process.
//!
//! arm64 only: x86_64 would need `BOOL` as `i8` and `objc_msgSend_stret` for
//! methods that return large structs.

use std::ffi::{c_char, c_void, CStr};

/// An Objective-C object, only ever used behind a pointer.
#[repr(C)]
pub struct Object {
  _private: [u8; 0],
}

/// `id`: a pointer to any object, including a class.
pub type Id = *mut Object;

/// `SEL`: a registered method name.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Sel(*const c_void);

#[link(name = "objc")]
extern "C" {
  fn objc_msgSend();
  fn objc_getClass(name: *const c_char) -> Id;
  fn sel_registerName(name: *const c_char) -> Sel;
  fn objc_retain(object: Id) -> Id;
  fn objc_release(object: Id);
  fn objc_autoreleasePoolPush() -> *mut c_void;
  fn objc_autoreleasePoolPop(pool: *mut c_void);
}

/// C types that can be passed to or returned from `msg_send!`.
///
/// # Safety
/// Implement only for types with a C equivalent. `i32` is deliberately
/// missing: Foundation's integers are 64-bit `NSInteger`/`NSUInteger`. `u32`
/// is here for `CGDirectDisplayID`.
pub unsafe trait ObjcType {}
unsafe impl ObjcType for () {}
unsafe impl ObjcType for bool {}
unsafe impl ObjcType for u32 {}
unsafe impl ObjcType for isize {}
unsafe impl ObjcType for usize {}
unsafe impl ObjcType for Id {}
unsafe impl ObjcType for Sel {}
unsafe impl ObjcType for *const c_void {}

/// The selector named by a NUL-terminated string; see `sel!`.
pub fn sel(name: &str) -> Sel {
  assert!(name.ends_with('\0'));
  unsafe { sel_registerName(name.as_ptr().cast()) }
}

/// The class of that name, if one is loaded.
pub fn class(name: &CStr) -> Option<Id> {
  let class = unsafe { objc_getClass(name.as_ptr()) };
  (!class.is_null()).then_some(class)
}

/// Argument tuples that `send` can pass through `objc_msgSend`.
pub trait MessageArgs {
  /// # Safety
  /// See `send`.
  unsafe fn send_to<R: ObjcType>(self, receiver: Id, sel: Sel) -> R;
}

macro_rules! message_args {
  ($($arg:ident: $ty:ident),*) => {
    impl<$($ty: ObjcType),*> MessageArgs for ($($ty,)*) {
      #[inline]
      unsafe fn send_to<R: ObjcType>(self, receiver: Id, sel: Sel) -> R {
        let ($($arg,)*) = self;
        let imp = std::mem::transmute::<
          unsafe extern "C" fn(),
          unsafe extern "C" fn(Id, Sel $(, $ty)*) -> R,
        >(objc_msgSend);
        imp(receiver, sel $(, $arg)*)
      }
    }
  };
}

message_args!();
message_args!(a: A);
message_args!(a: A, b: B);
message_args!(a: A, b: B, c: C);

/// Sends a message; `msg_send!` builds the selector and the argument tuple.
///
/// # Safety
/// `receiver` must be nil or a live object that responds to `sel`, and the
/// argument and return types must match the method's C signature exactly.
#[inline]
pub unsafe fn send<A: MessageArgs, R: ObjcType>(receiver: Id, sel: Sel, args: A) -> R {
  args.send_to(receiver, sel)
}

/// `sel!(name)` or `sel!(name:with:)`: a selector, spelled as in Objective-C.
macro_rules! sel {
  ($name:ident) => {
    $crate::macos::objc::sel(concat!(stringify!($name), "\0"))
  };
  ($($part:ident :)+) => {
    $crate::macos::objc::sel(concat!($(stringify!($part), ":",)+ "\0"))
  };
}

/// `msg_send![receiver, method]` or `msg_send![receiver, part: arg, part: arg]`
/// is Objective-C's `[receiver part:arg part:arg]`; the selector is built from
/// the parts, so it always has one colon per argument. Give the result a type:
/// `let count: usize = msg_send![...]`, `let () = msg_send![...]`.
macro_rules! msg_send {
  [$receiver:expr, $name:ident $(,)?] => {
    $crate::macos::objc::send($receiver, sel!($name), ())
  };
  [$receiver:expr, $($part:ident : $arg:expr),+ $(,)?] => {
    $crate::macos::objc::send($receiver, sel!($($part:)+), ($($arg,)+))
  };
}

/// An object that we own, released when dropped.
pub struct Retained(Id);

// The runtime's retain counts are thread-safe, and completion handlers hand
// objects over from their own queues.
unsafe impl Send for Retained {}

impl Retained {
  /// Takes over an object that we own already, from `alloc` and `init`,
  /// `new`, or `copy`; nil stays nil.
  ///
  /// # Safety
  /// `object` is nil, or a live object that no one else will release.
  pub unsafe fn from_owned(object: Id) -> Retained {
    Retained(object)
  }

  /// Retains `object`; nil stays nil. It never panics: completion handlers
  /// call it.
  ///
  /// # Safety
  /// `object` is nil or a live object.
  pub unsafe fn retain(object: Id) -> Retained {
    if object.is_null() {
      Retained(object)
    } else {
      Retained(objc_retain(object))
    }
  }

  pub fn get(&self) -> Id {
    self.0
  }

  pub fn is_null(&self) -> bool {
    self.0.is_null()
  }
}

impl Drop for Retained {
  fn drop(&mut self) {
    if !self.0.is_null() {
      unsafe { objc_release(self.0) }
    }
  }
}

/// Runs `f` inside an autorelease pool, like `@autoreleasepool { ... }`. The
/// pool is drained after a panic too, which the plugin turns into a reply.
pub fn autoreleasepool<T>(f: impl FnOnce() -> T) -> T {
  struct Pool(*mut c_void);
  impl Drop for Pool {
    fn drop(&mut self) {
      unsafe { objc_autoreleasePoolPop(self.0) }
    }
  }
  let _pool = Pool(unsafe { objc_autoreleasePoolPush() });
  f()
}

/// The text of an NSString; None for nil.
///
/// # Safety
/// `string` is nil or a live NSString.
pub unsafe fn string(string: Id) -> Option<String> {
  if string.is_null() {
    return None;
  }
  let utf8: *const c_void = msg_send![string, UTF8String];
  if utf8.is_null() {
    return None;
  }
  Some(CStr::from_ptr(utf8.cast()).to_string_lossy().into_owned())
}
