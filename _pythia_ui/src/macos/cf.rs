//! CoreFoundation: the few calls that the other modules need, and an owner
//! that releases a CF object. The functions are listed in
//! stub-sdk/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation.tbd.

use std::ffi::{c_char, c_void};

/// `CFIndex`.
pub type Index = isize;
/// `kCFStringEncodingUTF8`.
pub const UTF8: u32 = 0x0800_0100;

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
  fn CFRetain(object: *const c_void) -> *const c_void;
  fn CFRelease(object: *const c_void);
  pub fn CFDataCreateMutable(allocator: *const c_void, capacity: Index) -> *mut c_void;
  pub fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
  pub fn CFDataGetLength(data: *const c_void) -> Index;
  pub fn CFStringCreateWithCString(
    allocator: *const c_void,
    text: *const c_char,
    encoding: u32,
  ) -> *const c_void;
}

/// A CF object that we own, released when dropped. A CGImage, and the other
/// CoreGraphics and ImageIO types, are CF objects too.
pub struct Owned(*const c_void);

// CF retain counts are thread-safe, and completion handlers hand images over
// from their own queues.
unsafe impl Send for Owned {}

impl Owned {
  /// Takes over an object from a function named Create or Copy, which we own
  /// already; null stays null.
  ///
  /// # Safety
  /// `object` is null, or a live CF object that no one else will release.
  pub unsafe fn from_create(object: *const c_void) -> Owned {
    Owned(object)
  }

  /// Retains `object`; null stays null. It never panics: completion
  /// handlers call it.
  ///
  /// # Safety
  /// `object` is null or a live CF object.
  pub unsafe fn retain(object: *const c_void) -> Owned {
    if !object.is_null() {
      CFRetain(object);
    }
    Owned(object)
  }

  pub fn get(&self) -> *const c_void {
    self.0
  }

  pub fn is_null(&self) -> bool {
    self.0.is_null()
  }
}

impl Drop for Owned {
  fn drop(&mut self) {
    if !self.0.is_null() {
      unsafe { CFRelease(self.0) }
    }
  }
}
