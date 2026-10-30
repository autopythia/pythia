//! CoreGraphics: the displays, and the privacy permissions (screen recording,
//! posting input events). The functions are listed in
//! stub-sdk/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics.tbd.

use super::cf;
use crate::Rect;
use std::ffi::c_void;

/// `CGDirectDisplayID`.
pub type DisplayId = u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
  x: f64,
  y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
  width: f64,
  height: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
  origin: CGPoint,
  size: CGSize,
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
  fn CGGetActiveDisplayList(max: u32, displays: *mut DisplayId, count: *mut u32) -> i32;
  fn CGMainDisplayID() -> DisplayId;
  fn CGDisplayBounds(display: DisplayId) -> CGRect;
  /// `boolean_t`, an int on arm64.
  fn CGDisplayIsBuiltin(display: DisplayId) -> u32;
  fn CGDisplayCopyDisplayMode(display: DisplayId) -> *mut c_void;
  fn CGDisplayModeGetPixelWidth(mode: *mut c_void) -> usize;
  fn CGDisplayModeGetPixelHeight(mode: *mut c_void) -> usize;
  fn CGDisplayModeRelease(mode: *mut c_void);
  fn CGImageGetWidth(image: *const c_void) -> usize;
  fn CGImageGetHeight(image: *const c_void) -> usize;
  fn CGPreflightScreenCaptureAccess() -> bool;
  fn CGRequestScreenCaptureAccess() -> bool;
  fn CGPreflightPostEventAccess() -> bool;
}

/// The most displays listed; macOS drives far fewer.
const MAX_DISPLAYS: usize = 32;

/// An active display.
pub struct Display {
  pub id: DisplayId,
  pub main: bool,
  pub builtin: bool,
  /// In points, in the global display coordinates.
  pub bounds: Rect,
  /// The current mode's size in pixels (0 if macOS gives no mode). With a
  /// scaled resolution, it's the backing store's size, which the screen
  /// shows scaled.
  pub pixel_width: usize,
  pub pixel_height: usize,
}

impl Display {
  /// Pixels per point: 2 on a Retina display.
  pub fn scale(&self) -> f64 {
    if self.bounds.width > 0.0 && self.pixel_width > 0 {
      self.pixel_width as f64 / self.bounds.width
    } else {
      1.0
    }
  }
}

/// The active displays, in CoreGraphics' order; none when there's no window
/// server to ask, as over ssh.
pub fn displays() -> Vec<Display> {
  let mut ids = [0 as DisplayId; MAX_DISPLAYS];
  let mut count = 0u32;
  let status = unsafe { CGGetActiveDisplayList(MAX_DISPLAYS as u32, ids.as_mut_ptr(), &mut count) };
  if status != 0 {
    return Vec::new();
  }
  let main = unsafe { CGMainDisplayID() };
  let count = (count as usize).min(MAX_DISPLAYS);
  ids[..count].iter().map(|&id| display(id, main)).collect()
}

fn display(id: DisplayId, main: DisplayId) -> Display {
  let bounds = unsafe { CGDisplayBounds(id) };
  let mode = unsafe { CGDisplayCopyDisplayMode(id) };
  let (pixel_width, pixel_height) = if mode.is_null() {
    (0, 0)
  } else {
    unsafe {
      let size = (
        CGDisplayModeGetPixelWidth(mode),
        CGDisplayModeGetPixelHeight(mode),
      );
      CGDisplayModeRelease(mode);
      size
    }
  };
  Display {
    id,
    main: id == main,
    builtin: unsafe { CGDisplayIsBuiltin(id) } != 0,
    bounds: Rect {
      x: bounds.origin.x,
      y: bounds.origin.y,
      width: bounds.size.width,
      height: bounds.size.height,
    },
    pixel_width,
    pixel_height,
  }
}

/// A CGImage's size, in pixels.
pub fn image_size(image: &cf::Owned) -> (usize, usize) {
  unsafe { (CGImageGetWidth(image.get()), CGImageGetHeight(image.get())) }
}

/// Whether this process may record the screen; it never asks the user.
pub fn screen_capture_allowed() -> bool {
  unsafe { CGPreflightScreenCaptureAccess() }
}

/// Asks for screen recording: the first time, macOS shows its prompt and
/// lists the app in System Settings. The answer applies once the app
/// restarts, so this never waits for it.
pub fn request_screen_capture() {
  unsafe { CGRequestScreenCaptureAccess() };
}

/// Whether this process may post input events (Accessibility).
pub fn post_event_allowed() -> bool {
  unsafe { CGPreflightPostEventAccess() }
}
