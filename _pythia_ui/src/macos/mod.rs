//! macOS: the displays, the privacy permissions, and screenshots, through the
//! Objective-C runtime and Apple's C frameworks. Nothing comes from Apple's
//! SDK at build time: the linker reads the symbol lists in stub-sdk/
//! (README.md, "Stub SDK"), and macOS's own libraries supply the code at run
//! time.

#[macro_use]
mod objc;
mod block;
mod capture;
mod cf;
mod cg;
mod png;

use crate::json::{object, Value};
use crate::{Failure, Screenshot, ScreenshotOptions};
use std::ffi::{c_char, c_int, c_void, CStr};
use std::time::Instant;

// Classes are looked up by name (NSArray), so linking only makes dyld load
// Foundation, which ScreenCaptureKit needs anyway.
#[link(name = "Foundation", kind = "framework")]
extern "C" {}

extern "C" {
  fn sysctlbyname(
    name: *const c_char,
    old: *mut c_void,
    old_len: *mut usize,
    new: *mut c_void,
    new_len: usize,
  ) -> c_int;
}

/// The ops beyond `info`: `screenshot`, where this macOS has a capture path.
pub fn ops() -> Vec<&'static str> {
  if capture::paths().is_empty() {
    Vec::new()
  } else {
    vec!["screenshot"]
  }
}

/// `info`'s fields about the system, its displays and the permissions. It
/// never asks for a permission; a screenshot does.
pub fn info() -> Vec<(&'static str, Value)> {
  objc::autoreleasepool(|| {
    let paths = capture::paths();
    let names = || paths.iter().map(|path| path.name());
    vec![
      (
        "system",
        object([
          ("os", "macos".into()),
          ("version", os_version().into()),
          ("arch", std::env::consts::ARCH.into()),
        ]),
      ),
      (
        "displays",
        Value::Array(cg::displays().iter().map(display_value).collect()),
      ),
      (
        "capture",
        object([
          ("paths", Value::Array(names().map(Value::from).collect())),
          ("default", names().next().into()),
          ("allowed", cg::screen_capture_allowed().into()),
        ]),
      ),
      (
        "input",
        object([("allowed", cg::post_event_allowed().into())]),
      ),
    ]
  })
}

/// Takes a screenshot (capture.rs), until `deadline`.
pub fn screenshot(options: &ScreenshotOptions, deadline: Instant) -> Result<Screenshot, Failure> {
  objc::autoreleasepool(|| capture::screenshot(options, deadline))
}

fn display_value(display: &cg::Display) -> Value {
  object([
    ("id", display.id.into()),
    ("main", display.main.into()),
    ("builtin", display.builtin.into()),
    ("bounds", display.bounds.to_value()),
    (
      "pixels",
      object([
        ("width", display.pixel_width.into()),
        ("height", display.pixel_height.into()),
      ]),
    ),
    ("scale", display.scale().into()),
  ])
}

/// The macOS version, such as "14.5" (sysctl's kern.osproductversion).
fn os_version() -> Option<String> {
  let mut buffer = [0u8; 64];
  let mut len = buffer.len();
  let status = unsafe {
    sysctlbyname(
      c"kern.osproductversion".as_ptr(),
      buffer.as_mut_ptr().cast(),
      &mut len,
      std::ptr::null_mut(),
      0,
    )
  };
  if status != 0 {
    return None;
  }
  let text = CStr::from_bytes_until_nul(&buffer[..len.min(buffer.len())]).ok()?;
  text.to_str().ok().map(str::to_string)
}
