//! Systems other than macOS. The crate builds here for its tests, and for
//! trying the plugin interface: `info` answers, with no displays, and
//! screenshots are unsupported.

use crate::json::{object, Value};
use crate::{Failure, Screenshot, ScreenshotOptions};
use std::env::consts::{ARCH, OS};
use std::time::Instant;

/// The ops beyond `info`: none.
pub fn ops() -> Vec<&'static str> {
  Vec::new()
}

/// `info`'s fields about the system.
pub fn info() -> Vec<(&'static str, Value)> {
  vec![
    (
      "system",
      object([
        ("os", OS.into()),
        ("version", Value::Null),
        ("arch", ARCH.into()),
      ]),
    ),
    ("displays", Value::Array(Vec::new())),
    (
      "capture",
      object([
        ("paths", Value::Array(Vec::new())),
        ("default", Value::Null),
        ("allowed", false.into()),
      ]),
    ),
    ("input", object([("allowed", false.into())])),
  ]
}

pub fn screenshot(_options: &ScreenshotOptions, _deadline: Instant) -> Result<Screenshot, Failure> {
  Err(Failure::unsupported(format!(
    "screenshot: this build is for {OS}, and screenshots need macOS"
  )))
}
