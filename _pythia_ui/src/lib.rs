//! pythia_ui: screen info and screenshots for pythia's `ui` tool, on macOS
//! (Apple Silicon, macOS 11 and later).
//!
//! One dispatcher, [`dispatch`], takes a JSON request and returns a JSON
//! reply. Two thin layers expose it, with the same requests and replies:
//!
//! - **The plugin**, `libpythia_ui.dylib`, which pythia loads into its own
//!   process with ctypes. Its C interface is three functions:
//!   [`pythia_plugin_abi`], [`pythia_plugin_call`] and [`pythia_plugin_free`].
//! - **The program**, `pythia-ui` (src/main.rs), which reads one request per
//!   line on stdin and writes one reply per line on stdout.
//!
//! README.md describes the protocol. The crate builds on other systems too,
//! for its tests: there `info` answers, and screenshots are unsupported.

#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
compile_error!(
  "pythia_ui supports macOS on Apple Silicon only: src/macos/objc.rs assumes arm64's calling \
   convention"
);

mod base64;
pub mod json;
// Only the macOS code waits for completion handlers; the tests run anywhere.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod pending;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(not(target_os = "macos"))]
mod elsewhere;
#[cfg(not(target_os = "macos"))]
use elsewhere as platform;

use json::{object, Value};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

/// The name that `info` gives.
pub const NAME: &str = "pythia_ui";
/// This crate's version, which `info` gives.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The plugin interface's version. It changes only when the three functions'
/// signatures or contracts do; pythia refuses a version that it doesn't know.
pub const ABI_VERSION: u32 = 1;
/// The longest that a request takes: a wait for macOS that would run past it
/// ends the request with a failure instead.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// What a failure reply's `kind` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
  /// The request is malformed: not JSON, without an op, or with a value of
  /// the wrong type.
  Request,
  /// This build or this system can't do it: an op or a field that it doesn't
  /// know, or a capture path that this macOS lacks.
  Unsupported,
  /// macOS doesn't allow it: the user grants it in System Settings.
  Permission,
  /// It was tried, and it failed.
  Failed,
}

impl Kind {
  /// The kind's name in replies.
  pub fn name(self) -> &'static str {
    match self {
      Kind::Request => "request",
      Kind::Unsupported => "unsupported",
      Kind::Permission => "permission",
      Kind::Failed => "failed",
    }
  }
}

/// A failure, as a reply reports it. `error` is text for the model to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
  pub kind: Kind,
  pub error: String,
}

impl Failure {
  pub fn new(kind: Kind, error: impl Into<String>) -> Failure {
    Failure {
      kind,
      error: error.into(),
    }
  }

  pub fn request(error: impl Into<String>) -> Failure {
    Failure::new(Kind::Request, error)
  }

  pub fn unsupported(error: impl Into<String>) -> Failure {
    Failure::new(Kind::Unsupported, error)
  }

  pub fn permission(error: impl Into<String>) -> Failure {
    Failure::new(Kind::Permission, error)
  }

  pub fn failed(error: impl Into<String>) -> Failure {
    Failure::new(Kind::Failed, error)
  }
}

/// A rectangle in points, in macOS's global display coordinates: the origin
/// is the top-left corner of the main display, y grows downward, and input
/// events use the same space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
  pub x: f64,
  pub y: f64,
  pub width: f64,
  pub height: f64,
}

impl Rect {
  /// The rectangle as a JSON object: x, y, width and height.
  pub fn to_value(self) -> Value {
    object([
      ("x", self.x.into()),
      ("y", self.y.into()),
      ("width", self.width.into()),
      ("height", self.height.into()),
    ])
  }
}

/// A way to take a screenshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturePath {
  /// ScreenCaptureKit's `SCScreenshotManager`, from macOS 14.
  ScreenCaptureKit,
  /// CoreGraphics' `CGDisplayCreateImage`, on every macOS, but deprecated
  /// from 14. It never draws the cursor.
  CoreGraphics,
}

impl CapturePath {
  /// The path's name in requests and replies.
  pub fn name(self) -> &'static str {
    match self {
      CapturePath::ScreenCaptureKit => "screencapturekit",
      CapturePath::CoreGraphics => "coregraphics",
    }
  }
}

/// The path that a request's `capture` names: None for `auto`, which takes
/// the first path that this macOS has.
pub fn parse_capture(name: &str) -> Result<Option<CapturePath>, Failure> {
  match name {
    "auto" => Ok(None),
    "screencapturekit" => Ok(Some(CapturePath::ScreenCaptureKit)),
    "coregraphics" => Ok(Some(CapturePath::CoreGraphics)),
    _ => Err(capture_error(&format!("{name:?}"))),
  }
}

fn capture_error(shown: &str) -> Failure {
  Failure::request(format!(
    "screenshot: capture must be \"auto\", \"screencapturekit\" or \"coregraphics\", not {shown}"
  ))
}

/// A screenshot's options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenshotOptions {
  /// The display, by the ID that `info` lists; None is the main display.
  pub display: Option<u32>,
  /// Whether to draw the cursor (the CoreGraphics path never does).
  pub cursor: bool,
  /// The path to take; None takes the first that this macOS has, which
  /// `info` gives as `capture.default`.
  pub capture: Option<CapturePath>,
}

impl Default for ScreenshotOptions {
  fn default() -> ScreenshotOptions {
    ScreenshotOptions {
      display: None,
      cursor: true,
      capture: None,
    }
  }
}

/// A screenshot: a PNG of the display's own pixels, and what maps them to
/// the screen.
#[derive(Clone, Debug, PartialEq)]
pub struct Screenshot {
  /// The display's ID.
  pub display: u32,
  /// The path that took it.
  pub capture: CapturePath,
  /// The image's size, in pixels.
  pub width: usize,
  pub height: usize,
  /// The display's bounds, in points: the image's pixel (px, py) is at the
  /// point (x + px × width ÷ image width, y + py × height ÷ image height).
  pub bounds: Rect,
  /// Whether the cursor is drawn.
  pub cursor: bool,
  /// The image, as PNG.
  pub png: Vec<u8>,
}

/// The ops that this build answers, as `info` lists them.
pub fn ops() -> Vec<&'static str> {
  let mut ops = vec!["info"];
  ops.extend(platform::ops());
  ops
}

/// The fields of `info`'s reply, after `id` and `ok`.
pub fn info() -> Vec<(&'static str, Value)> {
  let mut fields = vec![
    ("name", NAME.into()),
    ("version", VERSION.into()),
    ("abi", ABI_VERSION.into()),
    (
      "ops",
      Value::Array(ops().into_iter().map(Value::from).collect()),
    ),
  ];
  fields.extend(platform::info());
  fields
}

/// Takes a screenshot, within [`REQUEST_TIMEOUT`].
pub fn screenshot(options: &ScreenshotOptions) -> Result<Screenshot, Failure> {
  platform::screenshot(options, Instant::now() + REQUEST_TIMEOUT)
}

/// The fields of a screenshot's reply, after `id` and `ok`. `png` adds the
/// image itself, in base64; the program leaves it out when it writes a file.
pub fn screenshot_fields(shot: &Screenshot, png: bool) -> Vec<(&'static str, Value)> {
  let mut fields = vec![
    ("display", shot.display.into()),
    ("capture", shot.capture.name().into()),
    ("width", shot.width.into()),
    ("height", shot.height.into()),
    ("bounds", shot.bounds.to_value()),
    ("cursor", shot.cursor.into()),
    ("media_type", "image/png".into()),
    ("bytes", shot.png.len().into()),
  ];
  if png {
    fields.push(("png", base64::encode(&shot.png).into()));
  }
  fields
}

/// What answering a request gives: a success reply's fields, or a failure.
type Answer = Result<Vec<(&'static str, Value)>, Failure>;

/// Answers one request, JSON text, with one reply: compact JSON text that
/// never holds a raw newline, so that the program can write it as one line.
/// It never panics: a panic becomes a `failed` reply.
pub fn dispatch(request: &[u8]) -> Vec<u8> {
  dispatch_with(request, answer)
}

/// [`dispatch`], with `answer` making a parsed request's reply.
fn dispatch_with(request: &[u8], answer: impl FnOnce(&Value) -> Answer) -> Vec<u8> {
  let mut id = Value::Null;
  let outcome = catch_unwind(AssertUnwindSafe(|| {
    let request = parse_request(request)?;
    if let Some(given) = request.get("id") {
      id = given.clone();
    }
    answer(&request)
  }))
  .unwrap_or_else(|payload| {
    Err(Failure::failed(format!(
      "the plugin panicked: {}",
      panic_message(&*payload)
    )))
  });
  render_reply(id, outcome)
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
  if let Some(message) = payload.downcast_ref::<&str>() {
    message
  } else if let Some(message) = payload.downcast_ref::<String>() {
    message
  } else {
    "no message"
  }
}

fn parse_request(request: &[u8]) -> Result<Value, Failure> {
  let text =
    std::str::from_utf8(request).map_err(|_| Failure::request("the request isn't UTF-8 text"))?;
  let value = json::parse(text)
    .map_err(|error| Failure::request(format!("the request isn't JSON: {error}")))?;
  match value {
    Value::Object(_) => Ok(value),
    other => Err(Failure::request(format!(
      "the request must be a JSON object, not {}",
      other.kind()
    ))),
  }
}

fn answer(request: &Value) -> Answer {
  let op = match request.get("op") {
    Some(Value::String(op)) => op.as_str(),
    Some(other) => {
      return Err(Failure::request(format!(
        "op must be a string, such as \"info\", not {}",
        other.kind()
      )))
    }
    None => {
      return Err(Failure::request(
        "the request has no op; give \"op\": \"info\", for one",
      ))
    }
  };
  match op {
    "info" => {
      check_fields(request, "info", &[])?;
      Ok(info())
    }
    "screenshot" => {
      let options = screenshot_options(request)?;
      screenshot(&options).map(|shot| screenshot_fields(&shot, true))
    }
    _ => Err(Failure::unsupported(format!(
      "there is no op {op:?} here; {}",
      ops_text()
    ))),
  }
}

/// "the only op is info", or "the ops are info and screenshot".
fn ops_text() -> String {
  match ops().as_slice() {
    [only] => format!("the only op is {only}"),
    ops => format!("the ops are {}", listing(ops)),
  }
}

/// "a", "a and b", "a, b and c".
fn listing(items: &[&str]) -> String {
  match items {
    [] => String::new(),
    [only] => only.to_string(),
    [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
  }
}

/// Refuses a field other than id, op and `known`: one that this build
/// doesn't know, such as a newer pythia's, is unsupported here.
fn check_fields(request: &Value, op: &str, known: &[&str]) -> Result<(), Failure> {
  let Value::Object(members) = request else {
    return Ok(());
  };
  let unknown = members
    .iter()
    .map(|(name, _)| name.as_str())
    .find(|name| !matches!(*name, "id" | "op") && !known.contains(name));
  match unknown {
    None => Ok(()),
    Some(name) => {
      let takes = if known.is_empty() {
        "it takes only id and op".to_string()
      } else {
        format!("it takes {}", listing(known))
      };
      Err(Failure::unsupported(format!(
        "{op}: there is no field {name:?} here; {takes}"
      )))
    }
  }
}

fn screenshot_options(request: &Value) -> Result<ScreenshotOptions, Failure> {
  check_fields(request, "screenshot", &["display", "cursor", "capture"])?;
  let mut options = ScreenshotOptions::default();
  match request.get("display") {
    None | Some(Value::Null) => {}
    Some(value) => {
      let id = value.as_u64().and_then(|id| u32::try_from(id).ok());
      options.display = Some(id.ok_or_else(|| {
        Failure::request(format!(
          "screenshot: display must be a display ID from info, such as 1, not {}",
          shown(value)
        ))
      })?);
    }
  }
  match request.get("cursor") {
    None => {}
    Some(Value::Bool(cursor)) => options.cursor = *cursor,
    Some(value) => {
      return Err(Failure::request(format!(
        "screenshot: cursor must be true or false, not {}",
        shown(value)
      )))
    }
  }
  match request.get("capture") {
    None => {}
    Some(Value::String(name)) => options.capture = parse_capture(name)?,
    Some(value) => return Err(capture_error(&shown(value))),
  }
  Ok(options)
}

/// A value as JSON text, for messages, cut after 40 characters.
fn shown(value: &Value) -> String {
  let text = String::from_utf8_lossy(&value.to_vec()).into_owned();
  match text.char_indices().nth(40) {
    Some((cut, _)) => format!("{}…", &text[..cut]),
    None => text,
  }
}

fn render_reply(id: Value, outcome: Answer) -> Vec<u8> {
  let mut members = vec![("id", id)];
  match outcome {
    Ok(fields) => {
      members.push(("ok", true.into()));
      members.extend(fields);
    }
    Err(failure) => members.extend([
      ("ok", false.into()),
      ("kind", failure.kind.name().into()),
      ("error", failure.error.into()),
    ]),
  }
  object(members).to_vec()
}

// The plugin's C interface (README.md, "The plugin interface").

/// The plugin interface's version, [`ABI_VERSION`].
#[no_mangle]
pub extern "C" fn pythia_plugin_abi() -> u32 {
  ABI_VERSION
}

/// Answers one request: sets `*reply` and `*reply_len` to the reply, which
/// the caller frees with [`pythia_plugin_free`], and returns 0. The reply may
/// report a failure (`"ok": false`). It returns 1, setting nothing, only when
/// `reply` or `reply_len` is null, so that no reply can be given. A panic
/// becomes an error reply: it never unwinds into the caller.
///
/// # Safety
/// `request` points to `request_len` readable bytes, or is null with a length
/// of 0; `reply` and `reply_len` are null or writable.
#[no_mangle]
pub unsafe extern "C" fn pythia_plugin_call(
  request: *const u8,
  request_len: usize,
  reply: *mut *mut u8,
  reply_len: *mut usize,
) -> i32 {
  if reply.is_null() || reply_len.is_null() {
    return 1;
  }
  quiet_panics();
  let bytes = if request.is_null() && request_len > 0 {
    render_reply(
      Value::Null,
      Err(Failure::request(format!(
        "the request is a null pointer with a length of {request_len}"
      ))),
    )
  } else if request.is_null() {
    dispatch(&[])
  } else {
    dispatch(std::slice::from_raw_parts(request, request_len))
  };
  let bytes = bytes.into_boxed_slice();
  *reply_len = bytes.len();
  *reply = Box::into_raw(bytes).cast::<u8>();
  0
}

/// Frees a reply of [`pythia_plugin_call`]'s; a null reply is ignored.
///
/// # Safety
/// `reply` and `reply_len` are what one call set, and are freed only once.
#[no_mangle]
pub unsafe extern "C" fn pythia_plugin_free(reply: *mut u8, reply_len: usize) {
  if !reply.is_null() {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
      reply, reply_len,
    )));
  }
}

/// A panic in the plugin prints nothing, since the host's terminal is the
/// CLI's: its message goes into the error reply instead. The hook is this
/// library's own (a cdylib has its own copy of std), not the host's. Tests
/// keep the default hook, so that their failures print.
fn quiet_panics() {
  #[cfg(not(test))]
  {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::panic::set_hook(Box::new(|_| {})));
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn parsed(reply: &[u8]) -> Value {
    assert!(!reply.contains(&b'\n'), "a raw newline in {reply:?}");
    json::parse(std::str::from_utf8(reply).unwrap()).unwrap()
  }

  fn call(request: &str) -> Value {
    parsed(&dispatch(request.as_bytes()))
  }

  fn text<'a>(reply: &'a Value, name: &str) -> &'a str {
    let field = reply.get(name).and_then(Value::as_str);
    field.unwrap_or_else(|| panic!("no {name} in {reply:?}"))
  }

  /// A failure reply's kind and error.
  fn failure(reply: &Value) -> (&str, &str) {
    assert_eq!(reply.get("ok"), Some(&Value::Bool(false)), "{reply:?}");
    (text(reply, "kind"), text(reply, "error"))
  }

  #[test]
  fn info_names_the_plugin_and_its_ops() {
    let reply = call(r#"{"id": 7, "op": "info"}"#);
    assert_eq!(reply.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(text(&reply, "name"), "pythia_ui");
    assert_eq!(text(&reply, "version"), env!("CARGO_PKG_VERSION"));
    assert_eq!(reply.get("abi").and_then(Value::as_u64), Some(1));
    let Some(Value::Array(ops)) = reply.get("ops") else {
      panic!("no ops in {reply:?}")
    };
    assert_eq!(ops.first(), Some(&Value::from("info")));
    let system = reply.get("system").unwrap();
    assert_eq!(text(system, "os"), std::env::consts::OS);
    for field in ["displays", "capture", "input"] {
      assert!(reply.get(field).is_some(), "no {field} in {reply:?}");
    }
  }

  #[test]
  fn the_id_is_echoed_as_given_and_first() {
    for id in ["12345678901234567890", r#""a\nb""#, "null", r#"[1,{}]"#] {
      let reply = dispatch(format!(r#"{{"op": "info", "id": {id}}}"#).as_bytes());
      let expected = format!(r#"{{"id":{id},"ok":true,"#);
      assert!(
        reply.starts_with(expected.as_bytes()),
        "{}",
        String::from_utf8_lossy(&reply)
      );
    }
    assert!(dispatch(br#"{"op":"info"}"#).starts_with(br#"{"id":null,"ok":true,"#));
    // A failure keeps it too.
    let reply = call(r#"{"id": "x", "op": "click"}"#);
    assert_eq!(text(&reply, "id"), "x");
  }

  #[test]
  fn malformed_requests_are_refused_with_what_to_give() {
    let ops = ops_text();
    let cases = [
      (
        "",
        "request",
        "the request isn't JSON: expected a value at the end".to_string(),
      ),
      (
        r#"{"op": "info""#,
        "request",
        "the request isn't JSON: expected ',' or '}' at the end".into(),
      ),
      (
        r#"["info"]"#,
        "request",
        "the request must be a JSON object, not an array".into(),
      ),
      (
        "{}",
        "request",
        r#"the request has no op; give "op": "info", for one"#.into(),
      ),
      (
        r#"{"op": 1}"#,
        "request",
        r#"op must be a string, such as "info", not a number"#.into(),
      ),
      (
        r#"{"op": "click"}"#,
        "unsupported",
        format!(r#"there is no op "click" here; {ops}"#),
      ),
      (
        r#"{"op": "info", "x": 1}"#,
        "unsupported",
        r#"info: there is no field "x" here; it takes only id and op"#.into(),
      ),
      (
        r#"{"op": "screenshot", "max_long_edge": 1568}"#,
        "unsupported",
        r#"screenshot: there is no field "max_long_edge" here; it takes display, cursor and capture"#
          .into(),
      ),
      (
        r#"{"op": "screenshot", "display": -1}"#,
        "request",
        "screenshot: display must be a display ID from info, such as 1, not -1".into(),
      ),
      (
        r#"{"op": "screenshot", "display": 4294967296}"#,
        "request",
        "screenshot: display must be a display ID from info, such as 1, not 4294967296".into(),
      ),
      (
        r#"{"op": "screenshot", "display": "main"}"#,
        "request",
        r#"screenshot: display must be a display ID from info, such as 1, not "main""#.into(),
      ),
      (
        r#"{"op": "screenshot", "cursor": "yes"}"#,
        "request",
        r#"screenshot: cursor must be true or false, not "yes""#.into(),
      ),
      (
        r#"{"op": "screenshot", "capture": "fast"}"#,
        "request",
        r#"screenshot: capture must be "auto", "screencapturekit" or "coregraphics", not "fast""#
          .into(),
      ),
      (
        r#"{"op": "screenshot", "capture": [1]}"#,
        "request",
        r#"screenshot: capture must be "auto", "screencapturekit" or "coregraphics", not [1]"#
          .into(),
      ),
    ];
    for (request, kind, error) in cases {
      let reply = call(request);
      assert_eq!(failure(&reply), (kind, error.as_str()), "{request}");
    }
    let reply = parsed(&dispatch(b"{\"op\": \"\xff\"}"));
    assert_eq!(failure(&reply), ("request", "the request isn't UTF-8 text"));
    // A long value is cut.
    let reply = call(&format!(
      r#"{{"op": "screenshot", "cursor": "{}"}}"#,
      "y".repeat(100)
    ));
    assert!(failure(&reply)
      .1
      .ends_with(&format!("\"{}…", "y".repeat(39))));
  }

  #[cfg(not(target_os = "macos"))]
  #[test]
  fn a_screenshot_needs_macos() {
    let reply = call(r#"{"op": "screenshot", "capture": "coregraphics"}"#);
    let error = format!(
      "screenshot: this build is for {}, and screenshots need macOS",
      std::env::consts::OS
    );
    assert_eq!(failure(&reply), ("unsupported", error.as_str()));
    assert_eq!(ops(), ["info"]);
  }

  #[test]
  fn a_panic_becomes_a_failed_reply_that_keeps_the_id() {
    let reply = parsed(&dispatch_with(br#"{"id": 3, "op": "info"}"#, |_| {
      panic!("probe {}\nsecond line", 1)
    }));
    assert_eq!(reply.get("id"), Some(&Value::Number("3".into())));
    assert_eq!(
      failure(&reply),
      ("failed", "the plugin panicked: probe 1\nsecond line")
    );
    let reply = parsed(&dispatch_with(b"{}", |_| std::panic::panic_any(5)));
    assert_eq!(
      failure(&reply),
      ("failed", "the plugin panicked: no message")
    );
  }

  #[test]
  fn a_screenshot_reply_maps_pixels_to_points_and_carries_the_png() {
    let shot = Screenshot {
      display: 2,
      capture: CapturePath::CoreGraphics,
      width: 3024,
      height: 1964,
      bounds: Rect {
        x: -1512.0,
        y: 0.0,
        width: 1512.0,
        height: 982.0,
      },
      cursor: false,
      png: b"\x89PNG\r\n\x1a\n".to_vec(),
    };
    let reply = render_reply(1u32.into(), Ok(screenshot_fields(&shot, true)));
    assert_eq!(
      String::from_utf8(reply).unwrap(),
      concat!(
        r#"{"id":1,"ok":true,"display":2,"capture":"coregraphics","width":3024,"#,
        r#""height":1964,"bounds":{"x":-1512,"y":0,"width":1512,"height":982},"#,
        r#""cursor":false,"media_type":"image/png","bytes":8,"png":"iVBORw0KGgo="}"#
      )
    );
    let without = screenshot_fields(&shot, false);
    assert!(!without.iter().any(|(name, _)| *name == "png"));
  }

  #[test]
  fn capture_names_parse() {
    assert_eq!(parse_capture("auto"), Ok(None));
    assert_eq!(
      parse_capture("screencapturekit"),
      Ok(Some(CapturePath::ScreenCaptureKit))
    );
    assert_eq!(
      parse_capture("coregraphics"),
      Ok(Some(CapturePath::CoreGraphics))
    );
    assert_eq!(
      parse_capture("CoreGraphics").unwrap_err().kind,
      Kind::Request
    );
    for path in [CapturePath::ScreenCaptureKit, CapturePath::CoreGraphics] {
      assert_eq!(parse_capture(path.name()), Ok(Some(path)));
    }
  }

  #[test]
  fn listings_read_as_english() {
    assert_eq!(listing(&[]), "");
    assert_eq!(listing(&["a"]), "a");
    assert_eq!(listing(&["a", "b"]), "a and b");
    assert_eq!(listing(&["a", "b", "c"]), "a, b and c");
  }

  /// Calls the plugin as a C caller would, and frees the reply.
  fn plugin_call(request: *const u8, len: usize) -> Value {
    let (mut reply, mut reply_len) = (std::ptr::null_mut(), 0usize);
    let status = unsafe { pythia_plugin_call(request, len, &mut reply, &mut reply_len) };
    assert_eq!(status, 0);
    let bytes = unsafe { std::slice::from_raw_parts(reply, reply_len) }.to_vec();
    unsafe { pythia_plugin_free(reply, reply_len) };
    parsed(&bytes)
  }

  #[test]
  fn the_plugin_functions_answer_and_free() {
    assert_eq!(pythia_plugin_abi(), 1);
    let request = br#"{"id": 1, "op": "info"}"#;
    let reply = plugin_call(request.as_ptr(), request.len());
    assert_eq!(reply.get("ok"), Some(&Value::Bool(true)));
    // A null request: empty when its length is 0, and refused otherwise.
    let reply = plugin_call(std::ptr::null(), 0);
    assert_eq!(
      failure(&reply),
      (
        "request",
        "the request isn't JSON: expected a value at the end"
      )
    );
    let reply = plugin_call(std::ptr::null(), 5);
    assert_eq!(
      failure(&reply),
      (
        "request",
        "the request is a null pointer with a length of 5"
      )
    );
    // Without a place for the reply: 1, and nothing is set.
    let (mut reply, mut reply_len) = (std::ptr::null_mut(), 0usize);
    let call = |reply, reply_len| unsafe {
      pythia_plugin_call(request.as_ptr(), request.len(), reply, reply_len)
    };
    assert_eq!(call(std::ptr::null_mut(), &mut reply_len), 1);
    assert_eq!(call(&mut reply, std::ptr::null_mut()), 1);
    assert!(reply.is_null());
    assert_eq!(reply_len, 0);
    unsafe { pythia_plugin_free(std::ptr::null_mut(), 0) };
  }
}
