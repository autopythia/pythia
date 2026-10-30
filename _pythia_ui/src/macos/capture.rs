//! Screenshots, by one of two paths. A request can name one; otherwise the
//! first that this macOS has is taken. What it has is checked at run time,
//! never by its version number:
//!
//! - **ScreenCaptureKit**: `SCScreenshotManager`, macOS 14 and later. The
//!   framework arrived in 12.3, so it's loaded with `dlopen` rather than
//!   linked, and the plugin and the program still load on macOS 11.
//! - **CoreGraphics**: `CGDisplayCreateImage`, on every macOS, but deprecated
//!   from 14, where ScreenCaptureKit takes over. It's looked up with `dlsym`,
//!   so that a macOS without it would still load them. It never draws the
//!   cursor.
//!
//! Both give the display's own pixels: nothing is scaled. ScreenCaptureKit
//! answers through completion handlers, on its own queues: each is a block
//! (block.rs) that hands its result over through a [`Pending`], which this
//! thread waits for until the request's deadline.

use super::block::{self, Block, Descriptor};
use super::cf;
use super::cg::{self, Display, DisplayId};
use super::objc::{self, Id, Retained};
use super::png;
use crate::pending::Pending;
use crate::{CapturePath, Failure, Screenshot, ScreenshotOptions, REQUEST_TIMEOUT};
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;
use std::time::Instant;

extern "C" {
  fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
  fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_LAZY: c_int = 0x1;
const RTLD_LOCAL: c_int = 0x4;
/// `RTLD_DEFAULT`: search every loaded image, in load order (dlfcn.h).
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

/// What the user does when screen recording isn't allowed.
const ALLOW_IT: &str = "Allow screen recording for the app that runs this, usually the \
  terminal, in System Settings > Privacy & Security > Screen Recording (Screen & System Audio \
  Recording on macOS 15 and later), then quit and reopen that app.";

/// SCStreamErrorUserDeclined (SCError.h): screen recording isn't allowed.
/// The check before capturing (`screen_capture_allowed`) usually comes first.
const USER_DECLINED: isize = -3801;

/// The paths that this macOS has, the default first.
pub fn paths() -> Vec<CapturePath> {
  let mut paths = Vec::new();
  if has_screenshot_manager() {
    paths.push(CapturePath::ScreenCaptureKit);
  }
  if display_create_image().is_some() {
    paths.push(CapturePath::CoreGraphics);
  }
  paths
}

/// Whether ScreenCaptureKit has `SCScreenshotManager` (macOS 14 and later).
/// The first call loads the framework, where there is one.
fn has_screenshot_manager() -> bool {
  static FOUND: OnceLock<bool> = OnceLock::new();
  *FOUND.get_or_init(|| {
    let path = c"/System/Library/Frameworks/ScreenCaptureKit.framework/ScreenCaptureKit";
    // macOS 11 has no such framework, and dlopen returns null.
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
    !handle.is_null() && objc::class(c"SCScreenshotManager").is_some()
  })
}

/// `CGImageRef CGDisplayCreateImage(CGDirectDisplayID display)`.
type DisplayCreateImage = unsafe extern "C" fn(DisplayId) -> *const c_void;

/// CoreGraphics' `CGDisplayCreateImage`, where this macOS has it.
fn display_create_image() -> Option<DisplayCreateImage> {
  static FOUND: OnceLock<Option<DisplayCreateImage>> = OnceLock::new();
  *FOUND.get_or_init(|| {
    let symbol = unsafe { dlsym(RTLD_DEFAULT, c"CGDisplayCreateImage".as_ptr()) };
    if symbol.is_null() {
      return None;
    }
    Some(unsafe { std::mem::transmute::<*mut c_void, DisplayCreateImage>(symbol) })
  })
}

/// Takes a screenshot of the display that `options` names, by the path that
/// it names, or by the default one.
pub fn screenshot(options: &ScreenshotOptions, deadline: Instant) -> Result<Screenshot, Failure> {
  let displays = cg::displays();
  let display = pick_display(&displays, options.display)?;
  let paths = paths();
  let path = match options.capture {
    None => paths.first().copied().ok_or_else(|| {
      Failure::unsupported(
        "screenshot: this macOS has neither ScreenCaptureKit's SCScreenshotManager nor \
         CoreGraphics' CGDisplayCreateImage",
      )
    })?,
    Some(path) if paths.contains(&path) => path,
    Some(path) => {
      let names: Vec<&str> = paths.iter().map(|path| path.name()).collect();
      let has = if names.is_empty() {
        "none".to_string()
      } else {
        names.join(" and ")
      };
      return Err(Failure::unsupported(format!(
        "screenshot: this macOS has no {} path; it has {has}",
        path.name()
      )));
    }
  };
  if !cg::screen_capture_allowed() {
    cg::request_screen_capture();
    return Err(Failure::permission(format!(
      "screenshot: macOS doesn't allow screen recording here. {ALLOW_IT}"
    )));
  }
  let (image, cursor) = match path {
    CapturePath::ScreenCaptureKit => (
      with_screencapturekit(display, options.cursor, deadline)?,
      options.cursor,
    ),
    CapturePath::CoreGraphics => (with_coregraphics(display.id)?, false),
  };
  let png = png::encode(&image)?;
  let (width, height) = cg::image_size(&image);
  Ok(Screenshot {
    display: display.id,
    capture: path,
    width,
    height,
    bounds: display.bounds,
    cursor,
    png,
  })
}

/// The display of `id`, or the main one.
fn pick_display(displays: &[Display], id: Option<DisplayId>) -> Result<&Display, Failure> {
  let Some(id) = id else {
    let main = displays.iter().find(|display| display.main);
    return main
      .or(displays.first())
      .ok_or_else(|| Failure::failed("screenshot: macOS lists no active display"));
  };
  displays
    .iter()
    .find(|display| display.id == id)
    .ok_or_else(|| {
      let ids: Vec<String> = displays
        .iter()
        .map(|display| display.id.to_string())
        .collect();
      let listed = if ids.is_empty() {
        "none".to_string()
      } else {
        ids.join(", ")
      };
      Failure::request(format!(
        "screenshot: there is no display {id}; the displays are {listed}"
      ))
    })
}

fn with_coregraphics(display: DisplayId) -> Result<cf::Owned, Failure> {
  let create = display_create_image()
    .ok_or_else(|| Failure::unsupported("screenshot: this macOS has no CGDisplayCreateImage"))?;
  let image = unsafe { cf::Owned::from_create(create(display)) };
  if image.is_null() {
    return Err(Failure::failed(format!(
      "screenshot: CGDisplayCreateImage gave no image of display {display}"
    )));
  }
  Ok(image)
}

fn with_screencapturekit(
  display: &Display,
  cursor: bool,
  deadline: Instant,
) -> Result<cf::Owned, Failure> {
  let content = shareable_content(deadline)?;
  unsafe {
    let found = find_display(content.get(), display.id)?;
    let windows: Id = msg_send![class_named(c"NSArray")?, array];
    let filter: Id = msg_send![class_named(c"SCContentFilter")?, alloc];
    let filter = Retained::from_owned(msg_send![
      filter,
      initWithDisplay: found,
      excludingWindows: windows,
    ]);
    let config = Retained::from_owned(msg_send![class_named(c"SCStreamConfiguration")?, new]);
    if filter.is_null() || config.is_null() {
      return Err(Failure::failed(format!(
        "screenshot: ScreenCaptureKit made no filter or configuration for display {}",
        display.id
      )));
    }
    // The display's own pixels, or its size in points if macOS gave no mode.
    let (width, height) = if display.pixel_width > 0 && display.pixel_height > 0 {
      (display.pixel_width, display.pixel_height)
    } else {
      (
        display.bounds.width as usize,
        display.bounds.height as usize,
      )
    };
    let () = msg_send![config.get(), setWidth: width];
    let () = msg_send![config.get(), setHeight: height];
    let () = msg_send![config.get(), setShowsCursor: cursor];
    capture_image(filter.get(), config.get(), deadline)
  }
}

/// The displays, windows and apps that ScreenCaptureKit can capture
/// (`SCShareableContent`).
fn shareable_content(deadline: Instant) -> Result<Retained, Failure> {
  let class = class_named(c"SCShareableContent")?;
  let handed = complete::<ContentHandler>(&CONTENT_BLOCK, deadline, |block| unsafe {
    // Without the desktop's windows, and on-screen ones only: fewer to list.
    let () = msg_send![
      class,
      getShareableContentExcludingDesktopWindows: true,
      onScreenWindowsOnly: true,
      completionHandler: block,
    ];
  });
  let (content, error) = handed.ok_or_else(|| timed_out("listing the displays"))?;
  if content.is_null() {
    return Err(unsafe { error_failure("listing the displays", error.get()) });
  }
  Ok(content)
}

/// One screenshot (`SCScreenshotManager`).
fn capture_image(filter: Id, config: Id, deadline: Instant) -> Result<cf::Owned, Failure> {
  let class = class_named(c"SCScreenshotManager")?;
  let handed = complete::<ImageHandler>(&IMAGE_BLOCK, deadline, |block| unsafe {
    let () = msg_send![
      class,
      captureImageWithFilter: filter,
      configuration: config,
      completionHandler: block,
    ];
  });
  let (image, error) = handed.ok_or_else(|| timed_out("taking the screenshot"))?;
  if image.is_null() {
    return Err(unsafe { error_failure("taking the screenshot", error.get()) });
  }
  Ok(image)
}

/// The SCDisplay of `id` in `content`, which it lives as long as.
///
/// # Safety
/// `content` is a live SCShareableContent.
unsafe fn find_display(content: Id, id: DisplayId) -> Result<Id, Failure> {
  let displays: Id = msg_send![content, displays];
  let count: usize = msg_send![displays, count];
  for index in 0..count {
    let display: Id = msg_send![displays, objectAtIndex: index];
    let display_id: u32 = msg_send![display, displayID];
    if display_id == id {
      return Ok(display);
    }
  }
  Err(Failure::failed(format!(
    "screenshot: ScreenCaptureKit doesn't list display {id}"
  )))
}

fn class_named(name: &CStr) -> Result<Id, Failure> {
  objc::class(name).ok_or_else(|| {
    Failure::unsupported(format!(
      "screenshot: this macOS has no {} class",
      name.to_string_lossy()
    ))
  })
}

fn timed_out(doing: &str) -> Failure {
  Failure::failed(format!(
    "screenshot: ScreenCaptureKit didn't finish {doing} within the request's {} s",
    REQUEST_TIMEOUT.as_secs()
  ))
}

/// A failure from a completion handler's NSError; nil means that the handler
/// gave neither a result nor an error.
///
/// # Safety
/// `error` is nil or a live NSError.
unsafe fn error_failure(doing: &str, error: Id) -> Failure {
  if error.is_null() {
    return Failure::failed(format!(
      "screenshot: ScreenCaptureKit gave no result and no error while {doing}"
    ));
  }
  let domain = objc::string(msg_send![error, domain]).unwrap_or_default();
  let code: isize = msg_send![error, code];
  let text = objc::string(msg_send![error, localizedDescription]).unwrap_or_default();
  let message = format!("screenshot: ScreenCaptureKit failed {doing}: {text} ({domain} {code})");
  if code == USER_DECLINED && domain.contains("ScreenCaptureKit") {
    Failure::permission(format!("{message}. {ALLOW_IT}"))
  } else {
    Failure::failed(message)
  }
}

// Completion handlers.

/// A completion handler of two pointer arguments, `(object, NSError *)`, and
/// what it hands over.
trait Handler {
  type Handed: Send;

  /// Takes the arguments over, retained. It runs on the framework's queue,
  /// so it must not panic.
  ///
  /// # Safety
  /// The arguments are what the framework passed: nil, or live objects.
  unsafe fn hand(object: *mut c_void, error: *mut c_void) -> Self::Handed;
}

/// `void (^)(SCShareableContent *shareableContent, NSError *error)`.
struct ContentHandler;

impl Handler for ContentHandler {
  type Handed = (Retained, Retained);

  unsafe fn hand(content: *mut c_void, error: *mut c_void) -> (Retained, Retained) {
    (
      Retained::retain(content.cast()),
      Retained::retain(error.cast()),
    )
  }
}

/// `void (^)(CGImageRef sampleBuffer, NSError *error)`.
struct ImageHandler;

impl Handler for ImageHandler {
  type Handed = (cf::Owned, Retained);

  unsafe fn hand(image: *mut c_void, error: *mut c_void) -> (cf::Owned, Retained) {
    (cf::Owned::retain(image), Retained::retain(error.cast()))
  }
}

// The blocks' type encodings: return type, the arguments' total size, then
// each argument and its offset, the block itself (`@?`) first.
static CONTENT_BLOCK: Descriptor = block::descriptor(c"v24@?0@8@16");
static IMAGE_BLOCK: Descriptor = block::descriptor(c"v24@?0^{CGImage=}8@16");

/// A block's invoke function for handler `H`. It runs on the framework's
/// queue, and must not panic; nothing here can.
unsafe extern "C" fn invoke<H: Handler>(
  block: *mut Block,
  object: *mut c_void,
  error: *mut c_void,
) {
  if let Some(pending) = Block::take::<H::Handed>(block) {
    pending.finish(H::hand(object, error));
  }
}

/// Calls `start` with a completion block for handler `H` (`start` sends the
/// message that takes it), and waits until `deadline` for what the handler
/// hands over; None if the time runs out first.
fn complete<H: Handler>(
  descriptor: &'static Descriptor,
  deadline: Instant,
  start: impl FnOnce(Id),
) -> Option<H::Handed> {
  let pending = Pending::<H::Handed>::new();
  let invoke: unsafe extern "C" fn(*mut Block, *mut c_void, *mut c_void) = invoke::<H>;
  let mut block = Block::new(invoke as *const c_void, descriptor, &pending);
  start(block.as_id());
  pending.wait(deadline.saturating_duration_since(Instant::now()))
}
