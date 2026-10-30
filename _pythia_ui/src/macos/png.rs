//! PNG bytes from a CGImage, through ImageIO. The functions are listed in
//! stub-sdk/System/Library/Frameworks/ImageIO.framework/ImageIO.tbd.

use super::cf::{self, Owned};
use crate::Failure;
use std::ffi::c_void;

#[link(name = "ImageIO", kind = "framework")]
extern "C" {
  fn CGImageDestinationCreateWithData(
    data: *mut c_void,
    kind: *const c_void,
    count: usize,
    options: *const c_void,
  ) -> *mut c_void;
  fn CGImageDestinationAddImage(
    destination: *mut c_void,
    image: *const c_void,
    properties: *const c_void,
  );
  fn CGImageDestinationFinalize(destination: *mut c_void) -> bool;
}

/// `image`, a CGImage, as PNG.
pub fn encode(image: &Owned) -> Result<Vec<u8>, Failure> {
  unsafe {
    let data = Owned::from_create(cf::CFDataCreateMutable(std::ptr::null(), 0));
    let kind = Owned::from_create(cf::CFStringCreateWithCString(
      std::ptr::null(),
      c"public.png".as_ptr(),
      cf::UTF8,
    ));
    if data.is_null() || kind.is_null() {
      return Err(Failure::failed(
        "screenshot: CoreFoundation couldn't make the PNG's buffer",
      ));
    }
    let destination = Owned::from_create(CGImageDestinationCreateWithData(
      data.get().cast_mut(),
      kind.get(),
      1,
      std::ptr::null(),
    ));
    if destination.is_null() {
      return Err(Failure::failed("screenshot: ImageIO has no PNG encoder"));
    }
    CGImageDestinationAddImage(destination.get().cast_mut(), image.get(), std::ptr::null());
    if !CGImageDestinationFinalize(destination.get().cast_mut()) {
      return Err(Failure::failed(
        "screenshot: ImageIO couldn't encode the screenshot as PNG",
      ));
    }
    let length = usize::try_from(cf::CFDataGetLength(data.get())).unwrap_or(0);
    let bytes = cf::CFDataGetBytePtr(data.get());
    if bytes.is_null() || length == 0 {
      return Err(Failure::failed("screenshot: ImageIO wrote an empty PNG"));
    }
    Ok(std::slice::from_raw_parts(bytes, length).to_vec())
  }
}
