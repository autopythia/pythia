use crate::algo::{Arc};

#[cfg(unix)]
use libc::{
  PROT_NONE, PROT_READ, PROT_WRITE, PROT_EXEC,
  MAP_FAILED, MAP_ANON, MAP_PRIVATE,
  _SC_PAGESIZE, _SC_NPROCESSORS_ONLN,
  mmap, munmap, mprotect, sysconf,
};

use std::ffi::{c_void};
use std::fs::{File};
#[cfg(unix)] use std::os::unix::fs::{MetadataExt};
#[cfg(unix)] use std::os::unix::io::{AsRawFd};
use std::path::{Path};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[cfg(target_os = "macos")]
extern "C" { pub fn sys_icache_invalidate(start: *mut c_void, len: usize); }

pub fn cast_u32s_as_bytes(src: &[u32]) -> &[u8] {
  unsafe { from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) }
}

#[derive(Clone)]
pub struct MmapFile {
  pub file: Arc<File>,
  pub ptr:  *mut c_void,
  pub size: usize,
}

impl Drop for MmapFile {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    // FIXME: weak count?
    if Arc::strong_count(&self.file) == 1 {
      assert!(!self.ptr.is_null());
      let ret = unsafe { munmap(self.ptr, self.size) };
      assert_eq!(ret, 0);
    }
  }
}

impl MmapFile {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<MmapFile, ()> {
    let file = Arc::new(File::open(path).map_err(|_| ())?);
    MmapFile::from_file(&file)
  }

  #[cfg(not(unix))]
  pub fn from_file(_f: &Arc<File>) -> Result<MmapFile, ()> {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn from_file(f: &Arc<File>) -> Result<MmapFile, ()> {
    let size = f.metadata().unwrap().size();
    if size > usize::max_value() as u64 {
      return Err(());
    }
    let size = size as usize;
    let fd = f.as_raw_fd();
    let ptr = unsafe { mmap(null_mut(), size, PROT_READ, MAP_PRIVATE, fd, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapFile{ptr, size, file: f.clone()})
  }

  pub fn as_ptr(&self) -> *mut c_void {
    self.ptr
  }

  pub fn len(&self) -> usize {
    self.size
  }

  #[cfg(unix)]
  pub unsafe fn as_bytes_unsafe(&self) -> &[u8] {
    from_raw_parts(self.ptr as *mut u8 as *const u8, self.size)
  }

  #[cfg(all(unix, target_os = "macos"))]
  pub fn as_bytes(&self) -> &[u8] {
    // NB: some ~decade old anecdotal evidence suggests that writes
    // made by another proc are not visible to this current proc's
    // MAP_PRIVATE mapping;
    // see: <https://stackoverflow.com/questions/14670869/file-changes-after-a-mmap-in-os-x-ios>
    unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8, self.size) }
  }
}

#[cfg(all(unix, target_os = "macos"))]
impl AsRef<[u8]> for MmapFile {
  fn as_ref(&self) -> &[u8] {
    self.as_bytes()
  }
}

pub struct MmapJitBuffer {
  pub ptr:  *mut c_void,
  pub size: usize,
  pub exec: bool,
}

impl Drop for MmapJitBuffer {
  #[cfg(not(unix))]
  fn drop(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  fn drop(&mut self) {
    assert!(!self.ptr.is_null());
    let ret = unsafe { munmap(self.ptr, self.size) };
    assert_eq!(ret, 0);
  }
}

impl MmapJitBuffer {
  #[cfg(not(unix))]
  pub fn new(size: usize) -> Result<MmapJitBuffer, ()> {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn new(size: usize) -> Result<MmapJitBuffer, ()> {
    let ptr = unsafe { mmap(null_mut(), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0) };
    if ptr == MAP_FAILED {
      return Err(());
    }
    assert!(!ptr.is_null());
    Ok(MmapJitBuffer{ptr, size, exec: false})
  }

  #[cfg(not(unix))]
  pub fn set_write(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn set_write(&mut self) {
    if !self.exec {
      return;
    }
    let ret = unsafe { mprotect(self.ptr, self.size, PROT_READ | PROT_WRITE) };
    assert_eq!(ret, 0);
    self.exec = false;
  }

  #[cfg(not(target_os = "macos"))]
  pub fn flush_write(&mut self) {
    unimplemented!();
  }

  #[cfg(target_os = "macos")]
  pub fn flush_write(&mut self) {
    assert!(!self.exec);
    unsafe { sys_icache_invalidate(self.ptr, self.size) };
  }

  #[cfg(not(unix))]
  pub fn set_exec(&mut self) {
    unimplemented!();
  }

  #[cfg(unix)]
  pub fn set_exec(&mut self) {
    if self.exec {
      return;
    }
    let ret = unsafe { mprotect(self.ptr, self.size, PROT_READ | PROT_EXEC) };
    assert_eq!(ret, 0);
    self.exec = true;
  }

  pub fn as_bytes(&self) -> &[u8] {
    unsafe { from_raw_parts(self.ptr as *mut u8 as *const u8, self.size) }
  }

  pub fn as_bytes_mut(&mut self) -> &mut [u8] {
    assert!(!self.exec);
    unsafe { from_raw_parts_mut(self.ptr as *mut u8, self.size) }
  }

  pub fn exec_ptr(&self) -> *mut c_void {
    assert!(self.exec);
    self.ptr
  }
}
