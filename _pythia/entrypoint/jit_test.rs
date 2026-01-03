extern crate _pythia;

use _pythia::sys::hw::{HardwareConfig};
use _pythia::sys::mmap::{MmapJitBuffer, cast_u32s_as_bytes};

use std::mem::{transmute};

type TestFun = unsafe extern "C" fn (u64) -> u64;

fn main() {
  let hw = HardwareConfig::new();
  println!("DEBUG: pcore ct = {}", hw.pcore_count);
  println!("DEBUG: mem sz   = {}", hw.mem_size);
  println!("DEBUG: page sz  = {}", hw.page_size);
  let words = &[
      0x91000400,
      0xd65f03c0,
  ];
  let bytes = cast_u32s_as_bytes(words);
  let mut jit = MmapJitBuffer::new(hw.page_size).unwrap();
  (&mut jit.as_bytes_mut()[ .. 8]).copy_from_slice(&bytes);
  jit.flush_write();
  jit.set_exec();
  let ret = unsafe {
    let fun: TestFun = transmute(jit.exec_ptr());
    fun(42)
  };
  println!("DEBUG: ret = {}", ret);
}
