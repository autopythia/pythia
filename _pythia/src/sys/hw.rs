#[cfg(unix)]
use libc::{
  _SC_PAGESIZE, _SC_NPROCESSORS_ONLN, sysconf,
};

#[cfg(target_os = "macos")]
use std::io::{BufRead, Cursor};
#[cfg(target_os = "macos")]
use std::process::{Command, Stdio};

#[derive(Clone, Copy, Debug)]
pub struct HardwareConfig {
  pub page_size: usize,
  pub mem_size: usize,
  pub pcore_count: usize,
}

impl HardwareConfig {
  pub fn new() -> HardwareConfig {
    let page_size = (unsafe { sysconf(_SC_PAGESIZE) }).try_into().unwrap();
    #[cfg(target_os = "macos")]
    let (pcore_count, mem_size) = {
      let sysctl = SysctlParse::open().unwrap();
      (sysctl.pcore_ct, sysctl.mem_sz)
    };
    HardwareConfig{
      page_size,
      mem_size,
      pcore_count,
    }
  }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct SysctlParse {
  pub pcore_ct: usize,
  pub mem_sz: usize,
}

#[cfg(target_os = "macos")]
impl SysctlParse {
  pub fn open() -> Result<SysctlParse, ()> {
    let out = Command::new("sysctl")
        .arg("-a")
        .stdout(Stdio::piped())
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
      return Err(());
    }
    SysctlParse::parse(out.stdout)
  }

  pub fn parse<O: AsRef<[u8]>>(out: O) -> Result<SysctlParse, ()> {
    let mut info = SysctlParse{
      pcore_ct: 0,
      mem_sz: 0,
    };
    let out = out.as_ref();
    for line in Cursor::new(out).lines() {
      let line = line.unwrap();
      if line.is_empty() {
        break;
      }
      let mut line_parts = line.split_ascii_whitespace();
      match (line_parts.next(), line_parts.next()) {
        (Some(key), Some(val)) => {
          match key {
            "hw.perflevel0.physicalcpu:" => {
              info.pcore_ct = val.parse().unwrap();
            }
            "hw.memsize_usable:" => {
              info.mem_sz = val.parse().unwrap();
            }
            _ => {}
          }
        }
        _ => {}
      }
    }
    Ok(info)
  }
}
