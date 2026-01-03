// TODO: temporarily disabled lint for debugging.
#![allow(unused_variables)]

use crate::clock::{Timedelta, Timestamp};
use crate::tap::*;
use crate::test_data::*;
use crate::zinterp::mach::*;
use crate::zinterp::parse::*;

use term_colors::{Colorize};

use std::io::{Read, Write, Error as IoError};

pub struct MachTestItem {
  pub key:  String,
  pub src:  String,
  //pub vdst: Option<String>,
  //pub snapshot_dst: Option<String>,
}

pub enum TestResult {
  //OK(Timedelta, Timedelta, Yield_),
  //Check(Timedelta, Timedelta, InterpCheck),
  OK(Timedelta, Timedelta, ),
  Check(Timedelta, Timedelta, ),
}

#[derive(Default)]
pub struct MachTestsProver {
  conf: TestDataConfig,
}

impl From<TestDataConfig> for MachTestsProver {
  fn from(conf: TestDataConfig) -> MachTestsProver {
    MachTestsProver{conf}
  }
}

impl MachTestsProver {
  pub fn _prove_item<W: Write + ?Sized>(&self, rank: usize, item: &MachTestItem, writer: &mut W) -> Result<(), IoError> {
    //let snapshot = self.conf.init_snapshot_file(&item.key);
    let mut parser = FastParser::new(&item.src);
    let mod_ = match parser.mod_() {
      Err(e) => {
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# parse error = {:?}", e)?;
        println!("{} {} - {:?}", "not ok".red().bold(), rank, &item.key);
        println!("# parse error = {:?}", e);
        let mut parser = FastParser::new(&item.src);
        parser.set_debug();
        let _ = parser.mod_();
        return Ok(());
      }
      Ok(mod_) => mod_
    };
    let tap_buf = SharedBuffer::new();
    let mut mach_log = ZKntMachLog::init();
    let mut mach_state = ZKntMachState::init(mod_.into());
    mach_state.set_tap_buffer(tap_buf.clone().wrap_writer());
    let _ = match mach_log.resume_inplace(&mut mach_state) {
      Err(e) => {
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# resume error = {:?}", e)?;
        println!("{} {} - {:?}", "not ok".red().bold(), rank, &item.key);
        println!("# resume error = {:?}", e);
        // TODO
        let _ = mach_state.unset_tap_buffer().unwrap();
        let tap_out = tap_buf.to_string();
        println!("{}", tap_out);
        return Ok(());
      }
      Ok(res) => res
    };
    writeln!(writer, "{} {} - {:?}", "ok".green(), rank, &item.key)?;
    println!("{} {} - {:?}", "ok".green(), rank, &item.key);
    // TODO
    let _ = mach_state.unset_tap_buffer().unwrap();
    let tap_out = tap_buf.to_string();
    println!("{}", tap_out);
    Ok(())
  }
}

impl TAPProver for MachTestsProver {
  fn prove<W: Write + ?Sized>(&self, writer: &mut W) -> Result<(), IoError> {
    let mut ctr = 0;
    for (idx, key) in self.conf.keys().iter().enumerate() {
      let mut f = self.conf.get_source_file(key);
      let mut src = String::new();
      f.read_to_string(&mut src).unwrap();
      /*let vdst = if let Some(mut f) = self.conf.maybe_get_vector_file(key) {
        let mut vdst = String::new();
        f.read_to_string(&mut vdst).unwrap();
        Some(vdst)
      } else {
        None
      };*/
      let item = MachTestItem{
        key: key.to_string(),
        src,
        //vdst,
        // FIXME
        //snapshot_dst: None,
      };
      self._prove_item(idx + 1, &item, writer)?;
      ctr += 1;
    }
    writeln!(writer, "1..{}", ctr)?;
    Ok(())
  }
}
