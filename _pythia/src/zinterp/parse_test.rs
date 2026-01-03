// TODO: temporarily disabled lint for debugging.
#![allow(unused_variables)]

use crate::clock::{Timedelta, Timestamp};
use crate::tap::*;
use crate::test_data::*;
use crate::zinterp::parse::*;

use term_colors::{Colorize};

use std::io::{Read, Write, Error as IoError};

pub struct ParseTestItem {
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
pub struct ParseTestsProver {
  conf: TestDataConfig,
}

impl From<TestDataConfig> for ParseTestsProver {
  fn from(conf: TestDataConfig) -> ParseTestsProver {
    ParseTestsProver{conf}
  }
}

impl ParseTestsProver {
  pub fn _prove_item<W: Write + ?Sized>(&self, rank: usize, item: &ParseTestItem, writer: &mut W) -> Result<(), IoError> {
    //let snapshot = self.conf.init_snapshot_file(&item.key);
    let mut parser = FastParser::new(&item.src);
    parser.set_debug();
    match parser.mod_() {
      Err(e) => {
        writeln!(writer, "{} {} - {:?}", "not ok".red().bold(), rank, &item.key)?;
        writeln!(writer, "# parse error = {:?}", e)?;
        return Ok(());
      }
      Ok(mod_) => {}
    }
    writeln!(writer, "{} {} - {:?}", "ok".green(), rank, &item.key)?;
    Ok(())
  }
}

impl TAPProver for ParseTestsProver {
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
      let item = ParseTestItem{
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
