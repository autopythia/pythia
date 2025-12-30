extern crate pythia;
//extern crate rayon;
//extern crate term_colors;

use pythia::clock::{Timedelta, Timestamp};
use pythia::interp::*;
use pythia::interp_test::*;
//use pythia::smp::{init_smp};
use pythia::tap::*;
use pythia::test_data::*;
//use rayon::prelude::*;
//use term_colors::{Colorize};

fn main() {
  //init_smp();
  let test_data_cfg = TestDataConfig::default();
  println!("DEBUG: boot: test data config = {:?}", test_data_cfg);
  let prover = InterpTestsProver::from(test_data_cfg);
  EchoTAPParser::parse(prover);
}
