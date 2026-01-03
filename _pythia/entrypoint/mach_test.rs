extern crate _pythia;

use _pythia::tap::*;
use _pythia::test_data::*;
use _pythia::zinterp::mach_test::*;

fn main() {
  let argv: Vec<_> = std::env::args().collect();
  let n: i32 = if argv.len() > 1 {
    argv[1].parse().unwrap()
  } else {
    -1
  };
  let test_data_cfg = if n == 1 {
    TestDataConfig::zinterp_mach_test_1()
  } else if n < 0 {
    TestDataConfig::zinterp_mach_tests()
  } else {
    unimplemented!();
  };
  println!("DEBUG: test data config = {:?}", test_data_cfg);
  let prover = MachTestsProver::from(test_data_cfg);
  DefaultTAPParser::parse(prover);
}
