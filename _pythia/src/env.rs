use once_cell::sync::{Lazy};

use std::env::{var};
//use std::path::{PathBuf};

pub static RUNTIME_ENV: Lazy<RuntimeEnv> = Lazy::new(|| RuntimeEnv::get_once());
thread_local! {
  pub static TL_RUNTIME_ENV: RuntimeEnv = RUNTIME_ENV.clone();
}

#[derive(Clone)]
pub struct RuntimeEnv {
  pub silent: bool,
  pub debug:  i8,
}

impl RuntimeEnv {
  pub fn get_once() -> RuntimeEnv {
    let silent = var("PYTHIA_SILENT")
      .map(|_| true)
      .unwrap_or_else(|_| false);
    let debug = var("PYTHIA_DEBUG")
      .map(|s| match s.parse() {
        Ok(d) => d,
        Err(_) => 1
      })
      .unwrap_or_else(|_| 0);
    RuntimeEnv{
      silent,
      debug,
    }
  }
}

pub fn rte_debug() -> bool {
  TL_RUNTIME_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= 1
  })
}

pub fn rte_trace() -> bool {
  TL_RUNTIME_ENV.with(|cfg| {
    !cfg.silent && cfg.debug >= 3
  })
}
