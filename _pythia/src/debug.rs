#![allow(unused_imports)]
#![allow(unused_macros)]

use crate::algo::cell::{RefCell};

use std::io::{
  BufRead, Read, Write, Error as IoError,
  BufReader, BufWriter, Cursor, stdout
};

pub struct DebugOutput {
  pub writer:   RefCell<Box<dyn Write>>,
  pub verbose:  i8,
}

impl Default for DebugOutput {
  fn default() -> DebugOutput {
    DebugOutput::stdout()
  }
}

impl DebugOutput {
  pub fn from_writer(writer: Box<dyn Write>) -> DebugOutput {
    DebugOutput{
      writer:   RefCell::new(writer),
      verbose:  0,
    }
  }

  pub fn stdout_writer() -> Box<dyn Write> {
    Box::new(std::io::stdout())
  }

  pub fn stdout() -> DebugOutput {
    DebugOutput::from_writer(DebugOutput::stdout_writer())
  }
}

macro_rules! _errorln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 0;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _errorln;

macro_rules! _warningln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 1;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _warningln;

macro_rules! _infoln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 2;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _infoln;

macro_rules! _debugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 3;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _debugln;

macro_rules! _vdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 4;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vdebugln;

macro_rules! _vvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 5;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vvdebugln;

macro_rules! _vvvdebugln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 6;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _vvvdebugln;

macro_rules! _traceln {
  ($self:expr, $($arg:tt)*) => {{
    let print = $self.tap.verbose >= 7;
    if print {
      writeln!($self.tap.writer.borrow_mut(), $($arg)*).unwrap();
    }
    print
  }};
}
pub(crate) use _traceln;
