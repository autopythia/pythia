use crate::algo::{Rc};
use crate::algo::cell::{RefCell, Ref};
use crate::algo::str::{SafeStr};

use term_colors::{Colorize};

use std::io::{
  BufRead, Read, Write, Error as IoError,
  BufReader, BufWriter, Cursor, stdout
};
use std::str::{from_utf8};

pub trait TAPProver {
  fn prove<W: Write + ?Sized>(&self, writer: &mut W) -> Result<(), IoError>;
}

pub type TAPParser = DefaultTAPParser;

#[derive(Default)]
pub struct DefaultTAPParser {
}

impl DefaultTAPParser {
  pub fn parse<P: TAPProver>(prover: P) /*-> DefaultTAPParser */{
    //let mut buf = BufWriter::new(stdout().lock());
    let buf = Vec::<u8>::new();
    let mut buf = BufWriter::new(Cursor::new(buf));
    prover.prove(&mut buf).unwrap();
    let buf = buf.into_inner().unwrap();
    let buf = buf.into_inner();
    //println!("DEBUG: DefaultTAPParser: {:?}", buf.as_bytes());
    let buf = BufReader::new(Cursor::new(buf));
    const OK_PREFIX_1: &'static str = "\u{1b}[32mok\u{1b}[0m ";
    const OK_PREFIX_2: &'static str = "\u{1b}[1;32mok\u{1b}[0m ";
    const NOT_OK_PREFIX_1: &'static str = "\u{1b}[31mnot ok\u{1b}[0m ";
    const NOT_OK_PREFIX_2: &'static str = "\u{1b}[1;31mnot ok\u{1b}[0m ";
    let mut ok_ct: usize = 0;
    let mut not_ok_ct: usize = 0;
    for line in buf.lines() {
      let line = line.unwrap();
      let line_buf = line.as_bytes();
      if line.starts_with("ok ")
      || line.starts_with(OK_PREFIX_1)
      || line.starts_with(OK_PREFIX_2)
      {
        ok_ct += 1;
      } else if line.starts_with("not ok ")
             || line.starts_with(NOT_OK_PREFIX_1)
             || line.starts_with(NOT_OK_PREFIX_2)
      {
        not_ok_ct += 1;
      } else if line_buf.starts_with(b"#") {
      } else if line_buf.starts_with(b"1..") {
      } else {
        println!("DEBUG: DefaultTAPParser: {:?}", line.as_bytes());
        println!("DEBUG: DefaultTAPParser: {:?}", line);
        println!("DEBUG: DefaultTAPParser: {}", line);
        panic!("bug");
        //break;
      }
      println!("{}", line);
    }
    if not_ok_ct > 0 {
      println!("Result: {}", "FAIL".red().bold());
      println!("Failed {} / {} test programs.",
          not_ok_ct,
          not_ok_ct + ok_ct
      );
    } else {
      println!("All tests successful.");
      println!("Result: {}", "PASS".green().bold());
      println!("Passed {} test programs.", ok_ct);
    }
  }
}

#[derive(Clone)]
pub struct SharedBuffer {
  buf:  Rc<RefCell<Vec<u8>>>,
}

impl SharedBuffer {
  pub fn new() -> SharedBuffer {
    SharedBuffer{
      buf:  Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub fn wrap_writer(self) -> Box<dyn Write> {
    wrap_writer(self)
  }

  pub fn borrow_bytes(&self) -> Ref<'_, Vec<u8>> {
    self.buf.borrow()
  }

  pub fn to_string(&self) -> String {
    from_utf8(&*self.borrow_bytes()).unwrap().to_owned()
  }
}

impl Write for SharedBuffer {
  fn write(&mut self, src: &[u8]) -> Result<usize, IoError> {
    self.buf.borrow_mut().write(src)
  }

  fn flush(&mut self) -> Result<(), IoError> {
    self.buf.borrow_mut().flush()
  }
}

// FIXME: Box<dyn Write> not flexible enough to unwrap...
pub fn wrap_writer<W: 'static + Write>(writer: W) -> Box<dyn Write> {
  Box::new(writer)
}
