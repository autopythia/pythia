#![allow(unused_variables)]

use crate::algo::{BTreeMap, Rc};

use crate::zinterp::mach::{
  LReg,
  ZKntMachLog, ZKntMachState, ZKMachCheck, ZKMachYield_,
};
use crate::zinterp::tree::*;

#[derive(Clone, Debug)]
pub struct ZChoicePtr {
  idx:  u16,
}

pub enum ZInterpCheck {
  K(ZKMachCheck),
}

impl From<ZKMachCheck> for ZInterpCheck {
  fn from(kchk: ZKMachCheck) -> ZInterpCheck {
    ZInterpCheck::K(kchk)
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ZInterpControlReg {
  pub ictr: u32,
  pub depth: u16,
}

impl ZInterpControlReg {
  pub fn init() -> ZInterpControlReg {
    ZInterpControlReg{
      ictr: 0,
      depth: 0,
    }
  }
}

pub type ZInterpCursorRef = Rc<ZInterpCursor>;

pub struct ZInterpCursor {
  pub klog: ZKntMachLog,
  // TODO: snapshot metadata.
  //KLog{klog: ZKntMachLog},
  //KLogSnapshot{...},
}

pub struct ZInterpCursorSnapshot {
  // TODO: snapshot metadata.
}

pub struct ZInterpState {
  pub kstate: ZKntMachState,
}

#[derive(Clone, Debug)]
pub enum ZInterpFailure {
  // TODO: failure metadata.
  _Bot,
}

#[derive(Clone, Debug)]
pub struct ZInterpHistoryChoice {
  pub ptr:  ZChoicePtr,
  pub fail: Option<ZInterpFailure>,
}

pub type ZInterpHistoryRef = Rc<ZInterpHistory>;

pub struct ZInterpHistory {
  // TODO: snapshot metadata.
  pub choices: Vec<ZInterpHistoryChoice>,
}

impl ZInterpHistory {
  pub fn arity(&self) -> usize {
    unimplemented!();
  }

  pub fn append(&self, interp: &ZInterpCursor, state: &ZInterpState) {
  }
}

pub type ZInterpLogSnapshotRef = Option<Rc<ZInterpLogSnapshot>>;

pub struct ZInterpLogSnapshot {
  pub ctl:  ZInterpControlReg,
  pub back: ZInterpLogSnapshotRef,
  pub cur:  ZInterpCursorSnapshot,
  pub prev: ZInterpHistory,
}

pub type ZInterpLogRef = Option<Rc<ZInterpLog>>;

pub struct ZInterpLog {
  pub ctl:  ZInterpControlReg,
  //pub back: ZInterpLogRef,
  pub back: ZInterpLogSnapshotRef,
  pub cur:  ZInterpCursor,
  pub prev: ZInterpHistory,
  /*pub next: _,*/
}

impl ZInterpLog {
  pub fn pop_next_inplace(&mut self, state: &ZInterpState) -> Result<ZChoicePtr, ()> {
    unimplemented!();
  }

  pub fn apply_next_inplace(&mut self, state: &mut ZInterpState, next: ZChoicePtr) {
    unimplemented!();
  }

  pub fn _step_inplace(&mut self, state: &mut ZInterpState) -> Result<(), Result<(), ZInterpCheck>> {
    unimplemented!();
  }

  pub fn _eval_inplace(&mut self, state: &mut ZInterpState) -> Result<(), Result<(), ZInterpCheck>> {
    unimplemented!();
  }

  pub fn _backup_inplace(&mut self, state: &mut ZInterpState) -> Result<(), Result<(), ZInterpCheck>> {
    unimplemented!();
  }

  pub fn resume_inplace(&mut self, state: &mut ZInterpState, /*resume_arg: &mut LReg<_>*/) -> Result<ZKMachYield_, ZKMachCheck> {
    //loop {
      match self.cur.klog.resume_inplace(&mut state.kstate) {
        Ok(ZKMachYield_::Yield) => {
          // TODO
          /*
          // TODO: step function.
          match self.cur.klog.cur.yield_._remove()? {
            //ZKMachYield::Start => {}
            ZKMachYield::Choice(_) => {
              return self._step_choice_inplace();
            }
            ZKMachYield::Fail(_) => {
            }
            ZKMachYield::Except(_) => {
            }
          }
          for _ in 0 .. 2 {
            let prev_arity = self.prev.arity();
            match arity {
              None | Some(a) if prev_arity < a => {
                if fail_._is_fill() {
                  let _ = fail_._remove()?;
                  self.prev._fill_fail();
                  continue;
                }
                return self._step_choice_inplace();
              }
              Some(a) if prev_arity == a => {
                return Err(Ok());
              }
              _ => {
                return bot();
              }
            }
          }
          // TODO: eval function (?).
          // TODO: eval should be where klog resume runs.
          // TODO: backup function.
          let back = replace(&mut self.back, None);
          if back.is_none() {
            // NB: safely ("linearly") set cur halt.
            self.cur.halt._fill(_).or_else(|_| bot())?;
            return Err(Ok());
          }
          let mut back = back.unwrap();
          self.ctl.depth = back.ctl.depth;
          swap(&mut self.back, &mut back.back);
          self.cur._restore(&mut back.cur);
          self.prev._restore(&mut back.prev);
          */
        }
        Ok(_) => {
        }
        Err(kchk) => {
          return Err(kchk.into());
        }
      }
    //}
    unimplemented!();
  }
}
