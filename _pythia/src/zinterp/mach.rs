#![allow(unused_imports)]
#![allow(unused_variables)]

use crate::algo::{BTreeMap, BTreeSet, Rc};
//use crate::algo::rc::{Rc};
use crate::algo::str::{SafeStr, into_safe_str};
use crate::panic_::{Loc, loc};
use crate::zinterp::tree::*;

use std::fmt::{Debug};
use std::io::{Write, Error as IoError};
use std::mem::{replace, swap};

pub type RawSNum = i32;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SNum(RawSNum);

/*impl SNum {
  #[inline]
  pub fn _nil() -> SNum {
    SNum(0)
  }

  #[inline]
  pub fn _is_nil(&self) -> bool {
    self.0 == 0
  }
}*/

pub trait Nil {
  fn nil() -> Self where Self: Sized;
  fn is_nil(&self) -> bool;
}

#[inline]
pub fn nil<T: Nil>() -> T {
  T::nil()
}

impl Nil for SNum {
  #[inline]
  fn nil() -> SNum {
    //SNum::_nil()
    SNum(0)
  }

  #[inline]
  fn is_nil(&self) -> bool {
    //SNum::_is_nil(self)
    self.0 == 0
  }
}

pub trait Lb {
  fn lb() -> Self where Self: Sized;
  fn is_lb(&self) -> bool;
}

#[inline]
pub fn lb<T: Lb>() -> T {
  T::lb()
}

impl Lb for SNum {
  #[inline]
  fn lb() -> SNum {
    SNum(RawSNum::min_value())
  }

  #[inline]
  fn is_lb(&self) -> bool {
    self.0 == RawSNum::min_value()
  }
}

pub trait Ub {
  fn ub() -> Self where Self: Sized;
  fn is_ub(&self) -> bool;
}

#[inline]
pub fn ub<T: Ub>() -> T {
  T::ub()
}

impl Ub for SNum {
  #[inline]
  fn ub() -> SNum {
    SNum(RawSNum::max_value())
  }

  #[inline]
  fn is_ub(&self) -> bool {
    self.0 == RawSNum::max_value()
  }
}

/*pub struct ZKVal {
}

pub struct ZKTerm {
}*/

// NB: a linear register.
#[derive(Clone, Debug)]
pub enum LReg<Item> {
  //Empty,
  Remove,
  Fill(Item),
}

impl<Item> LReg<Item> {
  #[inline]
  pub fn init() -> LReg<Item> {
    LReg::Remove
  }

  #[inline]
  pub fn init_(item: Item) -> LReg<Item> {
    LReg::Fill(item)
  }
}

impl<Item> LReg<Item> {
  pub fn _is_fill(&self) -> bool {
    match self {
      &LReg::Fill(_) => true,
      _ => false
    }
  }

  pub fn _fill(&mut self, item: Item) -> Result<(), Item> {
    let prev_item = replace(self, LReg::Fill(item));
    match prev_item {
      //LReg::Empty |
      LReg::Remove => {
        Ok(())
      }
      LReg::Fill(prev_item) => {
        Err(prev_item)
      }
    }
  }

  pub fn _remove(&mut self) -> Result<Item, ()> {
    let prev_item = replace(self, LReg::Remove);
    match prev_item {
      LReg::Fill(prev_item) => {
        Ok(prev_item)
      }
      //LReg::Empty |
      LReg::Remove => {
        Err(())
      }
    }
  }

  /*pub fn take(&mut self) -> Result<Item, _> {
  }*/
}

#[derive(Clone, Copy, Debug)]
pub enum ZKRetItem {
  // TODO
  //_Empty,
  _Term,
  Term(SNum),
  _Sent,
  _Mod,
}

#[derive(Clone, Copy, Debug)]
pub enum ZKCondReg {
  // TODO
  False,
  True,
}

#[derive(Clone, Copy, Debug)]
pub enum ZKHaltReg {
  // TODO
  Stop,
  Fail,
  Except,
  //Yield,
}

// FIXME: deprecated.
/*pub enum ZKnt {
  Mod(ZKModCodeRef_),
  Sent(ZKSentCodeRef_),
  Term(ZKTermCodeRef_),
}*/

#[derive(Debug)]
pub enum ZKMachCheck {
  // TODO
  Bot{loc: Loc},
  Bot2{loc: Loc, desc: String},
  Unimpl{loc: Loc},
  Io{loc: Loc, err: IoError},
}

pub trait ZKMachCheckExt<Ret> {
  fn chk(self) -> Result<(), Result<Ret, ZKMachCheck>> where Self: Sized;
}

//impl From<IoError> for Result<ZKMachSignal, ZKMachCheck> {}
impl<Ret> ZKMachCheckExt<Ret> for IoError {
  #[track_caller]
  fn chk(self) -> Result<(), Result<Ret, ZKMachCheck>> {
    let loc = loc();
    Err(Err(ZKMachCheck::Io{loc, err: self}))
  }
}

/*#[inline]
pub fn wrap_chk(check: Result<ZKMachSignal, ZKMachCheck>) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
  match check {
    Ok(_) => {
      Ok(())
    }
    Err(check_) => {
      Err(Err(check_))
    }
  }
}*/

#[inline]
pub fn ok_<Item, E>(item: Item) -> Result<Item, E> {
  Ok(item)
}

#[inline]
pub fn ok<Item: From<()>, E>() -> Result<Item, E> {
  Ok(().into())
}

/*#[inline]
pub fn err<E>(e: E) -> Result<(), E> {
  err(e)
}*/

/*pub fn bot<Item>() -> Result<Item, ZKMachCheck> {
  Err(ZKMachCheck::Bot)
}*/

#[allow(non_snake_case)]
#[track_caller]
pub fn Bot<Item>() -> Result<Item, ZKMachCheck> {
  let loc = loc();
  Err(ZKMachCheck::Bot{loc})
}

#[allow(non_snake_case)]
#[track_caller]
pub fn Bot2<Item, S: Into<String>>(desc: S) -> Result<Item, ZKMachCheck> {
  let loc = loc();
  let desc = desc.into();
  Err(ZKMachCheck::Bot2{loc, desc})
}

//#[inline]
#[track_caller]
pub fn bot<Item, Item2>() -> Result<Item, Result<Item2, ZKMachCheck>> {
  let loc = loc();
  Err(Err(ZKMachCheck::Bot{loc}))
}

#[track_caller]
pub fn unimpl<Item, Item2>() -> Result<Item, Result<Item2, ZKMachCheck>> {
  let loc = loc();
  Err(Err(ZKMachCheck::Unimpl{loc}))
}

#[derive(Clone, Copy, Debug)]
pub struct ZKntMachControlReg {
  // NB: the "control register" fragment of the "abstract machine tuple".
  pub ictr: u32,
  pub depth: u16,
  pub phase: ZKMachPhase,
  pub bits: u8,
}

impl ZKntMachControlReg {
  pub fn init() -> ZKntMachControlReg {
    ZKntMachControlReg{
      ictr: 0,
      bits: 0,
      phase: ZKMachPhase::Init,
      depth: 0,
    }
  }

  pub fn _get_eval_bit(&self) -> bool {
    (self.bits & 2) != 0
  }

  pub fn _set_match_bit(&mut self) {
    self.bits |= 0x10;
  }

  pub fn _set_fail_bit(&mut self) {
    self.bits |= 0x80;
  }
}

pub enum ZKCursorReg {
  // TODO
  Tree(Tree),
}

#[derive(Clone, Debug)]
pub struct ZKntMachCursor {
  // NB: the "data register" fragment of the "abstract machine tuple".

  pub tree: LReg<Tree>,
  pub ret:  LReg<ZKRetItem>,

  // NB: condition code register for stepping through if-elif-else sub-blocks
  // as separate sentences; each sub-block, if run, "returns" a condition code
  // True, otherwise False, and any subsequent sub-block only runs if the prev
  // condition code register is False.
  pub cond: LReg<ZKCondReg>,

  // TODO
  pub halt: LReg<ZKHaltReg>,
  //pub exc:  LReg<ZKExceptItem>,
  //pub yield_: LReg<ZKMachYield>,

  // TODO: other heavier "data" should go in the machine "state".

  //pub env:  _,
  //pub ctr:  _,
}

impl ZKntMachCursor {
  pub fn init() -> ZKntMachCursor {
    ZKntMachCursor{
      tree: LReg::init(),
      ret:  LReg::init(),
      cond: LReg::init(),
      halt: LReg::init(),
    }
  }

  pub fn init_from_tree(tree: Tree) -> ZKntMachCursor {
    ZKntMachCursor{
      tree: LReg::init_(tree),
      ret:  LReg::init(),
      cond: LReg::init(),
      halt: LReg::init(),
    }
  }
}

// NB: ZKntMachCursor too heavy for prev/next; do not need to stash
// the whole machine tuple.
#[derive(Clone, Debug)]
pub struct ZKPartialItem {
  pub tree: Tree,
  pub ret:  Option<ZKRetItem>,
}

/*#[derive(Clone, Copy, Debug)]
pub enum ZKTupleLabel {
  _Term,
  IdentTerm,
  _Sent,
  JustSent,
  PassSent,
}

impl ZKTupleLabel {
  pub fn from_term(term: &TreeTerm_) -> ZKTupleLabel {
    match term {
      &TreeTerm_::Ident(..) => ZKTupleLabel::IdentTerm,
      _ => ZKTupleLabel::_Term
    }
  }

  pub fn from_sent(sent: &TreeSent_) -> ZKTupleLabel {
    match sent {
      &TreeSent_::Just(..) => ZKTupleLabel::JustSent,
      &TreeSent_::Pass(..) => ZKTupleLabel::PassSent,
      _ => ZKTupleLabel::_Sent
    }
  }
}*/

pub struct ZKTermExt;

impl ZKTermExt {
  pub fn max_arity_from_term(term: &TreeTerm_) -> Result<usize, ZKMachCheck> {
    Ok(match term {
      &TreeTerm_::Ident(..) => 0,
      &TreeTerm_::Query(..) |
      &TreeTerm_::PQuery(..) => 1,
      &TreeTerm_::Attr(..) |
      &TreeTerm_::Equal(..) => 2,
      &TreeTerm_::Apply(ref _span, ref tup) => {
        tup.len()
      }
      &TreeTerm_::ListLit(ref _span, ref tup) => {
        tup.len()
      }
      _ => {
        return Bot2(format!("max_arity_from_term: {:?}", term));
      }
    })
  }

  pub fn max_arity_from_sent(sent: &TreeSent_) -> Result<usize, ZKMachCheck> {
    Ok(match sent {
      &TreeSent_::Pass(..) => 0,
      &TreeSent_::Just(..) |
      &TreeSent_::Raise(..) => 1,
      &TreeSent_::If(..) => {
        2
      }
      _ => {
        return Bot2(format!("max_arity_from_sent: {:?}", sent));
      }
    })
  }
}

#[derive(Clone, Debug)]
pub enum ZKntMachPartialEval {
  Empty,
  //Tuple{label: ZKTupleLabel, items: Vec<ZKPartialItem>},
  Tuple{max_arity: usize, items: Vec<ZKPartialItem>},
  // FIXME
  Block{offset: usize, ret_offset: usize, ret: Option<ZKRetItem>},
  // TODO
  //Unify{},
}

impl ZKntMachPartialEval {
  pub fn arity(&self) -> usize {
    match self {
      &ZKntMachPartialEval::Empty => {
        // TODO
        0
      }
      &ZKntMachPartialEval::Tuple{ref items, ..} => {
        let a = items.len();
        match a {
          0 => 0,
          _ => {
            if items[a-1].ret.is_none() {
              // TODO: assert items[a-2].ret.is_some().
              a - 1
            } else {
              a
            }
          }
        }
      }
      &ZKntMachPartialEval::Block{offset, ret_offset, ..} => {
        // TODO
        //offset
        ret_offset
      }
    }
  }

  pub fn _set_ret(&mut self, new_ret: Option<ZKRetItem>) -> Result<(), ()> {
    match self {
      &mut ZKntMachPartialEval::Empty => {
        unreachable!();
      }
      &mut ZKntMachPartialEval::Tuple{ref mut items, ..} => {
        if items.len() <= 0 {
          return Err(());
        }
        items.last_mut().unwrap().ret = new_ret;
      }
      &mut ZKntMachPartialEval::Block{offset, ret_offset, ref mut ret} => {
        if offset <= 0 {
          return Err(());
        }
        *ret = new_ret;
      }
    }
    Ok(())
  }

  pub fn _fill_ret(&mut self, new_ret: ZKRetItem) -> Result<(), ()> {
    match self {
      &mut ZKntMachPartialEval::Empty => {
        // TODO
        //unreachable!();
      }
      &mut ZKntMachPartialEval::Tuple{ref mut items, ..} => {
        if items.len() <= 0 {
          return Err(());
        }
        // TODO: must be None.
        items.last_mut().unwrap().ret = Some(new_ret);
      }
      &mut ZKntMachPartialEval::Block{offset, ref mut ret_offset, ref mut ret} => {
        if offset <= 0 {
          return Err(());
        }
        if offset <= *ret_offset {
          return Err(());
        } else {
          *ret_offset += 1;
          // TODO: must be None.
          *ret = Some(new_ret);
        }
      }
    }
    Ok(())
  }
}

/*impl From<(TreeRef, ZKRetItem)> ZKPartialItem {
}*/

#[derive(Clone, Copy, Debug)]
pub enum ZKMachSignal {
  _Nil,
  Yield,
}

impl From<()> for ZKMachSignal {
  fn from(_: ()) -> ZKMachSignal {
    ZKMachSignal::_Nil
  }
}

pub enum ZKMachYield {
  // TODO
}

/*#[derive(Clone, Debug)]
pub struct ZKFunValueCtx {
  pub ictr: u32,
  pub depth: u16,
  pub bits: u8,
  // TODO: general "partial evaluation".
  pub prev: ZKntMachPartialEval,
}*/

pub trait ZKFunValueImpl: Debug {
  fn _eval(&self, /*ctx: ZKFunValueCtx,*/ ctl: &mut ZKntMachControlReg, prev: &ZKntMachPartialEval, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>>;
}

#[derive(Debug, Default)]
pub struct ChoiceBuiltinFun {
}

impl ZKFunValueImpl for ChoiceBuiltinFun {
  fn _eval(&self, ctl: &mut ZKntMachControlReg, prev: &ZKntMachPartialEval, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    ctl._set_fail_bit();
    if state.tap.on() {
      let depth = if ctl._get_eval_bit() {
        ctl.depth + 1
      } else {
        ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      let buf = state.tap.buf.as_mut().unwrap();
      writeln!(buf, "! choice").or_else(|e| e.chk())?;
    }
    Err(Ok(ZKMachSignal::Yield))
  }
}

#[derive(Debug, Default)]
pub struct FailBuiltinFun {
}

impl ZKFunValueImpl for FailBuiltinFun {
  fn _eval(&self, /*ctx: ZKFunValueCtx,*/ ctl: &mut ZKntMachControlReg, prev: &ZKntMachPartialEval, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    ctl._set_fail_bit();
    if state.tap.on() {
      let depth = if ctl._get_eval_bit() {
        ctl.depth + 1
      } else {
        ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      let buf = state.tap.buf.as_mut().unwrap();
      writeln!(buf, "! fail").or_else(|e| e.chk())?;
    }
    Err(Ok(ZKMachSignal::Yield))
  }
}

#[derive(Debug, Default)]
pub struct InputBuiltinFun {
}

#[derive(Debug, Default)]
pub struct PrintBuiltinFun {
}

impl ZKFunValueImpl for PrintBuiltinFun {
  fn _eval(&self, /*ctx: ZKFunValueCtx,*/ ctl: &mut ZKntMachControlReg, prev: &ZKntMachPartialEval, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    if state.tap.on() {
      let depth = if ctl._get_eval_bit() {
        ctl.depth + 1
      } else {
        ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      let buf = state.tap.buf.as_mut().unwrap();
      writeln!(buf, "! print").or_else(|e| e.chk())?;
    }
    println!("hello world!");
    ok()
  }
}

#[derive(Debug)]
pub enum ZKValue {
  Fun(Rc<dyn ZKFunValueImpl + 'static>),
}

pub struct ZKntMachValueDomState {
  tab:  BTreeMap<SNum, ZKValue>,
}

impl ZKntMachValueDomState {
  pub fn init() -> ZKntMachValueDomState {
    ZKntMachValueDomState{
      tab:  BTreeMap::new(),
    }
  }
}

pub struct ZKntMachTAPState {
  buf:  Option<Box<dyn Write>>,
}

impl ZKntMachTAPState {
  pub fn init() -> ZKntMachTAPState {
    ZKntMachTAPState{
      buf:  None,
    }
  }

  pub fn on(&self) -> bool {
    self.buf.is_some()
  }

  pub fn _write_indent(&mut self, depth: u16) -> Result<(), IoError> {
    for _ in 0 .. depth {
      write!(self.buf.as_mut().unwrap(), " ")?;
    }
    Ok(())
  }
}

pub struct ZKTableEntry {
  label: Option<SafeStr>,
}

#[derive(Clone, Debug)]
pub enum ZKTabledTree {
  _AnyTree,
  _AnyTerm,
  _AnySent,
  IdentTerm(Span, TreeIdent),
  AttrTerm(Span, SNum, TreeIdent),
  QueryTerm(Span, SNum),
  PQueryTerm(Span, SNum),
  EqualTerm(Span, SNum, SNum),
  ApplyTerm(Span, Vec<SNum>),
  PassSent(Span),
  JustSent(Span, SNum),
  RaiseSent(Span, SNum),
  Mod(Span, Vec<SNum>),
}

pub struct ZKntMachState {
  // TODO: the continuation machine is fully deterministic,
  // so separate certain monotone state (e.g. counters) here.
  // NB: the state here should be "easy" to rollback/undo.

  ctr:  RawSNum,
  rctr: RawSNum,

  // TODO
  init_tree: LReg<Tree>,
  //tree_num: BTreeMap<TreeRef, SNum>,
  tree_tab: BTreeMap<SNum, ZKTabledTree>,

  //tableau:  BTreeMap<SNum, ()>,
  pt_tab:   BTreeMap<SNum, ZKTableEntry>,
  tup_tab:  BTreeMap<Box<[SNum]>, ZKTableEntry>,

  // TODO: inject tree into code w/ attendant code num (or SNum).
  global_anon_env: BTreeSet<SNum>,
  global_env: BTreeMap<SafeStr, SNum>,
  //lexical_envs: BTreeMap<(Span, SafeStr), SNum>,

  //attr_env: _,

  // TODO
  //unify_trail: _,

  vdom: ZKntMachValueDomState,

  tap:  ZKntMachTAPState,
}

impl ZKntMachState {
  pub fn init(init_tree: Tree) -> ZKntMachState {
    ZKntMachState{
      ctr:  0,
      rctr: 0,
      init_tree: LReg::init_(init_tree),
      //tree_num: BTreeMap::new(),
      tree_tab: BTreeMap::new(),
      pt_tab:   BTreeMap::new(),
      tup_tab:  BTreeMap::new(),
      global_anon_env: BTreeSet::new(),
      global_env: BTreeMap::new(),
      vdom: ZKntMachValueDomState::init(),
      tap:  ZKntMachTAPState::init(),
    }
  }

  pub fn set_tap_buffer(&mut self, buf: Box<dyn Write>) -> Option<Box<dyn Write>> {
    let prev_buf = self.tap.buf.take();
    self.tap.buf = Some(buf);
    prev_buf
  }

  pub fn unset_tap_buffer(&mut self) -> Option<Box<dyn Write>> {
    let prev_buf = self.tap.buf.take();
    prev_buf
  }

  pub fn _rev_fresh(&mut self) -> SNum {
    let next = self.rctr - 1;
    self.rctr = next;
    SNum(next)
  }

  pub fn _fresh(&mut self) -> SNum {
    let next = self.ctr + 1;
    self.ctr = next;
    SNum(next)
  }
}

#[derive(Clone, Copy, Debug)]
pub enum ZKResumeStatus {
  _Bot,
  Halt,
  Stop,
  Yield,
  Break,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum ZKMachPhase {
  Init,
  Step,
  Eval,
  Backup,
}

pub type ZKntMachLogRef = Option<Box<ZKntMachLog>>;

#[derive(Clone, Debug)]
pub struct ZKntMachLog {
  //pub ictr: u32,
  //pub bits: u8,
  //pub phase: ZKMachPhase,
  //pub depth: u16,
  pub ctl:  ZKntMachControlReg,
  pub back: ZKntMachLogRef,
  pub cur:  ZKntMachCursor,
  // TODO: general "partial evaluation".
  pub prev: ZKntMachPartialEval,
}

impl ZKntMachLog {
  pub fn init() -> ZKntMachLog {
    ZKntMachLog{
      ctl:  ZKntMachControlReg::init(),
      back: None,
      cur:  ZKntMachCursor::init(),
      prev: ZKntMachPartialEval::Block{offset: 0, ret_offset: 0, ret: None},
    }
  }

  pub fn get_lock_bit(&self) -> bool {
    (self.ctl.bits & 1) != 0
  }

  pub fn set_lock_bit(&mut self) {
    self.ctl.bits |= 1;
  }

  pub fn unset_lock_bit(&mut self) {
    self.ctl.bits &= !1;
  }

  pub fn get_eval_bit(&self) -> bool {
    (self.ctl.bits & 2) != 0
  }

  pub fn set_eval_bit(&mut self) {
    self.ctl.bits |= 2;
  }

  pub fn unset_eval_bit(&mut self) {
    self.ctl.bits &= !2;
  }

  pub fn get_match_bit(&self) -> bool {
    (self.ctl.bits | 0x10) != 0
  }

  pub fn set_match_bit(&mut self) {
    self.ctl.bits |= 0x10;
  }

  pub fn unset_match_bit(&mut self) {
    self.ctl.bits &= !0x10;
  }

  pub fn set_fail_bit(&mut self) {
    self.ctl.bits |= 0x80;
  }

  pub fn _insert_label(&mut self, x: SNum, label: SafeStr, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    state.global_anon_env.insert(x);
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?} # {:?}", x, &label).or_else(|e| e.chk())?;
      }
    }
    // TODO: doing this after tap is kinda klugy.
    state.pt_tab.insert(x, ZKTableEntry{label: label.into()});
    ok()
  }

  //pub fn _insert_tup_label<Tup: AsRef<[SNum]>>(&mut self, tup: Tup, label: SafeStr, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {}
  pub fn _insert_tup<Tup: AsRef<[SNum]>>(&mut self, tup: Tup, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?}", tup.as_ref()).or_else(|e| e.chk())?;
      }
    }
    // TODO: doing this after tap is kinda klugy.
    state.tup_tab.insert(tup.as_ref().into(), ZKTableEntry{label: None});
    ok()
  }

  pub fn rfresh_label(&mut self, label: SafeStr, state: &mut ZKntMachState) -> Result<SNum, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    let x = state._rev_fresh();
    state.global_anon_env.insert(x);
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?} # {:?}", x, &label).or_else(|e| e.chk())?;
      }
    }
    // TODO: doing this after tap is kinda klugy.
    state.pt_tab.insert(x, ZKTableEntry{label: label.into()});
    Ok(x)
  }

  pub fn rfresh_ident<Id: AsRef<SafeStr>>(&mut self, ident: Id, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    let ident = ident.as_ref();
    let x = state._rev_fresh();
    state.global_env.insert(ident.clone(), x);
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "> {:?}", ident).or_else(|e| e.chk())?;
      }
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?}", x).or_else(|e| e.chk())?;
      }
    }
    // TODO: doing this after tap is kinda klugy.
    state.pt_tab.insert(x, ZKTableEntry{label: None});
    ok()
  }

  pub fn rfresh_ident_value<Id: AsRef<SafeStr>>(&mut self, ident: Id, value: ZKValue, state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    let ident = ident.as_ref();
    let x = state._rev_fresh();
    state.global_env.insert(ident.clone(), x);
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "> {:?} ==> {:?}", ident, &value).or_else(|e| e.chk())?;
      }
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?}", x).or_else(|e| e.chk())?;
      }
    }
    // TODO: doing this after tap is kinda klugy.
    state.vdom.tab.insert(x, value);
    state.pt_tab.insert(x, ZKTableEntry{label: None});
    ok()
  }

  pub fn find_ident<Id: AsRef<SafeStr>>(&mut self, ident: Id, state: &mut ZKntMachState) -> Result<SNum, Result<ZKMachSignal, ZKMachCheck>> {
    let ident = ident.as_ref();
    // TODO: lexical scope.
    let x = match state.global_env.get(ident) {
      None => {
        let x = state._fresh();
        state.global_env.insert(ident.clone(), x);
        x
      }
      Some(&x) => x
    };
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "> {:?}", ident).or_else(|e| e.chk())?;
      }
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?}", x).or_else(|e| e.chk())?;
      }
    }
    Ok(x)
  }

  pub fn unify_terms(&mut self, lterm: (), rterm: (), state: &mut ZKntMachState) -> Result<()/*SNum*/, Result<ZKMachSignal, ZKMachCheck>> {
    // TODO
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "> {:?} = {:?}", 0, 0).or_else(|e| e.chk())?;
      }
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, "< {:?}", 0).or_else(|e| e.chk())?;
      }
    }
    ok()
  }

  pub fn _step_query(&mut self, term: TreeTermRef_) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    //println!("DEBUG: ZKntMachLog::_step_query: term = {:?}", &term);
    unimpl()
    /*
    let label = ZKTupleLabel::from_term(&*term);
    let tree = QTree::ConsTerm(term);
    match &mut self.prev {
      &mut ZKntMachPartialEval::Tuple{ref label, ref mut items} => {
        //println!("DEBUG: ZKntMachLog::_step_term: prev: label = {:?}", label);
        items.push(ZKPartialItem{
          tree: tree.clone(),
          ret:  None,
        });
      }
      _ => {
        return bot();
      }
    }
    let step_cur = ZKntMachCursor::init_query(term);
    let this_back = replace(&mut self.back, None);
    let this_cur = replace(&mut self.cur, step_cur);
    let this_prev = replace(&mut self.prev, ZKntMachPartialEval::Tuple{label, items: Vec::new()});
    let back = ZKntMachLog{
      ctl:  self.ctl,
      back: this_back,
      cur:  this_cur,
      prev: this_prev,
    };
    self.ctl.depth += 1;
    self.back = Some(back.into());
    ok()
    */
  }

  pub fn _step_term(&mut self, term: TreeTermRef_) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    self._step_term_(term, false)
  }

  pub fn _step_term_(&mut self, term: TreeTermRef_, match_: bool) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    //println!("DEBUG: ZKntMachLog::_step_term: term = {:?}", &term);
    //let label = ZKTupleLabel::from_term(&*term);
    let max_arity = ZKTermExt::max_arity_from_term(&*term).or_else(|e| Err(Err(e)))?;
    let tree = Tree::Term(term);
    match &mut self.prev {
      &mut ZKntMachPartialEval::Tuple{ref mut items, ..} => {
        //println!("DEBUG: ZKntMachLog::_step_term: prev: label = {:?}", label);
        items.push(ZKPartialItem{
          tree: tree.clone(),
          ret:  None,
        });
      }
      _ => {
        return bot();
      }
    }
    let step_cur = ZKntMachCursor::init_from_tree(tree);
    let this_back = replace(&mut self.back, None);
    let this_cur = replace(&mut self.cur, step_cur);
    let this_prev = replace(&mut self.prev, ZKntMachPartialEval::Tuple{max_arity, items: Vec::new()});
    let back = ZKntMachLog{
      ctl:  self.ctl,
      back: this_back,
      cur:  this_cur,
      prev: this_prev,
    };
    self.back = Some(back.into());
    self.ctl.depth += 1;
    if match_ {
      self.ctl._set_match_bit();
    }
    ok()
  }

  pub fn _step_sent(&mut self, sent: TreeSentRef_) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    //println!("DEBUG: ZKntMachLog::_step_sent: sent = {:?}", &sent);
    //let label = ZKTupleLabel::from_sent(&*sent);
    let max_arity = ZKTermExt::max_arity_from_sent(&*sent).or_else(|e| Err(Err(e)))?;
    let tree = Tree::Sent(sent);
    match &mut self.prev {
      &mut ZKntMachPartialEval::Block{ref mut offset, ..} => {
        /*items.push(ZKPartialItem{
          tree: tree.clone(),
          ret:  None,
        });*/
        *offset += 1;
      }
      _ => {
        return bot();
      }
    }
    let step_cur = ZKntMachCursor::init_from_tree(tree);
    let this_back = replace(&mut self.back, None);
    let this_cur = replace(&mut self.cur, step_cur);
    let this_prev = replace(&mut self.prev, ZKntMachPartialEval::Tuple{max_arity, items: Vec::new()});
    let back = ZKntMachLog{
      ctl:  self.ctl,
      back: this_back,
      cur:  this_cur,
      prev: this_prev,
    };
    self.back = Some(back.into());
    self.ctl.depth += 1;
    ok()
  }

  pub fn _step_block(&mut self, block: Vec<TreeSentRef_>) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    let tree = match block.len() {
      0 => {
        return bot();
      }
      1 => {
        Tree::Sent(block[0].clone())
      }
      _ => {
        Tree::Block(block)
      }
    };
    let step_cur = ZKntMachCursor::init_from_tree(tree);
    let this_back = replace(&mut self.back, None);
    let this_cur = replace(&mut self.cur, step_cur);
    let this_prev = replace(&mut self.prev, ZKntMachPartialEval::Block{offset: 0, ret_offset: 0, ret: None});
    let back = ZKntMachLog{
      ctl:  self.ctl,
      back: this_back,
      cur:  this_cur,
      prev: this_prev,
    };
    self.back = Some(back.into());
    self.ctl.depth += 1;
    ok()
  }

  pub fn _step(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    println!("DEBUG: ZKntMachLog::_step: ictr     = {:?}", self.ctl.ictr);
    println!("DEBUG: ZKntMachLog::_step: depth    = {:?}", self.ctl.depth);
    println!("DEBUG: ZKntMachLog::_step: back     = {:?}", &self.back);
    println!("DEBUG: ZKntMachLog::_step: cur tree = {:?}", &self.cur.tree);
    println!("DEBUG: ZKntMachLog::_step: cur ret  = {:?}", &self.cur.ret);
    println!("DEBUG: ZKntMachLog::_step: prev     = {:?}", &self.prev);
    match &self.cur.tree {
      &LReg::Fill(Tree::Term(ref term)) => {
        match &**term {
          &TreeTerm_::Query(ref span, ref query) => {
            // TODO: proving a first order term.
            return unimpl();
          }
          &TreeTerm_::PQuery(ref span, ref query) => {
            // TODO: proving a first order term,
            // w/ negation as failure.
            for _ in 0 .. 2 {
              let prev_arity = self.prev.arity();
              match prev_arity {
                0 => {
                  if self.cur.ret._is_fill() {
                    let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
                    self.prev._fill_ret(ret_item).or_else(|_| bot())?;
                    continue;
                  }
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "+ PQuery {}", prev_arity).or_else(|e| e.chk())?;
                  }
                  // TODO: set "match" mode in the step.
                  //self.set_match_bit();
                  return self._step_term_(query.clone(), true);
                }
                1 => {
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "- PQuery {}", prev_arity).or_else(|e| e.chk())?;
                  }
                  // TODO: should be unnecessary (w/ above).
                  //self.unset_match_bit();
                  return Err(ok());
                }
                _ => {
                  return bot();
                }
              }
            }
          }
          &TreeTerm_::Ident(ref span, ref id) => {
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "- Ident {:?}@{:?}", id, span).or_else(|e| e.chk())?;
            }
            return Err(ok());
          }
          &TreeTerm_::Equal(ref span, ref lterm, ref rterm) => {
            for _ in 0 .. 2 {
              let prev_arity = self.prev.arity();
              match prev_arity {
                0 => {
                  if self.cur.ret._is_fill() {
                    let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
                    self.prev._fill_ret(ret_item).or_else(|_| bot())?;
                    continue;
                  }
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "+ Equal {}", prev_arity).or_else(|e| e.chk())?;
                  }
                  return self._step_term(lterm.clone());
                }
                1 => {
                  if self.cur.ret._is_fill() {
                    let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
                    self.prev._fill_ret(ret_item).or_else(|_| bot())?;
                    continue;
                  }
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "+ Equal {}", prev_arity).or_else(|e| e.chk())?;
                  }
                  return self._step_term(rterm.clone());
                }
                2 => {
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "- Equal {}", prev_arity).or_else(|e| e.chk())?;
                  }
                  return Err(ok());
                }
                _ => {
                  return bot();
                }
              }
            }
          }
          &TreeTerm_::BindL(ref span, ref lterm, ref rterm) => {
          }
          &TreeTerm_::BindR(ref span, ref lterm, ref rterm) => {
          }
          &TreeTerm_::Apply(ref span, ref tup) => {
            let arity = tup.len();
            for _ in 0 .. 2 {
              let prev_arity = self.prev.arity();
              if prev_arity < arity {
                if self.cur.ret._is_fill() {
                  let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
                  self.prev._fill_ret(ret_item).or_else(|_| bot())?;
                  continue;
                }
                if state.tap.on() {
                  state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                  let buf = state.tap.buf.as_mut().unwrap();
                  writeln!(buf, "+ Apply {}", prev_arity).or_else(|e| e.chk())?;
                }
                return self._step_term(tup[prev_arity].clone());
              } else if prev_arity == arity {
                if state.tap.on() {
                  state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                  let buf = state.tap.buf.as_mut().unwrap();
                  writeln!(buf, "- Apply {}", prev_arity).or_else(|e| e.chk())?;
                }
                return Err(ok());
              } else {
                return bot();
              }
            }
          }
          _ => {
          }
        }
      }
      &LReg::Fill(Tree::Sent(ref sent)) => {
        match &**sent {
          &TreeSent_::Just(_, ref term) => {
            for _ in 0 .. 2 {
              let prev_arity = self.prev.arity();
              //println!("DEBUG: ZKntMachLog::step:   Sent: Just: prev arity = {}", prev_arity);
              match prev_arity {
                0 => {
                  if self.cur.ret._is_fill() {
                    let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
                    self.prev._fill_ret(ret_item).or_else(|_| bot())?;
                    continue;
                  }
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "+ Just").or_else(|e| e.chk())?;
                  }
                  return self._step_term(term.clone());
                }
                1 => {
                  if state.tap.on() {
                    state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
                    let buf = state.tap.buf.as_mut().unwrap();
                    writeln!(buf, "- Just").or_else(|e| e.chk())?;
                  }
                  return Err(ok());
                }
                _ => {
                  return bot();
                }
              }
            }
          }
          &TreeSent_::Pass(..) => {
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "- Pass").or_else(|e| e.chk())?;
            }
            return Err(ok());
          }
          &TreeSent_::If(_, ref cond, ref body) => {
            match self.prev.arity() {
              0 => {
                return self._step_term(cond.clone());
              }
              1 => {
                // FIXME: condition.
                //if state.find_truthy(cond) {
                self._step_block(body.clone())?;
                //}
              }
              2 => {
                /*let ret = replace(&mut self.cur.ret, LReg::Remove);
                match ret {
                  LReg::Empty => {
                    return bot();
                  }
                  LReg::Fill(ret) => {
                    return Err(Ok(ret));
                  }
                  LReg::Remove => {
                    return bot();
                  }
                }*/
                return Err(ok());
              }
              _ => {
                return bot();
              }
            }
          }
          &TreeSent_::Elif(..) => {
          }
          &TreeSent_::Else(..) => {
          }
          _ => {
          }
        }
      }
      &LReg::Fill(Tree::Block(ref block)) => {
        for _ in 0 .. 2 {
          let prev_arity = self.prev.arity();
          if prev_arity < block.len() {
            if self.cur.ret._is_fill() {
              let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
              self.prev._fill_ret(ret_item).or_else(|_| bot())?;
              continue;
            }
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "+ Block").or_else(|e| e.chk())?;
            }
            return self._step_sent(block[prev_arity].clone());
          } else if prev_arity == block.len() {
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "- Block").or_else(|e| e.chk())?;
            }
            return Err(ok());
          } else {
            return bot();
          }
        }
      }
      &LReg::Fill(Tree::Mod(ref mod_)) => {
        //println!("DEBUG: ZKntMachLog::step:   Mod: prev = {:?}", &self.prev);
        for _ in 0 .. 2 {
          let prev_arity = self.prev.arity();
          //println!("DEBUG: ZKntMachLog::_step:   Mod: prev arity = {} prev = {:?}", prev_arity, &self.prev);
          if prev_arity < mod_.body.len() {
            if self.cur.ret._is_fill() {
              let ret_item = self.cur.ret._remove().or_else(|_| bot())?;
              self.prev._fill_ret(ret_item).or_else(|_| bot())?;
              continue;
            }
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "+ Mod {}", prev_arity).or_else(|e| e.chk())?;
            }
            return self._step_sent(mod_.body[prev_arity].clone());
          } else if prev_arity == mod_.body.len() {
            if state.tap.on() {
              state.tap._write_indent(self.ctl.depth).or_else(|e| e.chk())?;
              let buf = state.tap.buf.as_mut().unwrap();
              writeln!(buf, "- Mod {}", prev_arity).or_else(|e| e.chk())?;
            }
            return Err(ok());
          } else {
            return bot();
          }
        }
      }
      _ => {
      }
    }
    unimpl()
  }

  pub fn _eval(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    println!("DEBUG: ZKntMachLog::_eval: ictr     = {:?}", self.ctl.ictr);
    println!("DEBUG: ZKntMachLog::_eval: cur ret  = {:?}", &self.cur.ret);
    println!("DEBUG: ZKntMachLog::_eval: prev     = {:?}", &self.prev);
    // TODO: this is the tree interpretation kernel.
    match &self.cur.tree.clone() {
      &LReg::Fill(Tree::Term(ref term)) => {
        match &**term {
          &TreeTerm_::Query(ref span, ref query) => {
            // TODO: proving a first order term.
            return unimpl();
          }
          &TreeTerm_::PQuery(ref span, ref query) => {
            // TODO: proving a first order term,
            // w/ negation as failure.
            return unimpl();
            /*
            for _ in 0 .. 2 {
              match &self.prev {
                &ZKntMachPartialEval::Tuple{..} => {
                  // TODO
                }
                &ZKntMachPartialEval::Unify{..} => {
                  // TODO
                }
                _ => return bot()
              }
            }
            */
          }
          &TreeTerm_::Ident(ref span, ref ident) => {
            // TODO
            let x = self.find_ident(ident, state)?;
            self.cur.ret._fill(ZKRetItem::Term(x)).or_else(|_| bot())?;
            return Err(ok());
          }
          &TreeTerm_::Equal(ref span, ..) => {
            // TODO
            match &self.prev {
              &ZKntMachPartialEval::Tuple{ref items, ..} => {
                let _ = items[0].ret;
                let _ = items[1].ret;
                self.unify_terms((), (), state)?;
              }
              _ => return bot()
            }
            self.cur.ret._fill(ZKRetItem::_Term).or_else(|_| bot())?;
            return Err(ok());
          }
          &TreeTerm_::BindL(ref span, ..) => {
          }
          &TreeTerm_::BindR(ref span, ..) => {
          }
          &TreeTerm_::Apply(ref span, ..) => {
            // TODO
            match &self.prev {
              &ZKntMachPartialEval::Tuple{ref items, ..} => {
                match &items[0].ret {
                  &None => {
                    return bot();
                  }
                  &Some(ZKRetItem::Term(x)) => {
                    match state.vdom.tab.get(&x) {
                      Some(&ZKValue::Fun(ref inner)) => {
                        let inner = inner.clone();
                        inner._eval(&mut self.ctl, &self.prev, state)?;
                      }
                      _ => {}
                    }
                  }
                  _ => {}
                }
              }
              _ => return bot()
            }
            self.cur.ret._fill(ZKRetItem::_Term).or_else(|_| bot())?;
            return Err(ok());
          }
          _ => {
          }
        }
      }
      &LReg::Fill(Tree::Sent(ref sent)) => {
        match &**sent {
          &TreeSent_::Just(..) => {
            //println!("DEBUG: ZKntMachLog::_eval:   Sent: Just: prev arity = {:?}", self.prev.arity());
            match self.prev.arity() {
              1 => {
                self.cur.ret._fill(ZKRetItem::_Sent).or_else(|_| bot())?;
                return Err(ok());
              }
              _ => {
                return bot();
              }
            }
          }
          &TreeSent_::Pass(..) => {
            self.cur.ret._fill(ZKRetItem::_Sent).or_else(|_| bot())?;
            return Err(ok());
          }
          &TreeSent_::If(..) => {
            match self.prev.arity() {
              0 => {
                // NB: mach check.
                //return bot();
              }
              1 => {
                let ret = replace(&mut self.cur.ret, LReg::Remove);
                match ret {
                  //LReg::Empty |
                  LReg::Remove => {
                    // TODO: case.
                    //self.prev.last_mut().unwrap().ret = None;
                    if self.prev._set_ret(None).is_err() {
                      return bot();
                    }
                  }
                  LReg::Fill(ret) => {
                    //self.prev.last_mut().unwrap().ret = Some(ret);
                    if self.prev._set_ret(Some(ret)).is_err() {
                      return bot();
                    }
                  }
                  /*LReg::Remove => {
                    // NB: mach check.
                    //return bot();
                  }*/
                }
              }
              2 => {
                let ret = replace(&mut self.cur.ret, LReg::Remove);
                match ret {
                  //LReg::Empty |
                  LReg::Remove => {
                    // TODO: case.
                    //self.prev.last_mut().unwrap().ret = None;
                    if self.prev._set_ret(None).is_err() {
                      return bot();
                    }
                  }
                  LReg::Fill(ret) => {
                    //self.prev.last_mut().unwrap().ret = Some(ret);
                    if self.prev._set_ret(Some(ret)).is_err() {
                      return bot();
                    }
                  }
                  /*LReg::Remove => {
                    // NB: mach check.
                    //return bot();
                  }*/
                }
              }
              _ => {
                // NB: mach check.
                //return bot();
              }
            }
          }
          &TreeSent_::Elif(..) => {
          }
          &TreeSent_::Else(..) => {
          }
          _ => {
          }
        }
      }
      &LReg::Fill(Tree::Block(ref body)) => {
      }
      &LReg::Fill(Tree::Mod(ref mod_)) => {
        let prev_arity = self.prev.arity();
        if prev_arity == mod_.body.len() {
          return Err(ok());
        } else {
          return bot();
        }
      }
      _ => {
      }
    }
    unimpl()
  }

  pub fn _backup(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    println!("DEBUG: ZKntMachLog::_backup: ictr     = {:?}", self.ctl.ictr);
    println!("DEBUG: ZKntMachLog::_backup: back     = {:?}", &self.back);
    println!("DEBUG: ZKntMachLog::_backup: cur ret  = {:?}", &self.cur.ret);
    println!("DEBUG: ZKntMachLog::_backup: prev     = {:?}", &self.prev);
    /*if self.cur.halt._is_fill() {
      return Err(ok());
    }*/
    let back = replace(&mut self.back, None);
    if back.is_none() {
      // NB: safely ("linearly") set cur halt.
      self.cur.halt._fill(ZKHaltReg::Stop).or_else(|_| bot())?;
      return Err(ok());
    }
    let mut back = back.unwrap();
    // NB: carry control registers through backup.
    // NB: _do not_ carry the ictr through backup.
    // NB: using `swap` for efficient context replacement.
    self.ctl.depth = back.ctl.depth;
    swap(&mut self.back, &mut back.back);
    swap(&mut self.cur.tree, &mut back.cur.tree);
    swap(&mut self.prev, &mut back.prev);
    ok()
  }

  pub fn _coldstart_prelude(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    let label = into_safe_str("Nil");
    self._insert_label(nil(), label, state)?;
    let label = into_safe_str("Lb");
    self._insert_label(lb(), label, state)?;
    let label = into_safe_str("Ub");
    self._insert_label(ub(), label, state)?;
    let label = into_safe_str("True");
    self.rfresh_label(label, state)?;
    let label = into_safe_str("False");
    self.rfresh_label(label, state)?;
    let label = into_safe_str("Unk");
    self.rfresh_label(label, state)?;
    let label = into_safe_str("Bot");
    self.rfresh_label(label, state)?;
    let label = into_safe_str("None");
    self.rfresh_label(label, state)?;
    let label = into_safe_str("__isaclass__");
    let isaclass = self.rfresh_label(label, state)?;
    let label = into_safe_str("__instanceof__");
    let instance = self.rfresh_label(label, state)?;
    let label = into_safe_str("__meta_class__");
    let meta_class = self.rfresh_label(label, state)?;
    let label = into_safe_str("__attribute__");
    let _attr = self.rfresh_label(label, state)?;
    let label = into_safe_str("object");
    let object = self.rfresh_label(label, state)?;
    let label = into_safe_str("object_meta");
    let object_meta = self.rfresh_label(label, state)?;
    let label = into_safe_str("meta_object");
    let meta_object = self.rfresh_label(label, state)?;
    let label = into_safe_str("meta_object_meta");
    let meta_object_meta = self.rfresh_label(label, state)?;
    self._insert_tup(&[instance, object, object], state)?;
    self._insert_tup(&[instance, object_meta, meta_object], state)?;
    self._insert_tup(&[instance, meta_object, object], state)?;
    self._insert_tup(&[instance, meta_object_meta, meta_object], state)?;
    self._insert_tup(&[meta_class, object, object_meta], state)?;
    self._insert_tup(&[meta_class, meta_object, meta_object_meta], state)?;
    let ident = into_safe_str("Failure");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("LogicError");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("UnificationError");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("AttributeError");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("AssertionError");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("NotImplementedError");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("choice");
    let value = ZKValue::Fun(Rc::new(ChoiceBuiltinFun::default()) as _);
    self.rfresh_ident_value(ident, value, state)?;
    let ident = into_safe_str("fail");
    let value = ZKValue::Fun(Rc::new(FailBuiltinFun::default()) as _);
    self.rfresh_ident_value(ident, value, state)?;
    let ident = into_safe_str("input");
    let value = ZKValue::Fun(Rc::new(PrintBuiltinFun::default()) as _);
    self.rfresh_ident_value(ident, value, state)?;
    let ident = into_safe_str("print");
    let value = ZKValue::Fun(Rc::new(PrintBuiltinFun::default()) as _);
    self.rfresh_ident_value(ident, value, state)?;
    ok()
  }

  pub fn _table_tree(&mut self, tree: Tree, state: &mut ZKntMachState) -> Result<SNum, Result<ZKMachSignal, ZKMachCheck>> {
    let x = state._rev_fresh();
    let tree = match tree {
      Tree::Term(term) => {
        match &*term {
          &TreeTerm_::Ident(ref span, ref ident) => {
            ZKTabledTree::IdentTerm(span.clone(), ident.clone())
          }
          &TreeTerm_::Attr(ref span, ref lterm, ref ident) => {
            let lterm = self._table_tree(Tree::Term(lterm.clone()), state)?;
            ZKTabledTree::AttrTerm(span.clone(), lterm, ident.clone())
          }
          &TreeTerm_::Query(ref span, ref term) => {
            let term = self._table_tree(Tree::Term(term.clone()), state)?;
            ZKTabledTree::QueryTerm(span.clone(), term)
          }
          &TreeTerm_::PQuery(ref span, ref term) => {
            let term = self._table_tree(Tree::Term(term.clone()), state)?;
            ZKTabledTree::PQueryTerm(span.clone(), term)
          }
          &TreeTerm_::Equal(ref span, ref lterm, ref rterm) => {
            let lterm = self._table_tree(Tree::Term(lterm.clone()), state)?;
            let rterm = self._table_tree(Tree::Term(rterm.clone()), state)?;
            ZKTabledTree::EqualTerm(span.clone(), lterm, rterm)
          }
          &TreeTerm_::Apply(ref span, ref tup) => {
            let mut tup_ = Vec::with_capacity(tup.len());
            for term in tup.iter() {
              let t = self._table_tree(Tree::Term(term.clone()), state)?;
              tup_.push(t);
            }
            ZKTabledTree::ApplyTerm(span.clone(), tup_)
          }
          _ => {
            ZKTabledTree::_AnyTerm
          }
        }
      }
      Tree::Sent(sent) => {
        match &*sent {
          &TreeSent_::Pass(ref span) => {
            ZKTabledTree::PassSent(span.clone())
          }
          &TreeSent_::Just(ref span, ref term) => {
            let term = self._table_tree(Tree::Term(term.clone()), state)?;
            ZKTabledTree::JustSent(span.clone(), term)
          }
          &TreeSent_::Raise(ref span, ref term) => {
            let term = self._table_tree(Tree::Term(term.clone()), state)?;
            ZKTabledTree::RaiseSent(span.clone(), term)
          }
          _ => {
            ZKTabledTree::_AnySent
          }
        }
      }
      Tree::Mod(mod_) => {
        let mut body_ = Vec::with_capacity(mod_.body.len());
        for sent in mod_.body.iter() {
          let sent = self._table_tree(Tree::Sent(sent.clone()), state)?;
          body_.push(sent);
        }
        ZKTabledTree::Mod(mod_.span.clone(), body_)
      }
      _ => {
        ZKTabledTree::_AnyTree
      }
    };
    if state.tap.on() {
      let depth = if self.get_eval_bit() {
        self.ctl.depth + 1
      } else {
        self.ctl.depth
      };
      state.tap._write_indent(depth).or_else(|e| e.chk())?;
      {
        let buf = state.tap.buf.as_mut().unwrap();
        writeln!(buf, ". {:?} = {:?}", x, &tree).or_else(|e| e.chk())?;
      }
    }
    state.tree_tab.insert(x, tree);
    Ok(x)
  }

  pub fn _coldstart_init(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKMachSignal, ZKMachCheck>> {
    // TODO: actually, the cursor tree should initially never be fill.
    if !self.cur.tree._is_fill() {
      // TODO
      let tree = state.init_tree._remove().unwrap();
      self.cur.tree._fill(tree.clone()).unwrap();
      if state.tap.on() {
        let depth = if self.get_eval_bit() {
          self.ctl.depth + 1
        } else {
          self.ctl.depth
        };
        state.tap._write_indent(depth).or_else(|e| e.chk())?;
        {
          let buf = state.tap.buf.as_mut().unwrap();
          match &tree {
            &Tree::Mod(ref mod_) => {
              writeln!(buf, "> @{:?}", &mod_.span).or_else(|e| e.chk())?;
            }
            _ => {
              writeln!(buf, "> ???").or_else(|e| e.chk())?;
            }
          }
        }
        state.tap._write_indent(depth).or_else(|e| e.chk())?;
        {
          let buf = state.tap.buf.as_mut().unwrap();
          writeln!(buf, "< ").or_else(|e| e.chk())?;
        }
      }
      self._table_tree(tree, state)?;
    }
    /*let label = into_safe_str("__namespace__");
    self.rfresh_label(label, state)?;*/
    let ident = into_safe_str("__local_namespace__");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("__global_namespace__");
    self.rfresh_ident(ident, state)?;
    let ident = into_safe_str("__namespace__");
    self.rfresh_ident(ident, state)?;
    /*let label = into_safe_str("__main__");
    self.rfresh_label(label, state)?;*/
    /*let ident = into_safe_str("__main__");
    self.rfresh_ident(ident, state)?;*/
    let ident = into_safe_str("__name__");
    self.rfresh_ident(ident, state)?;
    ok()
  }

  #[inline]
  pub fn _resume_start(&mut self, state: &mut ZKntMachState, /*resume_arg: &mut LReg<_>*/) -> Result<(), Result<ZKResumeStatus, ZKMachCheck>> {
    // TODO: debug.
    if self.ctl.ictr >= 100 {
      println!("DEBUG: ZKntMachLog::_resume: break: ictr = {:?}", self.ctl.ictr);
      return Err(Ok(ZKResumeStatus::Break));
    }
    println!("DEBUG: ZKntMachLog::_resume: start: ictr = {:?}", self.ctl.ictr);
    // TODO: resume after yield (e.g. b/c of choice) may skip step and go
    // directly to eval; detect w/ lock == true, phase == Step.
    let save_start = true;
    let (save_phase, save_lock) = {
      if self.get_lock_bit() {
        if !save_start {
          return bot()
        }
        let save_phase = self.ctl.phase;
        let save_lock = self.get_lock_bit();
        (save_phase, save_lock)
      } else {
        let save_lock = self.get_lock_bit();
        self.set_lock_bit();
        self.ctl.phase = ZKMachPhase::Init;
        if self.ctl.ictr == 0 {
          match self._coldstart_prelude(state) {
            Ok(_) => {}
            Err(check) => {
              match check {
                Err(check) => {
                  return Err(Err(check));
                }
                Ok(ZKMachSignal::Yield) => {
                  return Err(Ok(ZKResumeStatus::Yield));
                }
                Ok(_) => {}
              }
            }
          }
          match self._coldstart_init(state) {
            Ok(_) => {}
            Err(check) => {
              match check {
                Err(check) => {
                  return Err(Err(check));
                }
                Ok(ZKMachSignal::Yield) => {
                  return Err(Ok(ZKResumeStatus::Yield));
                }
                Ok(_) => {}
              }
            }
          }
        }
        self.ctl.ictr += 1;
        let save_phase = self.ctl.phase;
        (save_phase, save_lock)
      }
    };
    match (save_start, save_phase) {
      (true, ZKMachPhase::Eval) => {}
      (_, ZKMachPhase::Init) => {
        self.ctl.phase = ZKMachPhase::Step;
        match self._step(state) {
          Ok(_) => {
            self.unset_lock_bit();
            return ok();
          }
          // NB: Ret.
          Err(check) => {
            match check {
              Err(check) => {
                return Err(Err(check));
              }
              Ok(ZKMachSignal::Yield) => {
                return Err(Ok(ZKResumeStatus::Yield));
              }
              Ok(_) => {}
            }
          }
        }
      }
      _ => {
        println!("DEBUG: ZKntMachLog::_resume: {:?} {:?} {:?}", save_start, save_phase, save_lock);
        return bot();
      }
    }
    // TODO: "reduce" tree code into a term (with optional "value"),
    // and "return" this term.
    // TODO: current continuation record "evaluates" into a term,
    // and "return" this term.
    // NB: the returned item may not only be a "term".
    self.ctl.phase = ZKMachPhase::Eval;
    // TODO: resume after yield directly to eval may need to reset the
    // (partial) eval state.
    // TODO: if given, resume arg is used in first eval.
    match (save_start, save_phase) {
      (_, ZKMachPhase::Init) => {
      }
      (true, ZKMachPhase::Eval) => {
        //self._eval_reset(state);
        //resume_arg._maybe_remove()
      }
      _ => {
        println!("DEBUG: ZKntMachLog::_resume: {:?} {:?} {:?}", save_start, save_phase, save_lock);
        return bot();
      }
    }
    self.set_eval_bit();
    match self._eval(state) {
      Ok(_) => {
        self.unset_eval_bit();
        self.unset_lock_bit();
        return ok();
      }
      Err(check) => {
        match check {
          Err(check) => {
            return Err(Err(check));
          }
          Ok(ZKMachSignal::Yield) => {
            return Err(Ok(ZKResumeStatus::Yield));
          }
          Ok(_) => {}
        }
      }
    }
    self.unset_eval_bit();
    // FIXME: ret convention.
    self.ctl.phase = ZKMachPhase::Backup;
    match self._backup(state) {
      Ok(_) => {
        self.unset_lock_bit();
        return ok();
      }
      // NB: Hlt.
      Err(check) => {
        match check {
          Err(check) => {
            return Err(Err(check));
          }
          Ok(ZKMachSignal::Yield) => {
            return Err(Ok(ZKResumeStatus::Yield));
          }
          Ok(_) => {}
        }
      }
    }
    println!("DEBUG: ZKntMachLog::_resume: stop:  ictr = {:?}", self.ctl.ictr);
    self.unset_lock_bit();
    Err(Ok(ZKResumeStatus::Stop))
  }

  #[inline]
  pub fn _resume_next(&mut self, state: &mut ZKntMachState) -> Result<(), Result<ZKResumeStatus, ZKMachCheck>> {
    // TODO: debug.
    if self.ctl.ictr >= 100 {
      println!("DEBUG: ZKntMachLog::_resume: break: ictr = {:?}", self.ctl.ictr);
      return Err(Ok(ZKResumeStatus::Break));
    }
    self.set_lock_bit();
    self.ctl.ictr += 1;
    self.ctl.phase = ZKMachPhase::Step;
    match self._step(state) {
      Ok(_) => {
        self.unset_lock_bit();
        return ok();
      }
      // NB: Ret.
      Err(check) => {
        match check {
          Err(check) => {
            return Err(Err(check));
          }
          Ok(ZKMachSignal::Yield) => {
            return Err(Ok(ZKResumeStatus::Yield));
          }
          Ok(_) => {}
        }
      }
    }
    // TODO: "reduce" tree code into a term (with optional "value"),
    // and "return" this term.
    // TODO: current continuation record "evaluates" into a term,
    // and "return" this term.
    // NB: the returned item may not only be a "term".
    self.ctl.phase = ZKMachPhase::Eval;
    self.set_eval_bit();
    match self._eval(state) {
      Ok(_) => {
        self.unset_eval_bit();
        self.unset_lock_bit();
        return ok();
      }
      Err(check) => {
        match check {
          Err(check) => {
            return Err(Err(check));
          }
          Ok(ZKMachSignal::Yield) => {
            return Err(Ok(ZKResumeStatus::Yield));
          }
          Ok(_) => {}
        }
      }
    }
    self.unset_eval_bit();
    // FIXME: ret convention.
    self.ctl.phase = ZKMachPhase::Backup;
    match self._backup(state) {
      Ok(_) => {
        self.unset_lock_bit();
        return ok();
      }
      // NB: Hlt.
      Err(check) => {
        match check {
          Err(check) => {
            return Err(Err(check));
          }
          Ok(ZKMachSignal::Yield) => {
            return Err(Ok(ZKResumeStatus::Yield));
          }
          Ok(_) => {}
        }
      }
    }
    println!("DEBUG: ZKntMachLog::_resume: stop:  ictr = {:?}", self.ctl.ictr);
    self.unset_lock_bit();
    Err(Ok(ZKResumeStatus::Stop))
  }

  pub fn resume(&mut self, state: &mut ZKntMachState, /*resume_arg: &mut LReg<_>*/) -> Result<ZKResumeStatus, ZKMachCheck> {
    if self.cur.halt._is_fill() {
      println!("DEBUG: ZKntMachLog::resume: halt:  ictr = {:?}", self.ctl.ictr);
      return Ok(ZKResumeStatus::Halt);
    }
    match self._resume_start(state) {
      Ok(_) => {}
      Err(yield_) => {
        println!("DEBUG: ZKntMachLog::resume: yield: ictr = {:?}", self.ctl.ictr);
        return yield_;
      }
    }
    loop {
      match self._resume_next(state) {
        Ok(_) => {}
        Err(yield_) => {
          println!("DEBUG: ZKntMachLog::resume: yield: ictr = {:?}", self.ctl.ictr);
          return yield_;
        }
      }
    }
  }

  pub fn resume_inplace(&mut self, state: &mut ZKntMachState, /*resume_arg: &mut LReg<_>*/) -> Result<ZKResumeStatus, ZKMachCheck> {
    self.resume(state, )
  }
}
