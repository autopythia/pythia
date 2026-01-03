use crate::algo::{Rc};
use crate::zinterp::parse::{
  Term as ParseTerm,
  Sent as ParseSent,
  Mod as ParseMod,
  SpanIdent as ParseIdent,
  //SpanLit as ParseLit,
};
pub use crate::zinterp::parse::{
  Span,
};

pub type TreeIdent = ParseIdent;

pub type TreeTermRef_ = Rc<TreeTerm_>;
pub type TreeSentRef_ = Rc<TreeSent_>;
pub type TreeModRef_ = Rc<TreeMod_>;

pub type TreeTerm_ = ParseTerm;
pub type TreeSent_ = ParseSent;
pub type TreeMod_ = ParseMod;

//pub type TreeRef = Rc<Tree>;

#[derive(Clone, Debug)]
pub enum Tree {
  Term(TreeTermRef_),
  Sent(TreeSentRef_),
  Block(Vec<TreeSentRef_>),
  Mod(TreeModRef_),
}

impl From<TreeMod_> for Tree {
  fn from(mod_: TreeMod_) -> Tree {
    Tree::Mod(mod_.into())
  }
}

pub type TreeLogRef = Option<Rc<TreeLog>>;

pub struct TreeLog {
  pub back: TreeLogRef,
  pub cur:  Tree,
  pub prev: Vec<Tree>,
  pub next: Vec<Tree>,
}
