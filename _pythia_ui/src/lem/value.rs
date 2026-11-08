//! lem's values and types.

use super::print::quote;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

/// The largest whole number, 2^53 − 1, which is JavaScript's largest safe
/// integer, so that every Int and Time is exact in every runtime. The
/// smallest is its negation.
pub const MAX_INT: i64 = (1 << 53) - 1;

/// Whether a whole number is within Int's range, ±(2^53 − 1).
pub fn int_fits(n: i64) -> bool {
  (-MAX_INT..=MAX_INT).contains(&n)
}

/// A type: a base type, or a named one: an enum, a type made with `type`,
/// or `Id` or `Msg`, whose constructors the `id` and `msg` sentences add.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
  Bool,
  Int,
  Num,
  Text,
  Time,
  /// Any value but a Num fits: values of different types order by type,
  /// then by value, as a sort key needs. Without Nums, mixed values order
  /// alike in every runtime, whose numbers needn't say Int or Num.
  Any,
  Named(Rc<str>),
}

impl Ty {
  /// The type after "a" or "an", as error messages name it.
  pub fn a(&self) -> String {
    match self {
      Ty::Any => "any value".to_string(),
      ty if ty.to_string().starts_with(['A', 'E', 'I', 'O', 'U']) => format!("an {ty}"),
      ty => format!("a {ty}"),
    }
  }
}

impl Display for Ty {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    f.write_str(match self {
      Ty::Bool => "Bool",
      Ty::Int => "Int",
      Ty::Num => "Num",
      Ty::Text => "Text",
      Ty::Time => "Time",
      Ty::Any => "Any",
      Ty::Named(name) => name,
    })
  }
}

/// A value. `Int` holds Int and Time values, and `Text` the values of Text
/// and of the types made from it, as `JobId` is. An Int is within
/// ±(2^53 − 1) ([`int_fits`]), and a Num is finite and never −0.0
/// ([`Value::num`]), so that values compute and print alike in every
/// runtime.
#[derive(Clone, Debug)]
pub enum Value {
  Bool(bool),
  Int(i64),
  /// Ordered by `total_cmp`, so that every value sorts.
  Num(f64),
  Text(Rc<str>),
  /// A constructor and its arguments: an enum's, an id's or a message's.
  Ctor(Rc<str>, Rc<[Value]>),
}

impl Value {
  /// A Num, or None for a number that isn't finite; −0.0 becomes 0.0.
  pub fn num(x: f64) -> Option<Value> {
    if !x.is_finite() {
      return None;
    }
    Some(Value::Num(if x == 0.0 { 0.0 } else { x }))
  }

  /// The value as `text(x)` gives it: a Text as it is, and any other value
  /// as lem writes it.
  pub fn text(&self) -> Rc<str> {
    match self {
      Value::Text(text) => text.clone(),
      value => value.to_string().into(),
    }
  }

  fn rank(&self) -> u8 {
    match self {
      Value::Bool(_) => 0,
      Value::Int(_) => 1,
      Value::Num(_) => 2,
      Value::Text(_) => 3,
      Value::Ctor(..) => 4,
    }
  }
}

impl Ord for Value {
  fn cmp(&self, other: &Value) -> Ordering {
    match (self, other) {
      (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
      (Value::Int(a), Value::Int(b)) => a.cmp(b),
      (Value::Num(a), Value::Num(b)) => a.total_cmp(b),
      (Value::Text(a), Value::Text(b)) => a.cmp(b),
      (Value::Ctor(a, x), Value::Ctor(b, y)) => a.cmp(b).then_with(|| x.cmp(y)),
      _ => self.rank().cmp(&other.rank()),
    }
  }
}

impl PartialOrd for Value {
  fn partial_cmp(&self, other: &Value) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl PartialEq for Value {
  fn eq(&self, other: &Value) -> bool {
    self.cmp(other) == Ordering::Equal
  }
}

impl Eq for Value {}

/// As lem writes it: text quoted, and a Num as Rust's `{:?}` writes it, with
/// a decimal point, and in exponent form below 1e-4 and from 1e16 (`1e-5`,
/// `1.5e16`). Every runtime prints numbers so.
impl Display for Value {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    match self {
      Value::Bool(b) => f.write_str(if *b { "True" } else { "False" }),
      Value::Int(n) => write!(f, "{n}"),
      Value::Num(n) => write!(f, "{n:?}"),
      Value::Text(s) => {
        let mut buf = String::new();
        quote(&mut buf, s);
        f.write_str(&buf)
      }
      Value::Ctor(name, args) => {
        f.write_str(name)?;
        if !args.is_empty() {
          f.write_str("(")?;
          for (i, arg) in args.iter().enumerate() {
            if i > 0 {
              f.write_str(", ")?;
            }
            write!(f, "{arg}")?;
          }
          f.write_str(")")?;
        }
        Ok(())
      }
    }
  }
}
