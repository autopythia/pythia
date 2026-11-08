//! lem's values and types.

use super::print::quote;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

/// A type: a base type, or a named one: an enum, a type made with `type`,
/// or `Id` or `Msg`, whose constructors the `id` and `msg` sentences add.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
  Bool,
  Int,
  Num,
  Text,
  Time,
  /// Any value fits: values of different types order by type, then by
  /// value, as a sort key needs.
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
/// and of the types made from it, as `JobId` is.
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

/// As lem writes it: text quoted, and a Num always with a fraction.
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
