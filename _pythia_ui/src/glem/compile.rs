//! glem's part of the compiled form ([`crate::lem::compile`]): which of the
//! program's relations are widget kinds and surfaces, and which `state`
//! relations are `keep`, as one more member, after lem's:
//!
//! ```text
//! "glem":{"kinds":[[19,"surface"],[20,"widget"]],"keep":[27]}
//! ```
//!
//! `glem.js`, glem's runtime in JavaScript, beside this file, reads it as
//! [`read`] does.

use super::lower::Kind;
use super::Program;
use crate::json::{self, object, Value as Json};
use crate::lem::check::DeclKind;
use crate::lem::compile::{compile_with, read_form};
use crate::lem::value::Ty;

/// The program as its compiled form: lem's, then glem's member.
pub fn compile(prog: &Program) -> String {
  let kinds = prog
    .kinds
    .iter()
    .map(|(rel, kind)| Json::Array(vec![Json::from(*rel), Json::from(kind.word())]));
  let keep = prog.keep.iter().map(|rel| Json::from(*rel));
  let glem = object([
    ("kinds", Json::Array(kinds.collect())),
    ("keep", Json::Array(keep.collect())),
  ]);
  compile_with(&prog.lem, &[("glem", glem)])
}

/// Reads a glem program's compiled form back ([`compile`]), and checks
/// glem's member against lem's: a widget kind's relation has an `id`, its
/// own fields, a `parent` and an `order`, of types Id, Id and Any; a
/// surface's has an `id` first; and a `keep` relation is a `state` one.
pub fn read(text: &str) -> Result<Program, String> {
  let form = json::parse(text).map_err(|e| format!("the compiled form isn't JSON: {e}"))?;
  let lem = read_form(&form)?;
  let Some(glem) = form.get("glem") else {
    return Err("this isn't the compiled form of a glem program: it has no `glem`".to_string());
  };
  let id = Ty::Named("Id".into());
  let mut kinds: Vec<(usize, Kind)> = Vec::new();
  for kind in items(glem, "kinds")? {
    let (rel, word) = match kind {
      Json::Array(parts) => match parts.as_slice() {
        [rel, Json::String(word)] => (rel, word.as_str()),
        _ => return Err("a kind is [relation, \"widget\" or \"surface\"]".to_string()),
      },
      _ => return Err("a kind is [relation, \"widget\" or \"surface\"]".to_string()),
    };
    let rel = relation(rel, &lem.rels.len())?;
    let r = &lem.rels[rel];
    let (fields, n) = (&r.fields, r.fields.len());
    let kind = match word {
      "widget"
        if n >= 3
          && fields[0].ty == id
          && fields[n - 2].ty == id
          && fields[n - 1].ty == Ty::Any =>
      {
        Kind::Widget
      }
      "surface" if n >= 1 && fields[0].ty == id => Kind::Surface,
      _ => return Err(format!("{} isn't a {word}'s relation", r.name)),
    };
    if r.kind != DeclKind::Rel || kinds.iter().any(|(other, _)| *other == rel) {
      return Err(format!("{} isn't a {word}'s relation", r.name));
    }
    kinds.push((rel, kind));
  }
  kinds.sort_by_key(|(rel, _)| *rel);
  let mut keep = Vec::new();
  for rel in items(glem, "keep")? {
    let rel = relation(rel, &lem.rels.len())?;
    if lem.rels[rel].kind != DeclKind::State {
      let name = &lem.rels[rel].name;
      return Err(format!("{name} is `keep`, but isn't a `state` relation"));
    }
    keep.push(rel);
  }
  Ok(Program { lem, kinds, keep })
}

/// The items of the array that's the member `name` of glem's member.
fn items<'j>(glem: &'j Json, name: &str) -> Result<&'j [Json], String> {
  match glem.get(name) {
    Some(Json::Array(items)) => Ok(items),
    _ => Err(format!("glem's `{name}` is an array")),
  }
}

/// A relation, by its number below `len`.
fn relation(value: &Json, len: &usize) -> Result<usize, String> {
  match value.as_u64().and_then(|n| usize::try_from(n).ok()) {
    Some(n) if n < *len => Ok(n),
    _ => Err(format!("a relation is a number below {len}")),
  }
}
