//! glem: lem's GUI layer. It adds widgets to lem ([`crate::lem`]):
//!
//! - `widget` and `surface` declarations. Besides its own fields, a widget
//!   has an `id`, a `parent` and an `order`; a surface, such as a window,
//!   has an `id`, and is the root of a tree.
//! - Blocks, `Head:`, which give the widgets inside them their parent and
//!   order.
//! - The widget tree, checked after each tick: an id names one widget,
//!   every parent is a widget, and the parents form no cycle. Siblings sort
//!   by `order`, then by id.
//! - State that names a widget goes when the widget does, unless its
//!   relation says `keep`.
//! - Its prelude, `prelude.lem`: `Parent`, `Within`, keys, and where the
//!   pointer and the keyboard are.
//!
//! glem rewrites a program as plain lem ([`lower`]), which lem checks and
//! runs, and wraps lem's runtime ([`Runtime`]). Which widgets exist, and how
//! they look, is up to a design language above glem, which also gives the
//! outline its notes ([`Notes`]). For the web, a program compiles to data
//! with glem's part ([`compile`]), which `glem.js`, glem's runtime in
//! JavaScript, runs.

pub mod compile;
#[cfg(test)]
mod compile_test;
pub mod lower;
#[cfg(test)]
mod tests;

use crate::lem::check::{self, DeclKind, LoadError};
use crate::lem::eval::{self, Effect, Script, Tuple, Why};
use crate::lem::value::{Ty, Value};
use lower::Kind;
use std::collections::{BTreeMap, BTreeSet};

/// glem's prelude, which every glem program loads after lem's.
pub const PRELUDE: (&str, &str) = ("src/glem/prelude.lem", include_str!("prelude.lem"));

/// A checked glem program: lem's, and where its widgets are.
#[derive(Debug)]
pub struct Program {
  pub lem: check::Program,
  /// The widget kinds and surfaces, by relation, in the program's order.
  kinds: Vec<(usize, Kind)>,
  /// The `state` relations whose facts stay when their widget goes.
  keep: Vec<usize>,
}

/// Loads files, given as (name, text) pairs, after glem's prelude: parses
/// them, rewrites them as plain lem, and has lem check them.
pub fn load(files: &[(&str, &str)]) -> Result<Program, LoadError> {
  let mut all = vec![PRELUDE];
  all.extend_from_slice(files);
  let mods = check::parse_all(&all)?;
  let lowered = lower::lower(&all, &mods)?;
  let lem = check::check(&all, &lowered.mods)?;
  let rel = |name: &str| lem.rel(name).expect("a declaration that lem checked");
  let mut kinds: Vec<(usize, Kind)> = lowered.kinds.iter().map(|(n, k)| (rel(n), *k)).collect();
  kinds.sort_by_key(|(rel, _)| *rel);
  let keep = lowered.keep.iter().map(|name| rel(name)).collect();
  Ok(Program { lem, kinds, keep })
}

/// Notes on a widget, for the outline, from a layer above glem: given lem's
/// runtime and the widget's id.
pub type Notes<'a> = &'a dyn Fn(&eval::Runtime, &Value) -> Vec<String>;

/// No notes, for a program without a layer above glem.
pub fn no_notes(_: &eval::Runtime, _: &Value) -> Vec<String> {
  Vec::new()
}

/// A widget or a surface, in the tree's order.
#[derive(Clone, Debug)]
pub struct Node {
  pub depth: usize,
  pub rel: usize,
  pub tuple: Tuple,
  pub why: Why,
}

/// A glem program's runtime: lem's, and the widget tree.
pub struct Runtime {
  pub lem: eval::Runtime,
  kinds: Vec<(usize, Kind)>,
  keep: Vec<usize>,
}

impl Runtime {
  pub fn new(prog: Program) -> Runtime {
    Runtime {
      lem: eval::Runtime::new(prog.lem),
      kinds: prog.kinds,
      keep: prog.keep,
    }
  }

  /// One tick: lem's ([`eval::Runtime::tick`]), then the tree's checks,
  /// then dropping the state whose widget is gone. Gives the effects that
  /// the `on` blocks sent.
  pub fn tick(&mut self, msg: Option<&Value>) -> Result<Vec<Effect>, String> {
    let effects = self.lem.tick(msg)?;
    self.tree()?;
    self.drop_orphans();
    Ok(effects)
  }

  /// Runs a tick written as lines ([`eval::Runtime::read_script`]), and
  /// gives the outline (or `(no widgets)`), then the facts asked for and the
  /// effects sent ([`eval::Runtime::show`]).
  pub fn script(&mut self, lines: &str, notes: Notes) -> Result<String, String> {
    let script = self.lem.read_script(lines)?;
    self.run(&script, notes)
  }

  /// Runs the tick that a script describes, as lines or JSON give it, and
  /// gives what [`Runtime::script`] gives.
  pub fn run(&mut self, script: &Script, notes: Notes) -> Result<String, String> {
    let effects = self.tick(script.msg.as_ref())?;
    let mut out = self.outline(notes)?;
    if out.is_empty() {
      out.push_str("(no widgets)\n");
    }
    out.push_str(&self.lem.show(script, &effects));
    Ok(out.trim_end().to_string())
  }

  fn is_widget(&self, rel: usize) -> bool {
    self.kinds.contains(&(rel, Kind::Widget))
  }

  /// The widget tree: each surface, and each widget under its parent, by
  /// `order` and then id. An id names one widget, every parent is a widget,
  /// and every widget is under a surface.
  pub fn tree(&self) -> Result<Vec<Node>, String> {
    let prog = &self.lem.prog;
    let mut nodes: BTreeMap<&Value, (usize, &Tuple, Why)> = BTreeMap::new();
    let mut kids: BTreeMap<&Value, Vec<(&Value, &Value)>> = BTreeMap::new();
    let mut roots = Vec::new();
    for &(rel, kind) in &self.kinds {
      for (tuple, why) in self.lem.rel_facts(rel) {
        let id = &tuple[0];
        if let Some((other, _, first)) = nodes.get(id) {
          let (a, b) = (&prog.rels[*other].name, &prog.rels[rel].name);
          let (at, other) = (self.lem.why(*first), self.lem.why(why));
          return Err(format!(
            "the id {id} names two widgets: {a} ({at}) and {b} ({other})"
          ));
        }
        nodes.insert(id, (rel, tuple, why));
        match kind {
          // A widget's last fields are its `parent` and `order` (lower.rs).
          Kind::Widget => {
            let n = tuple.len();
            kids
              .entry(&tuple[n - 2])
              .or_default()
              .push((&tuple[n - 1], id));
          }
          Kind::Surface => roots.push(id),
        }
      }
    }
    for (parent, under) in &kids {
      if !nodes.contains_key(parent) {
        let (_, child) = under[0];
        let why = self.lem.why(nodes[child].2);
        return Err(format!(
          "{child}'s parent, {parent}, isn't a widget ({why})"
        ));
      }
    }
    roots.sort();
    let mut out = Vec::new();
    let mut stack: Vec<(&Value, usize)> = roots.iter().rev().map(|id| (*id, 0)).collect();
    while let Some((id, depth)) = stack.pop() {
      let (rel, tuple, why) = nodes[id];
      out.push(Node {
        depth,
        rel,
        tuple: tuple.clone(),
        why,
      });
      if let Some(under) = kids.get(id) {
        let mut under = under.clone();
        under.sort();
        stack.extend(under.iter().rev().map(|(_, kid)| (*kid, depth + 1)));
      }
    }
    if out.len() < nodes.len() {
      let placed: BTreeSet<&Value> = out.iter().map(|n| &n.tuple[0]).collect();
      let lost: Vec<String> = nodes
        .keys()
        .filter(|id| !placed.contains(*id))
        .map(|id| id.to_string())
        .collect();
      return Err(format!(
        "these widgets' parents form a cycle: {}",
        lost.join(", ")
      ));
    }
    Ok(out)
  }

  /// The widget tree as text: a widget a line, indented under its parent,
  /// with its own fields by name, then its notes, if any.
  pub fn outline(&self, notes: Notes) -> Result<String, String> {
    let mut out = String::new();
    for node in self.tree()? {
      let r = &self.lem.prog.rels[node.rel];
      let n = r.fields.len();
      let own = if self.is_widget(node.rel) {
        1..n - 2
      } else {
        1..n
      };
      out.push_str(&"  ".repeat(node.depth));
      out.push_str(&format!("{}({}", r.name, node.tuple[0]));
      for i in own {
        let name = r.fields[i].name.as_deref().unwrap_or("_");
        out.push_str(&format!(", {name}={}", node.tuple[i]));
      }
      out.push(')');
      let notes = notes(&self.lem, &node.tuple[0]);
      if !notes.is_empty() {
        out.push_str(&format!(" -- {}", notes.join("; ")));
      }
      out.push('\n');
    }
    Ok(out)
  }

  /// Drops the `state` facts that name a widget that's gone, unless their
  /// relation says `keep`.
  fn drop_orphans(&mut self) {
    let mut live = BTreeSet::new();
    for &(rel, _) in &self.kinds {
      live.extend(self.lem.rel_facts(rel).map(|(tuple, _)| tuple[0].clone()));
    }
    let id = Ty::Named("Id".into());
    let rels = &self.lem.prog.rels;
    let doomed: Vec<(usize, Vec<usize>)> = (0..rels.len())
      .filter(|&rel| rels[rel].kind == DeclKind::State && !self.keep.contains(&rel))
      .map(|rel| {
        let fields = &rels[rel].fields;
        (
          rel,
          (0..fields.len()).filter(|&i| fields[i].ty == id).collect(),
        )
      })
      .collect();
    for (rel, columns) in doomed {
      if !columns.is_empty() {
        let gone = |tuple: &Tuple| columns.iter().any(|&i| !live.contains(&tuple[i]));
        self.lem.drop_facts(rel, gone);
      }
    }
  }
}
