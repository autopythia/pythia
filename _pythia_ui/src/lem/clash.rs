//! What a program's keys can't promise: rules that may give one key two
//! facts, which would fail a tick, found before any tick runs ([`clashes`]).
//!
//! A relation with a key is checked, and one that's `one`, as a key of no
//! columns; and, for the layer above, kinds of relations whose facts share
//! their first column, an id that names one fact of them all, as glem's
//! widgets do. For a rule, and for each pair of rules, of one such relation,
//! the check takes two of their solutions whose facts agree on the key, and
//! finds what else must be equal in them:
//!
//! - the key's variables, and those of a constructor's arguments, since a
//!   constructor gives equal values for equal arguments alone;
//! - through two atoms of a relation with a key, or `one`, that agree on the
//!   key, the rest of their columns, since one fact has that key;
//! - in two copies of one rule, each binding that `=` gives from what's
//!   equal;
//! - and the value that a negation picks as the least, or the greatest, of a
//!   column: `First(f) :- Chunk(f, _), not Below(f)`, where `Below(f) :-
//!   Chunk(f, _), Chunk(g, _), g < f`, since two values can't both be least.
//!
//! The two facts can't clash if they must then agree in full; if one rule
//! has a negation that the other's atom would refute; if two constants that
//! differ must be equal, so that the two solutions can't both hold, as
//! `Opened(True)` and `Opened(False)` of a relation that's `one`; or if a
//! test of either rule must fail, as `title /= ""` where the title must be
//! "". Else the check warns, at the rule, or at the later of the two.

use super::check::{Atom, Expr, Lit, Pat, Program, Rule, Test};
use super::parse::Cmp;
use super::value::Value;
use std::collections::BTreeMap;

/// The warnings for `prog`'s rules, each as lem shows an error
/// ([`super::render`]): those of relations with a key, or `one`; and of
/// the relations `kinds`, whose facts share an id, their first column, of
/// which one fact of them all may have each. The rules `skip` are left out.
pub fn clashes(prog: &Program, kinds: &[usize], skip: &[usize]) -> Vec<String> {
  let checked = |rel: usize| keyed(prog, kinds, rel) || kinds.contains(&rel);
  let rules: Vec<usize> = (0..prog.rules.len())
    .filter(|r| !skip.contains(r) && checked(prog.rules[*r].head))
    .collect();
  let mut found = Vec::new();
  for (i, &r1) in rules.iter().enumerate() {
    for &r2 in &rules[i..] {
      let (h1, h2) = (prog.rules[r1].head, prog.rules[r2].head);
      let shared = h1 == h2 || (kinds.contains(&h1) && kinds.contains(&h2));
      if shared {
        if let Some(text) = clash(prog, kinds, r1, r2) {
          let at = &prog.rules[r2].at;
          let (name, src) = &prog.files[at.file];
          found.push(super::render(
            name,
            src,
            &at.span,
            &format!("warning: {text}"),
          ));
        }
      }
    }
  }
  found
}

/// Whether facts of relation `rel` that agree on some columns, its key,
/// must agree on the rest: it has a key, or one fact at most; or it's one of
/// `kinds`, whose key is its first column.
fn keyed(prog: &Program, kinds: &[usize], rel: usize) -> bool {
  prog.rels[rel].keyed() || kinds.contains(&rel)
}

/// A relation's key's columns: its first, for one of `kinds`.
fn key(prog: &Program, kinds: &[usize], rel: usize) -> Vec<usize> {
  match kinds.contains(&rel) {
    true => vec![0],
    false => prog.rels[rel].key.clone(),
  }
}

/// Why two solutions of rules `r1` and `r2`, which may be one rule, may give
/// two facts with one key; or None if they can't.
fn clash(prog: &Program, kinds: &[usize], r1: usize, r2: usize) -> Option<String> {
  let (a, b) = (&prog.rules[r1], &prog.rules[r2]);
  let key = key(prog, kinds, a.head);
  if !key.iter().all(|&c| may_equal(&a.args[c], &b.args[c])) {
    return None;
  }
  let mut same = Same::default();
  for &c in &key {
    same.join_values(&a.args[c], 0, &b.args[c], 1);
  }
  let rules = [a, b];
  loop {
    let mut changed = false;
    for (x, y) in [(0, 1), (0, 0), (1, 1)] {
      changed |= same.through_keys(prog, kinds, rules[x], x as u8, rules[y], y as u8);
    }
    if r1 == r2 {
      changed |= same.through_lets(a);
    }
    changed |= same.through_extremes(prog, a, b);
    if !changed {
      break;
    }
  }
  if same.impossible || same.fails(a, 0) || same.fails(b, 1) {
    return None;
  }
  if a.head == b.head
    && (0..a.args.len()).all(|c| key.contains(&c) || same.equal(&a.args[c], 0, &b.args[c], 1))
  {
    return None;
  }
  if refutes(&mut same, a, 0, b, 1) || refutes(&mut same, b, 1, a, 0) {
    return None;
  }
  let rel = &prog.rels[a.head];
  let other = match r1 == r2 {
    true => "this rule".to_string(),
    false => format!("this rule and the one at {}", a.place),
  };
  if a.head != b.head {
    let kind = &prog.rels[b.head].name;
    return Some(format!(
      "{other} may give one id two widgets, of kinds {} and {kind}",
      rel.name
    ));
  }
  let differ: Vec<String> = (0..a.args.len())
    .filter(|&c| !key.contains(&c) && !same.equal(&a.args[c], 0, &b.args[c], 1))
    .map(|c| match &rel.fields[c].name {
      Some(name) => format!("`{name}`"),
      None => format!("column {}", c + 1),
    })
    .collect();
  let differ = differ.join(", ");
  Some(match (kinds.contains(&a.head), rel.one) {
    (true, _) => format!(
      "{other} may give one id two {} widgets, which differ in {differ}",
      rel.name
    ),
    (false, true) => format!(
      "{other} may give {} two facts, where `one` allows one",
      rel.name
    ),
    (false, false) => format!(
      "{other} may give {} two facts with one key, which differ in {differ}",
      rel.name
    ),
  })
}

/// Whether two head values may be equal: not if their constants, or their
/// constructors, differ.
fn may_equal(a: &Expr, b: &Expr) -> bool {
  match (a, b) {
    (Expr::Const(x), Expr::Const(y)) => x == y,
    (Expr::Ctor(n, xs), Expr::Ctor(m, ys)) => {
      n == m && xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| may_equal(x, y))
    }
    (Expr::Ctor(n, xs), Expr::Const(v)) | (Expr::Const(v), Expr::Ctor(n, xs)) => match v {
      Value::Ctor(m, vs) => {
        n == m
          && xs.len() == vs.len()
          && (xs.iter().zip(vs.iter())).all(|(x, v)| may_equal(x, &Expr::Const(v.clone())))
      }
      _ => false,
    },
    _ => true,
  }
}

/// Whether a negation of rule `n`, of copy `cn`, refutes an atom of rule `p`,
/// of copy `cp`: every fact that the atom matches, it matches too.
fn refutes(same: &mut Same, n: &Rule, cn: u8, p: &Rule, cp: u8) -> bool {
  let atoms: Vec<&Atom> = atoms(p).collect();
  n.body.iter().any(|lit| {
    let Lit::Not(neg) = lit else {
      return false;
    };
    (atoms.iter()).any(|atom| {
      atom.rel == neg.rel
        && (neg.pats.iter().zip(&atom.pats))
          .all(|(q, r)| matches!(q, Pat::Any) || same.equal_pats(q, cn, r, cp))
    })
  })
}

fn atoms(rule: &Rule) -> impl Iterator<Item = &Atom> {
  rule.body.iter().filter_map(|lit| match lit {
    Lit::Atom(atom) => Some(atom),
    _ => None,
  })
}

/// A pattern as the value that it matches, or None if it has `_`.
fn value_of(pat: &Pat) -> Option<Expr> {
  match pat {
    Pat::Any => None,
    Pat::Var(v) => Some(Expr::Var(*v)),
    Pat::Const(c) => Some(Expr::Const(c.clone())),
    Pat::Ctor(name, pats) => Some(Expr::Ctor(
      name.clone(),
      pats.iter().map(value_of).collect::<Option<_>>()?,
    )),
    Pat::Expr(e) => Some(e.clone()),
  }
}

/// A variable of one of the two copies, or a constant.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Term {
  Var(u8, usize),
  Const(Value),
}

/// What's equal in two solutions, copies 0 and 1: classes of terms; and
/// whether two constants that differ are in one class, so that the two
/// can't both hold.
#[derive(Default)]
struct Same {
  ids: BTreeMap<Term, usize>,
  up: Vec<usize>,
  /// For each class, by its root, the constant in it, if any.
  value: BTreeMap<usize, Value>,
  impossible: bool,
}

impl Same {
  fn id(&mut self, term: Term) -> usize {
    let next = self.up.len();
    let constant = match &term {
      Term::Const(v) => Some(v.clone()),
      Term::Var(..) => None,
    };
    let id = *self.ids.entry(term).or_insert(next);
    if id == next {
      self.up.push(id);
      if let Some(v) = constant {
        self.value.insert(id, v);
      }
    }
    id
  }

  fn root(&mut self, mut id: usize) -> usize {
    while self.up[id] != id {
      self.up[id] = self.up[self.up[id]];
      id = self.up[id];
    }
    id
  }

  fn join(&mut self, a: Term, b: Term) -> bool {
    let (a, b) = (self.id(a), self.id(b));
    let (a, b) = (self.root(a), self.root(b));
    if a == b {
      return false;
    }
    self.up[a] = b;
    if let Some(x) = self.value.remove(&a) {
      match self.value.get(&b) {
        Some(y) if *y != x => self.impossible = true,
        Some(_) => {}
        None => _ = self.value.insert(b, x),
      }
    }
    true
  }

  fn same(&mut self, a: Term, b: Term) -> bool {
    let (a, b) = (self.id(a), self.id(b));
    self.root(a) == self.root(b)
  }

  /// Whether two values, of copies `ca` and `cb`, must be equal.
  fn equal(&mut self, a: &Expr, ca: u8, b: &Expr, cb: u8) -> bool {
    match (a, b) {
      (Expr::Var(x), Expr::Var(y)) => self.same(Term::Var(ca, *x), Term::Var(cb, *y)),
      (Expr::Var(x), Expr::Const(v)) => self.same(Term::Var(ca, *x), Term::Const(v.clone())),
      (Expr::Const(v), Expr::Var(y)) => self.same(Term::Const(v.clone()), Term::Var(cb, *y)),
      (Expr::Const(x), Expr::Const(y)) => x == y,
      (Expr::Ctor(n, xs), Expr::Ctor(m, ys)) => {
        n == m && xs.len() == ys.len() && (xs.iter().zip(ys)).all(|(x, y)| self.equal(x, ca, y, cb))
      }
      (Expr::Neg(x), Expr::Neg(y)) | (Expr::ToNum(x), Expr::ToNum(y)) => self.equal(x, ca, y, cb),
      (Expr::Arith(o, t, x1, x2), Expr::Arith(p, u, y1, y2)) => {
        o == p && t == u && self.equal(x1, ca, y1, cb) && self.equal(x2, ca, y2, cb)
      }
      (Expr::Text(t, x), Expr::Text(u, y)) => t == u && self.equal(x, ca, y, cb),
      _ => false,
    }
  }

  /// The constant that a value of copy `c` must be, if any.
  fn constant(&mut self, e: &Expr, c: u8) -> Option<Value> {
    match e {
      Expr::Const(v) => Some(v.clone()),
      Expr::Var(v) => {
        let id = self.id(Term::Var(c, *v));
        let root = self.root(id);
        self.value.get(&root).cloned()
      }
      _ => None,
    }
  }

  /// Whether a test of `rule`, of copy `c`, must fail: `x /= y` of values
  /// that must be equal, or `x = y` of constants that differ.
  fn fails(&mut self, rule: &Rule, c: u8) -> bool {
    rule.body.iter().any(|lit| match lit {
      Lit::Test(Test::Ne(x, y)) | Lit::NotTest(Test::Eq(x, y)) => self.equal(x, c, y, c),
      Lit::Test(Test::Eq(x, y)) | Lit::NotTest(Test::Ne(x, y)) => {
        matches!((self.constant(x, c), self.constant(y, c)), (Some(p), Some(q)) if p != q)
      }
      _ => false,
    })
  }

  fn equal_pats(&mut self, p: &Pat, cp: u8, q: &Pat, cq: u8) -> bool {
    match (value_of(p), value_of(q)) {
      (Some(a), Some(b)) => self.equal(&a, cp, &b, cq),
      _ => false,
    }
  }

  /// Joins what two equal values, of copies `ca` and `cb`, make equal: the
  /// variables that they are, and a constructor's arguments. Whether that
  /// joined anything.
  fn join_values(&mut self, a: &Expr, ca: u8, b: &Expr, cb: u8) -> bool {
    match (a, b) {
      (Expr::Var(x), Expr::Var(y)) => self.join(Term::Var(ca, *x), Term::Var(cb, *y)),
      (Expr::Var(x), Expr::Const(v)) => self.join(Term::Var(ca, *x), Term::Const(v.clone())),
      (Expr::Const(v), Expr::Var(y)) => self.join(Term::Const(v.clone()), Term::Var(cb, *y)),
      (Expr::Const(x), Expr::Const(y)) => self.join(Term::Const(x.clone()), Term::Const(y.clone())),
      (Expr::Ctor(n, xs), Expr::Ctor(m, ys)) if n == m && xs.len() == ys.len() => {
        let mut changed = false;
        for (x, y) in xs.iter().zip(ys) {
          changed |= self.join_values(x, ca, y, cb);
        }
        changed
      }
      (Expr::Ctor(n, xs), Expr::Const(Value::Ctor(m, vs))) if n == m && xs.len() == vs.len() => {
        let mut changed = false;
        for (x, v) in xs.iter().zip(vs.iter()) {
          changed |= self.join_values(x, ca, &Expr::Const(v.clone()), cb);
        }
        changed
      }
      (Expr::Const(Value::Ctor(..)), Expr::Ctor(..)) => self.join_values(b, cb, a, ca),
      (Expr::Neg(x), Expr::Neg(y)) | (Expr::ToNum(x), Expr::ToNum(y)) => {
        self.join_values(x, ca, y, cb)
      }
      _ => false,
    }
  }

  fn join_pats(&mut self, p: &Pat, cp: u8, q: &Pat, cq: u8) -> bool {
    match (p, q) {
      (Pat::Ctor(n, ps), Pat::Ctor(m, qs)) if n == m && ps.len() == qs.len() => {
        let mut changed = false;
        for (p, q) in ps.iter().zip(qs) {
          changed |= self.join_pats(p, cp, q, cq);
        }
        changed
      }
      _ => match (value_of(p), value_of(q)) {
        (Some(a), Some(b)) => self.join_values(&a, cp, &b, cq),
        _ => false,
      },
    }
  }

  /// Joins, through each two atoms of rules `a`, of copy `ca`, and `b`, of
  /// copy `cb`, of a relation with a key, or `one`, that agree on the key,
  /// the rest of their columns. Whether that joined anything.
  fn through_keys(
    &mut self,
    prog: &Program,
    kinds: &[usize],
    a: &Rule,
    ca: u8,
    b: &Rule,
    cb: u8,
  ) -> bool {
    let mut changed = false;
    for (i, x) in atoms(a).enumerate() {
      for (j, y) in atoms(b).enumerate() {
        if (ca == cb && i == j) || x.rel != y.rel || !keyed(prog, kinds, x.rel) {
          continue;
        }
        let key = key(prog, kinds, x.rel);
        if key
          .iter()
          .all(|&c| self.equal_pats(&x.pats[c], ca, &y.pats[c], cb))
        {
          for (p, q) in x.pats.iter().zip(&y.pats) {
            changed |= self.join_pats(p, ca, q, cb);
          }
        }
      }
    }
    changed
  }

  /// Joins, in two copies of `rule`, each variable that `=` binds from
  /// values that must be equal. Whether that joined anything.
  fn through_lets(&mut self, rule: &Rule) -> bool {
    let mut changed = false;
    for lit in &rule.body {
      if let Lit::Let(v, e) = lit {
        if self.equal(e, 0, e, 1) {
          changed |= self.join(Term::Var(0, *v), Term::Var(1, *v));
        }
      }
    }
    changed
  }

  /// Joins the values that rules `a`, of copy 0, and `b`, of copy 1, each
  /// pick as the least, or the greatest, of a column ([`extreme`]): two
  /// such can't differ. Whether that joined anything.
  fn through_extremes(&mut self, prog: &Program, a: &Rule, b: &Rule) -> bool {
    let mut changed = false;
    for (x, col) in extremes(prog, a) {
      for (y, other) in extremes(prog, b) {
        if col == other {
          changed |= self.join(Term::Var(0, x), Term::Var(1, y));
        }
      }
    }
    changed
  }
}

/// The variables that `rule` picks as the least, or the greatest, of a
/// column: each with the column, (relation, column, negated relation). The
/// rule has an atom of the relation with the variable in that column, and
/// `not B(v)`, where every rule of `B` holds of a value of the column that
/// another of its values is less than, or greater than ([`witness`]).
fn extremes(prog: &Program, rule: &Rule) -> Vec<(usize, (usize, usize, usize))> {
  let mut found = Vec::new();
  for lit in &rule.body {
    let Lit::Not(neg) = lit else {
      continue;
    };
    let [Pat::Var(v)] = neg.pats.as_slice() else {
      continue;
    };
    for atom in atoms(rule) {
      for (c, pat) in atom.pats.iter().enumerate() {
        if matches!(pat, Pat::Var(w) if w == v) && witness(prog, neg.rel, atom.rel, c) {
          found.push((*v, (atom.rel, c, neg.rel)));
        }
      }
    }
  }
  found
}

/// Whether relation `b` has rules, and each is `B(x) :- A(.. x ..), A(.. y
/// ..), y < x`, or with `>`, where x and y are in column `c` of relation
/// `a`, and the atoms' other columns are `_` or variables that appear
/// nowhere else: then `B` holds of every value of the column but the least,
/// or the greatest, and `not B(v)` picks that one.
fn witness(prog: &Program, b: usize, a: usize, c: usize) -> bool {
  let rules: Vec<&Rule> = prog.rules.iter().filter(|rule| rule.head == b).collect();
  !rules.is_empty()
    && rules.iter().all(|rule| {
      let [Expr::Var(x)] = rule.args.as_slice() else {
        return false;
      };
      let [Lit::Atom(p), Lit::Atom(q), Lit::Test(Test::Cmp(cmp, l, r))] = rule.body.as_slice() else {
        return false;
      };
      let strict = matches!(cmp, Cmp::Lt | Cmp::Gt);
      let var_at = |atom: &Atom| match atom.pats.get(c) {
        Some(Pat::Var(v)) => Some(*v),
        _ => None,
      };
      let (Some(px), Some(qy)) = (var_at(p), var_at(q)) else {
        return false;
      };
      let loose = |atom: &Atom, keep: usize| {
        (atom.pats.iter().enumerate()).all(|(i, pat)| {
          i == c
            || match pat {
              Pat::Any => true,
              Pat::Var(v) => *v != keep && uses(rule, *v) == 1,
              _ => false,
            }
        })
      };
      let tested = matches!((l, r), (Expr::Var(m), Expr::Var(n)) if (*m, *n) == (qy, px) || (*m, *n) == (px, qy));
      p.rel == a && q.rel == a && px == *x && qy != px && strict && tested && loose(p, px) && loose(q, qy)
    })
}

/// How many times variable `v` appears in `rule`'s body.
fn uses(rule: &Rule, v: usize) -> usize {
  fn in_pat(pat: &Pat, v: usize) -> usize {
    match pat {
      Pat::Var(w) => (*w == v) as usize,
      Pat::Ctor(_, pats) => pats.iter().map(|p| in_pat(p, v)).sum(),
      _ => 0,
    }
  }
  (rule.body.iter())
    .map(|lit| match lit {
      Lit::Atom(atom) | Lit::Not(atom) => atom.pats.iter().map(|p| in_pat(p, v)).sum(),
      _ => 0,
    })
    .sum()
}
