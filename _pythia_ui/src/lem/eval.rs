//! lem's runtime: the host's facts in, the rules run to a fixed point, and
//! the derived facts out; and messages, which `on` blocks turn into state
//! changes and effects.
//!
//! A tick ([`Runtime::tick`]) is the design's: the message's `on` blocks run
//! against the facts as they are, derived ones as the last tick left them;
//! their changes apply; each stratum's rules run until nothing new appears
//! (naively, for now); and the keys are checked. A layer above lem can add
//! its own checks after a tick, through [`Runtime::rel_facts`] and
//! [`Runtime::drop_facts`].

use super::check::{Change, DeclKind, Expr, Lit, Op, Pat, Program, Test};
use super::parse::Cmp;
use super::value::{int_fits, Ty, Value, MAX_INT};
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

/// A fact's values, a column each.
pub type Tuple = Rc<[Value]>;

type Env = Vec<Option<Value>>;

/// Why a fact holds: the host asserted it, an `on` block's sentence (an
/// action) added it, or a rule, the first, derived it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
  Host,
  Action(usize),
  Rule(usize),
}

/// What an `on` block sends to the host: an effect relation's fact, such
/// as `Tell("finished review")`.
#[derive(Clone, Debug, PartialEq)]
pub struct Effect {
  pub rel: Rc<str>,
  pub args: Tuple,
}

impl Display for Effect {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "{}", Value::Ctor(self.rel.clone(), self.args.clone()))
  }
}

pub struct Runtime {
  pub prog: Program,
  /// Each relation's facts, and why each holds.
  facts: Vec<BTreeMap<Tuple, Why>>,
}

impl Runtime {
  pub fn new(prog: Program) -> Runtime {
    let facts = vec![BTreeMap::new(); prog.rels.len()];
    Runtime { prog, facts }
  }

  /// A relation's facts, in order.
  pub fn facts(&self, rel: &str) -> Vec<Tuple> {
    match self.prog.rel(rel) {
      Some(rel) => self.facts[rel].keys().cloned().collect(),
      None => Vec::new(),
    }
  }

  /// A relation's facts, by its number in the program, in order, and why
  /// each holds.
  pub fn rel_facts(&self, rel: usize) -> impl Iterator<Item = (&Tuple, Why)> + '_ {
    self.facts[rel].iter().map(|(tuple, why)| (tuple, *why))
  }

  /// Drops a relation's facts that `drop` picks: for a layer above lem that
  /// cleans up between ticks.
  pub fn drop_facts(&mut self, rel: usize, mut drop: impl FnMut(&Tuple) -> bool) {
    self.facts[rel].retain(|tuple, _| !drop(tuple));
  }

  /// The host's fact, written in lem, such as `Now(1000)`. In a relation
  /// with a key, it replaces the fact with its key.
  pub fn assert(&mut self, text: &str) -> Result<(), String> {
    let (rel, tuple) = self.input(text)?;
    let r = &self.prog.rels[rel];
    let key: Vec<usize> = r.key.clone();
    if !key.is_empty() {
      self.facts[rel].retain(|t, _| key.iter().any(|&i| t[i] != tuple[i]));
    }
    self.facts[rel].insert(tuple, Why::Host);
    Ok(())
  }

  /// Retracts the host's fact, written in lem; whether it held.
  pub fn retract(&mut self, text: &str) -> Result<bool, String> {
    let (rel, tuple) = self.input(text)?;
    Ok(self.facts[rel].remove(&tuple).is_some())
  }

  fn input(&mut self, text: &str) -> Result<(usize, Tuple), String> {
    let (rel, values) = self.prog.fact(text)?;
    let r = &self.prog.rels[rel];
    if r.kind != DeclKind::Input {
      return Err(format!(
        "{} isn't an `input` relation, so the host can't change it",
        r.name
      ));
    }
    Ok((rel, values.into()))
  }

  /// A message, written in lem, such as `Finish("job_1")`.
  pub fn msg(&mut self, text: &str) -> Result<Value, String> {
    self.prog.value(text, &Ty::Named("Msg".into()))
  }

  /// Reads a tick written as lines, as the tests' files and `pythia-ui lem`
  /// give them: `+ Fact` asserts the host's fact, `- Fact` retracts it,
  /// `! Msg` is the tick's message, and `? Name` asks for a relation's facts
  /// after the tick. Asserts and retracts as the lines say, and gives the
  /// message and the relations to show.
  pub fn read_script(&mut self, lines: &str) -> Result<Script, String> {
    let (mut msg, mut show) = (None, Vec::new());
    for line in lines.lines() {
      if let Some(fact) = line.strip_prefix("+ ") {
        self.assert(fact)?;
      } else if let Some(fact) = line.strip_prefix("- ") {
        if !self.retract(fact)? {
          return Err(format!("{fact} doesn't hold, so it can't be retracted"));
        }
      } else if let Some(text) = line.strip_prefix("! ") {
        if msg.is_some() {
          return Err("a tick has one message at most".to_string());
        }
        msg = Some(self.msg(text)?);
      } else if let Some(rel) = line.strip_prefix("? ") {
        if self.prog.rel(rel.trim()).is_none() {
          return Err(format!("there's no relation `{}`", rel.trim()));
        }
        show.push(rel.trim().to_string());
      } else if !line.trim().is_empty() {
        return Err(format!(
          "a tick's lines start with `+ `, `- `, `! ` or `? `, not {line:?}"
        ));
      }
    }
    Ok(Script { msg, show })
  }

  /// The facts of the relations that a script asked for, then the effects,
  /// as `effect Send(…)`, a line each.
  pub fn show(&self, script: &Script, effects: &[Effect]) -> String {
    let mut out = String::new();
    for rel in &script.show {
      for tuple in self.facts(rel) {
        out.push_str(&format!("{}\n", Value::Ctor(rel.as_str().into(), tuple)));
      }
    }
    for effect in effects {
      out.push_str(&format!("effect {effect}\n"));
    }
    out
  }

  /// Runs a tick written as lines ([`Runtime::read_script`]), and gives
  /// what [`Runtime::show`] gives, or `(nothing to show)`.
  pub fn script(&mut self, lines: &str) -> Result<String, String> {
    let script = self.read_script(lines)?;
    let effects = self.tick(script.msg.as_ref())?;
    let out = self.show(&script, &effects);
    Ok(match out.trim_end() {
      "" => "(nothing to show)".to_string(),
      out => out.to_string(),
    })
  }

  /// One tick: the message's `on` blocks, if there's a message, then the
  /// rules. Gives the effects that the `on` blocks sent, in order.
  pub fn tick(&mut self, msg: Option<&Value>) -> Result<Vec<Effect>, String> {
    let mut effects = Vec::new();
    if let Some(msg) = msg {
      let (inserts, deletes) = self.update(msg)?;
      for (rel, tuple, a) in &inserts {
        if let Some((_, _, d)) = deletes.iter().find(|(r, t, _)| r == rel && t == tuple) {
          let fact = self.fact_text(*rel, tuple);
          let (at, other) = (self.why(Why::Action(*a)), self.why(Why::Action(*d)));
          return Err(format!("{fact} is both added ({at}) and deleted ({other})"));
        }
      }
      for (rel, tuple, _) in deletes {
        self.facts[rel].remove(&tuple);
      }
      let mut added: BTreeMap<(usize, Vec<Value>), (Tuple, usize)> = BTreeMap::new();
      for (rel, tuple, a) in inserts {
        let r = &self.prog.rels[rel];
        if r.kind == DeclKind::Effect {
          effects.push(Effect {
            rel: r.name.clone(),
            args: tuple,
          });
          continue;
        }
        let key: Vec<usize> = if r.key.is_empty() {
          (0..r.fields.len()).collect()
        } else {
          r.key.clone()
        };
        let k: Vec<Value> = key.iter().map(|&i| tuple[i].clone()).collect();
        if let Some((other, b)) = added.get(&(rel, k.clone())) {
          if *other != tuple {
            let (x, y) = (self.fact_text(rel, other), self.fact_text(rel, &tuple));
            let (at, other) = (self.why(Why::Action(*b)), self.why(Why::Action(a)));
            return Err(format!("one key, two facts: {x} ({at}) and {y} ({other})"));
          }
        }
        self.facts[rel].retain(|t, _| key.iter().any(|&i| t[i] != tuple[i]));
        self.facts[rel].insert(tuple.clone(), Why::Action(a));
        added.insert((rel, k), (tuple, a));
      }
    }
    self.derive()?;
    self.check_keys()?;
    Ok(effects)
  }

  /// What the `on` blocks of a message would add and delete: (relation,
  /// fact, action) each.
  #[allow(clippy::type_complexity)]
  fn update(
    &self,
    msg: &Value,
  ) -> Result<(Vec<(usize, Tuple, usize)>, Vec<(usize, Tuple, usize)>), String> {
    let (mut inserts, mut deletes) = (Vec::new(), Vec::new());
    for (a, action) in self.prog.actions.iter().enumerate() {
      let mut env = vec![None; action.vars.len()];
      if !match_pat(&action.msg, msg, &mut env, &mut Vec::new())? {
        continue;
      }
      self.solve(&action.body, &mut env, &mut |env| {
        match &action.change {
          Change::Insert(exprs) => {
            let values = exprs
              .iter()
              .map(|e| eval(e, env))
              .collect::<Result<Vec<_>, _>>()?;
            inserts.push((action.rel, values.into(), a));
          }
          Change::Delete(pats) => {
            for tuple in self.facts[action.rel].keys() {
              if match_all(pats, tuple, &mut env.clone(), &mut Vec::new())? {
                deletes.push((action.rel, tuple.clone(), a));
              }
            }
          }
        }
        Ok(())
      })?;
    }
    Ok((inserts, deletes))
  }

  /// Runs the rules again from the input and state facts: each stratum, in
  /// order, until nothing new appears.
  fn derive(&mut self) -> Result<(), String> {
    for (facts, rel) in self.facts.iter_mut().zip(&self.prog.rels) {
      if rel.kind == DeclKind::Rel {
        facts.clear();
      }
    }
    for stratum in 0..self.prog.strata.len() {
      loop {
        let mut new = Vec::new();
        for &r in &self.prog.strata[stratum] {
          let rule = &self.prog.rules[r];
          let mut env = vec![None; rule.vars.len()];
          self.solve(&rule.body, &mut env, &mut |env| {
            let values = rule
              .args
              .iter()
              .map(|e| eval(e, env))
              .collect::<Result<Vec<_>, _>>();
            let values =
              values.map_err(|e| format!("{e}, in the rule at {}", self.prog.place(&rule.at)))?;
            new.push((rule.head, Tuple::from(values), r));
            Ok(())
          })?;
        }
        let mut changed = false;
        for (rel, tuple, r) in new {
          if let Entry::Vacant(entry) = self.facts[rel].entry(tuple) {
            entry.insert(Why::Rule(r));
            changed = true;
          }
        }
        if !changed {
          break;
        }
      }
    }
    Ok(())
  }

  /// Calls `out` with each solution of a body, as the variables' values.
  fn solve(
    &self,
    lits: &[Lit],
    env: &mut Env,
    out: &mut dyn FnMut(&Env) -> Result<(), String>,
  ) -> Result<(), String> {
    let Some((lit, rest)) = lits.split_first() else {
      return out(env);
    };
    match lit {
      Lit::Atom(rel, pats) => {
        for tuple in self.facts[*rel].keys() {
          let mut trail = Vec::new();
          if match_all(pats, tuple, env, &mut trail)? {
            self.solve(rest, env, out)?;
          }
          trail.iter().for_each(|&v| env[v] = None);
        }
        Ok(())
      }
      Lit::Let(v, e) => {
        env[*v] = Some(eval(e, env)?);
        let result = self.solve(rest, env, out);
        env[*v] = None;
        result
      }
      Lit::Test(t) if test(t, env)? => self.solve(rest, env, out),
      Lit::NotTest(t) if !test(t, env)? => self.solve(rest, env, out),
      Lit::Test(_) | Lit::NotTest(_) => Ok(()),
      Lit::Not(rel, pats) => {
        for tuple in self.facts[*rel].keys() {
          let mut trail = Vec::new();
          let found = match_all(pats, tuple, env, &mut trail)?;
          trail.iter().for_each(|&v| env[v] = None);
          if found {
            return Ok(());
          }
        }
        self.solve(rest, env, out)
      }
    }
  }

  /// Where a fact comes from, for errors.
  pub fn why(&self, why: Why) -> String {
    match why {
      Why::Host => "from the host".to_string(),
      Why::Action(a) => format!(
        "from the `on` block at {}",
        self.prog.place(&self.prog.actions[a].at)
      ),
      Why::Rule(r) => format!(
        "from the rule at {}",
        self.prog.place(&self.prog.rules[r].at)
      ),
    }
  }

  /// A fact as lem writes it, its values by position.
  pub fn fact_text(&self, rel: usize, tuple: &Tuple) -> String {
    Value::Ctor(self.prog.rels[rel].name.clone(), tuple.clone()).to_string()
  }

  /// Facts that agree on their relation's key agree on the rest.
  fn check_keys(&self) -> Result<(), String> {
    for (rel, r) in self.prog.rels.iter().enumerate() {
      if r.key.is_empty() {
        continue;
      }
      let mut seen: BTreeMap<Vec<Value>, (&Tuple, Why)> = BTreeMap::new();
      for (tuple, why) in &self.facts[rel] {
        let key: Vec<Value> = r.key.iter().map(|&i| tuple[i].clone()).collect();
        if let Some((other, first)) = seen.get(&key) {
          let (x, y) = (self.fact_text(rel, other), self.fact_text(rel, tuple));
          let (at, other) = (self.why(*first), self.why(*why));
          return Err(format!("one key, two facts: {x} ({at}) and {y} ({other})"));
        }
        seen.insert(key, (tuple, *why));
      }
    }
    Ok(())
  }
}

/// A tick written as lines ([`Runtime::read_script`]): its message, and the
/// relations whose facts to show after it.
#[derive(Clone, Debug, Default)]
pub struct Script {
  pub msg: Option<Value>,
  pub show: Vec<String>,
}

fn match_all(
  pats: &[Pat],
  tuple: &Tuple,
  env: &mut Env,
  trail: &mut Vec<usize>,
) -> Result<bool, String> {
  for (pat, value) in pats.iter().zip(tuple.iter()) {
    if !match_pat(pat, value, env, trail)? {
      return Ok(false);
    }
  }
  Ok(true)
}

/// Whether a value matches; binds the pattern's unbound variables, and
/// records them in `trail`, so that the caller can unbind them.
fn match_pat(
  pat: &Pat,
  value: &Value,
  env: &mut Env,
  trail: &mut Vec<usize>,
) -> Result<bool, String> {
  Ok(match pat {
    Pat::Any => true,
    Pat::Var(v) => match &env[*v] {
      Some(bound) => bound == value,
      None => {
        env[*v] = Some(value.clone());
        trail.push(*v);
        true
      }
    },
    Pat::Const(c) => c == value,
    Pat::Ctor(name, pats) => match value {
      Value::Ctor(n, args) if n == name && args.len() == pats.len() => {
        for (pat, arg) in pats.iter().zip(args.iter()) {
          if !match_pat(pat, arg, env, trail)? {
            return Ok(false);
          }
        }
        true
      }
      _ => false,
    },
    Pat::Expr(e) => eval(e, env)? == *value,
  })
}

fn eval(e: &Expr, env: &Env) -> Result<Value, String> {
  Ok(match e {
    Expr::Var(v) => env[*v]
      .clone()
      .ok_or("a variable isn't bound (a bug in lem's checker)")?,
    Expr::Const(c) => c.clone(),
    Expr::Ctor(name, args) => {
      let args = args
        .iter()
        .map(|a| eval(a, env))
        .collect::<Result<Vec<_>, _>>()?;
      Value::Ctor(name.clone(), args.into())
    }
    Expr::Neg(e) => match eval(e, env)? {
      Value::Int(n) => Value::Int(
        n.checked_neg()
          .filter(|n| int_fits(*n))
          .ok_or_else(|| int_overflow("-"))?,
      ),
      Value::Num(x) => Value::num(-x).ok_or_else(|| num_overflow("-"))?,
      v => return Err(format!("`-` can't negate {v}")),
    },
    Expr::ToNum(e) => match eval(e, env)? {
      Value::Int(n) => Value::Num(n as f64),
      v => v,
    },
    Expr::Arith(op, a, b) => arith(*op, eval(a, env)?, eval(b, env)?)?,
    Expr::Text(e) => Value::Text(eval(e, env)?.text()),
  })
}

fn int_overflow(op: &str) -> String {
  format!("`{op}` overflowed: a whole number stays within ±{MAX_INT} (2^53 − 1)")
}

fn num_overflow(op: &str) -> String {
  format!("`{op}` overflowed: a Num stays finite")
}

fn arith(op: Op, a: Value, b: Value) -> Result<Value, String> {
  Ok(match (a, b) {
    (Value::Int(a), Value::Int(b)) => Value::Int(
      match op {
        Op::Add => a.checked_add(b),
        Op::Sub => a.checked_sub(b),
        Op::Mul => a.checked_mul(b),
        Op::Div => return arith(op, Value::Num(a as f64), Value::Num(b as f64)),
      }
      .filter(|n| int_fits(*n))
      .ok_or_else(|| int_overflow(op.spelling()))?,
    ),
    (Value::Num(_), Value::Num(b)) if op == Op::Div && b == 0.0 => {
      return Err("a division by zero".to_string());
    }
    (Value::Num(a), Value::Num(b)) => Value::num(match op {
      Op::Add => a + b,
      Op::Sub => a - b,
      Op::Mul => a * b,
      Op::Div => a / b,
    })
    .ok_or_else(|| num_overflow(op.spelling()))?,
    (Value::Text(a), Value::Text(b)) if op == Op::Add => Value::Text(format!("{a}{b}").into()),
    (a, b) => return Err(format!("`{}` can't take {a} and {b}", op.spelling())),
  })
}

fn test(t: &Test, env: &Env) -> Result<bool, String> {
  Ok(match t {
    Test::Eq(a, b) => eval(a, env)? == eval(b, env)?,
    Test::Ne(a, b) => eval(a, env)? != eval(b, env)?,
    Test::Cmp(cmp, a, b) => {
      let order = eval(a, env)?.cmp(&eval(b, env)?);
      match cmp {
        Cmp::Lt => order.is_lt(),
        Cmp::LtEq => order.is_le(),
        Cmp::Gt => order.is_gt(),
        Cmp::GtEq => order.is_ge(),
      }
    }
  })
}
