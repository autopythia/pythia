//! lem's runtime: the host's facts in, the rules run to a fixed point, and
//! the derived facts out; and messages, which `on` blocks turn into state
//! changes and effects.
//!
//! A tick ([`Runtime::tick`]) is the design's: the message's `on` blocks run
//! against the facts as they are, derived ones as the last tick left them;
//! their changes apply; the rules run, component by component, in the
//! checker's order; and the keys are checked. A layer above lem can add its
//! own checks after a tick, through [`Runtime::rel_facts`] and
//! [`Runtime::drop_facts`].
//!
//! A tick runs a component's rules only if a relation that they read has
//! changed since they last ran. Each relation's table has a version, which
//! every change to its facts moves on, and each component keeps the versions of
//! what it read when it last ran. A recursive component runs again whole
//! ([`Runtime::rerun`]): a relation whose facts come out the same keeps its
//! table, version and all, so that what reads it needn't run either.
//!
//! A component that isn't recursive follows what changed instead
//! (`Runtime::follow`), once it has run: for each rule, it keeps how many ways
//! the rule derives each fact, and each table logs the facts that came and went
//! since the last tick (`Log`). A rule that reads one relation that changed
//! joins there just those facts, with the rest of its body as it is, where that
//! counts exactly (`followed`); else it runs whole. The relation's facts change
//! in place, by the counts, and keep their indexes. Keys are checked by a count
//! of each relation's facts by its key's values, which the tables keep
//! (`Table::key_count`). A component that has no facts to keep, and most of
//! whose inputs are new, starts over instead, running whole as on its first
//! tick (`Runtime::starts_over`).
//!
//! An error while the rules run leaves the relations derived after the failing
//! component without facts, as every tick that derived everything left them,
//! and forgets what each component read, so that the next tick runs every rule;
//! a component that fails to follow what changed runs again whole, so that it
//! fails as such a tick does. The facts, the effects and the errors are those
//! of a tick that derives everything anew, which [`Runtime::forget`] makes the
//! next tick do, for the tests that compare the two. Both runtimes' joins
//! visit the same facts, since they take the facts that came in one order.
//!
//! The rules run cheaply. A body's positive atoms join in the order that
//! their facts ask for, after meringue's joins
//! (`doc/pythia-ui-gui-design.md` in pythia): at each step, every atom that's
//! ready is looked up by the columns bound so far, in an index on them; if
//! one has no facts there, the step fails; else the join takes the atom
//! with the fewest, the first written on a tie. A recursive component runs
//! semi-naively: each round joins what the last round added, at one atom,
//! with all the facts at the others. Whatever order a join takes, its
//! results are the same, so the runtimes agree; only an error in an atom's
//! expression may come at another step.

use super::check::{Atom, Change, DeclKind, Expr, Lit, Op, Pat, Program, Rule, Test, MAX_ATOMS};
use super::parse::Cmp;
use super::value::{int_fits, Ty, Value, MAX_INT};
use std::cell::{Cell, RefCell};
use std::collections::btree_map::{self, Entry};
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::rc::Rc;

/// A fact's values, a column each.
pub type Tuple = Rc<[Value]>;

type Env = Vec<Option<Value>>;

/// Why a fact holds: the host asserted it, an `on` block's sentence (an
/// action) added it, or rules derived it: the lowest-numbered of them, so
/// that any order of evaluation names the same.
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

/// An index: by the values of some columns, the facts that have them.
type Index = HashMap<Vec<Value>, Vec<Tuple>>;

/// A fact's values in `cols`.
fn project(tuple: &Tuple, cols: &[usize]) -> Vec<Value> {
  cols.iter().map(|&c| tuple[c].clone()).collect()
}

/// Indexes by their columns, built as joins ask for them.
type Indexes = RefCell<HashMap<Vec<usize>, Rc<Index>>>;

/// The index on `cols` in `indexes`, built from `tuples` if it isn't there.
fn index<'a>(
  indexes: &Indexes,
  cols: &[usize],
  tuples: impl Iterator<Item = &'a Tuple>,
) -> Rc<Index> {
  if let Some(index) = indexes.borrow().get(cols) {
    return index.clone();
  }
  let mut index = Index::new();
  for tuple in tuples {
    index
      .entry(project(tuple, cols))
      .or_default()
      .push(tuple.clone());
  }
  let index = Rc::new(index);
  indexes.borrow_mut().insert(cols.to_vec(), index.clone());
  index
}

/// Facts in the order that they came, each with a value, as a JavaScript
/// `Map` keeps its keys, so that both runtimes take them in one order: a
/// fact that's set again keeps its place; one that goes, and comes back,
/// comes last. Each fact's place is found by a map made only once a fact
/// is looked up among others, so that facts that only come cost no hashing.
struct Ordered<V> {
  order: Vec<Option<(Tuple, V)>>,
  at: Option<HashMap<Tuple, usize>>,
  len: usize,
}

impl<V> Default for Ordered<V> {
  fn default() -> Self {
    Ordered {
      order: Vec::new(),
      at: None,
      len: 0,
    }
  }
}

impl<V> Ordered<V> {
  fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Each fact's place in `order`, made the first time it's asked for.
  fn places(&mut self) -> &mut HashMap<Tuple, usize> {
    let order = &self.order;
    self.at.get_or_insert_with(|| {
      let held = order.iter().enumerate();
      held
        .filter_map(|(i, kept)| Some((kept.as_ref()?.0.clone(), i)))
        .collect()
    })
  }

  fn get_mut(&mut self, tuple: &Tuple) -> Option<&mut V> {
    if self.len == 0 {
      return None;
    }
    let &i = self.places().get(tuple)?;
    self.order[i].as_mut().map(|(_, value)| value)
  }

  /// Sets a fact's value: in its place, if it's here; else last.
  fn insert(&mut self, tuple: Tuple, value: V) {
    let at = match self.len {
      0 => None,
      _ => self.places().get(&tuple).copied(),
    };
    match at {
      Some(i) => self.order[i] = Some((tuple, value)),
      None => self.push(tuple, value),
    }
  }

  /// Puts last a fact that isn't here.
  fn push(&mut self, tuple: Tuple, value: V) {
    if let Some(at) = &mut self.at {
      at.insert(tuple.clone(), self.order.len());
    }
    self.order.push(Some((tuple, value)));
    self.len += 1;
  }

  fn remove(&mut self, tuple: &Tuple) -> Option<V> {
    if self.len == 0 {
      return None;
    }
    let i = self.places().remove(tuple)?;
    self.len -= 1;
    self.order[i].take().map(|(_, value)| value)
  }

  fn iter(&self) -> impl Iterator<Item = (&Tuple, &V)> {
    self
      .order
      .iter()
      .flatten()
      .map(|(tuple, value)| (tuple, value))
  }
}

/// What came to a relation, and what went from it, since the last tick that
/// ran the rules ([`Runtime::settle`]), and none before the first, when
/// nothing follows what changed yet: in the order that they did. `lem.js`
/// keeps the facts that came in the same order, which sets the order of
/// what follows them in the indexes ([`Runtime::follow`]): a whole run notes
/// those that it adds in the order that its rules first derived them
/// ([`Runtime::rerun`]). The order of those that went matters to nothing,
/// since an index keeps the rest in their order.
#[derive(Default)]
struct Log {
  came: Ordered<()>,
  went: Ordered<()>,
}

impl Log {
  /// Notes that a fact came, or went: one that went and comes back, or came
  /// and goes again, is no change.
  fn note(&mut self, tuple: &Tuple, came: bool) {
    let (to, from) = match came {
      true => (&mut self.came, &mut self.went),
      false => (&mut self.went, &mut self.came),
    };
    if from.remove(tuple).is_none() {
      to.push(tuple.clone(), ());
    }
  }

  /// Whether anything came or went.
  fn changed(&self) -> bool {
    !self.came.is_empty() || !self.went.is_empty()
  }
}

/// How many of a relation's facts have each of its key's values, and how
/// many of those values more than one fact has ([`Table::key_count`]).
struct Keys {
  cols: Vec<usize>,
  counts: HashMap<Vec<Value>, u32>,
  clashes: usize,
}

impl Keys {
  /// Counts a fact's key's values once more, or once less.
  fn count(&mut self, tuple: &Tuple, more: bool) {
    let k = project(tuple, &self.cols);
    let n = self.counts.get(&k).copied().unwrap_or(0);
    if more {
      self.clashes += usize::from(n == 1);
      self.counts.insert(k, n + 1);
    } else {
      self.clashes -= usize::from(n == 2);
      match n {
        1 => _ = self.counts.remove(&k),
        _ => _ = self.counts.insert(k, n - 1),
      }
    }
  }
}

/// A relation's facts, and why each holds; and indexes on them, which it
/// keeps as facts come and go. And, for the host's facts in a relation with
/// a key, the facts by their key's values ([`Table::put_keyed`]), which only
/// the host's changes keep. Its version moves on with each change to its
/// facts but a rule's ([`Table::changed`]): a table that rules fill is new,
/// a version on from the one that it replaces ([`Runtime::rerun`]). Its log
/// ([`Log`]) notes what comes and goes; and, once asked for, it counts the
/// facts by their key's values ([`Table::key_count`]).
#[derive(Default)]
struct Table {
  facts: BTreeMap<Tuple, Why>,
  indexes: Indexes,
  keyed: Option<HashMap<Vec<Value>, Tuple>>,
  version: u64,
  log: Option<Log>,
  keys: Option<Keys>,
}

impl Table {
  /// The index on `cols`, which mustn't be empty.
  fn index(&self, cols: &[usize]) -> Rc<Index> {
    index(&self.indexes, cols, self.facts.keys())
  }

  /// A fact that rule `rule` derives, into the indexes too: whether it's
  /// new. One that holds already keeps the lowest-numbered of its rules.
  fn derive(&mut self, tuple: &Tuple, rule: usize) -> bool {
    match self.facts.entry(tuple.clone()) {
      Entry::Occupied(mut entry) => {
        if matches!(entry.get(), Why::Rule(r) if rule < *r) {
          entry.insert(Why::Rule(rule));
        }
        false
      }
      Entry::Vacant(entry) => {
        entry.insert(Why::Rule(rule));
        for (cols, index) in self.indexes.get_mut().iter_mut() {
          Rc::make_mut(index)
            .entry(project(tuple, cols))
            .or_default()
            .push(tuple.clone());
        }
        if let Some(keys) = &mut self.keys {
          keys.count(tuple, true);
        }
        true
      }
    }
  }

  /// Adds a fact that doesn't hold, and why it holds: to the indexes too,
  /// and to the log.
  fn add(&mut self, tuple: Tuple, why: Why) {
    for (cols, index) in self.indexes.get_mut().iter_mut() {
      Rc::make_mut(index)
        .entry(project(&tuple, cols))
        .or_default()
        .push(tuple.clone());
    }
    if let Some(keys) = &mut self.keys {
      keys.count(&tuple, true);
    }
    if let Some(log) = &mut self.log {
      log.note(&tuple, true);
    }
    self.facts.insert(tuple, why);
  }

  /// Takes away a fact that holds: from the indexes too, where the rest keep
  /// their order, and notes it in the log.
  fn cut(&mut self, tuple: &Tuple) {
    self.facts.remove(tuple);
    for (cols, index) in self.indexes.get_mut().iter_mut() {
      let index = Rc::make_mut(index);
      let key = project(tuple, cols);
      let bucket = index.get_mut(&key).expect("an index has its table's facts");
      let at = bucket.iter().position(|t| t == tuple);
      bucket.remove(at.expect("an index has its table's facts"));
      if bucket.is_empty() {
        index.remove(&key);
      }
    }
    if let Some(keys) = &mut self.keys {
      keys.count(tuple, false);
    }
    if let Some(log) = &mut self.log {
      log.note(tuple, false);
    }
  }

  /// The facts changed otherwise than by the host's facts with a key
  /// ([`Table::put_keyed`]), which no longer finds them by key; and the
  /// version moves on.
  fn changed(&mut self) {
    self.keyed = None;
    self.version += 1;
  }

  /// A fact that doesn't hold, and why it holds.
  fn put(&mut self, tuple: Tuple, why: Why) {
    self.add(tuple, why);
    self.changed();
  }

  /// Removes a fact: whether it held. One that doesn't hold changes nothing.
  fn remove(&mut self, tuple: &Tuple) -> bool {
    if !self.facts.contains_key(tuple) {
      return false;
    }
    self.cut(tuple);
    self.changed();
    true
  }

  /// Drops the facts that `drop` picks. If it picks none, nothing changes,
  /// so the table keeps its version.
  fn drop_where(&mut self, mut drop: impl FnMut(&Tuple) -> bool) {
    let gone: Vec<Tuple> = self.facts.keys().filter(|t| drop(t)).cloned().collect();
    if gone.is_empty() {
      return;
    }
    for tuple in &gone {
      self.cut(tuple);
    }
    self.changed();
  }

  /// Takes away every fact, if it has any, noting each in the log; so its
  /// indexes go.
  fn clear(&mut self) {
    if self.facts.is_empty() {
      return;
    }
    let facts = std::mem::take(&mut self.facts);
    if let Some(log) = &mut self.log {
      facts.into_keys().for_each(|tuple| log.note(&tuple, false));
    }
    self.indexes.get_mut().clear();
    self.keys = None;
    self.changed();
  }

  /// The facts counted by their values in the columns `cols`, the relation's
  /// key ([`Keys`]): counted once, without sorting, and kept as facts come
  /// and go.
  fn key_count(&mut self, cols: &[usize]) -> &Keys {
    let facts = &self.facts;
    self.keys.get_or_insert_with(|| {
      let mut keys = Keys {
        cols: cols.to_vec(),
        counts: HashMap::new(),
        clashes: 0,
      };
      facts.keys().for_each(|tuple| keys.count(tuple, true));
      keys
    })
  }

  /// The host's fact, in place of the one with its values in the columns
  /// `cols`, the relation's key. The facts by key, made once, find that
  /// one, so that the host's facts take time linear in their number.
  fn put_keyed(&mut self, tuple: Tuple, cols: &[usize]) {
    let mut keyed = match self.keyed.take() {
      Some(keyed) => keyed,
      None => self
        .facts
        .keys()
        .map(|t| (project(t, cols), t.clone()))
        .collect(),
    };
    if let Some(old) = keyed.insert(project(&tuple, cols), tuple.clone()) {
      self.cut(&old);
    }
    self.add(tuple, Why::Host);
    self.keyed = Some(keyed);
    self.version += 1;
  }

  /// Removes the host's fact, which holds, from a relation whose key is
  /// `cols`, and from the facts by key.
  fn remove_keyed(&mut self, tuple: &Tuple, cols: &[usize]) {
    self.cut(tuple);
    if let Some(keyed) = &mut self.keyed {
      keyed.remove(&project(tuple, cols));
    }
    self.version += 1;
  }
}

/// What a recursive round added to a relation, and indexes on it.
struct Fresh {
  tuples: Vec<Tuple>,
  indexes: Indexes,
}

/// The facts that a lookup gives: all of a relation's, or a key's.
enum Tuples<'a> {
  All(btree_map::Keys<'a, Tuple, Why>),
  Some(std::slice::Iter<'a, Tuple>),
}

impl<'a> Iterator for Tuples<'a> {
  type Item = &'a Tuple;

  fn next(&mut self) -> Option<&'a Tuple> {
    match self {
      Tuples::All(tuples) => tuples.next(),
      Tuples::Some(tuples) => tuples.next(),
    }
  }
}

/// The facts that might match an atom, given the columns bound so far.
enum Bucket<'a> {
  /// All of a relation's: no column is bound.
  Table(&'a BTreeMap<Tuple, Why>),
  /// All that a round added: no column is bound.
  Fresh(&'a [Tuple]),
  /// Those with a key's values, in an index on its columns.
  Index(Rc<Index>, Vec<Value>),
}

impl Bucket<'_> {
  fn len(&self) -> usize {
    match self {
      Bucket::Table(facts) => facts.len(),
      Bucket::Fresh(tuples) => tuples.len(),
      Bucket::Index(index, key) => index.get(key).map_or(0, Vec::len),
    }
  }

  fn tuples(&self) -> Tuples<'_> {
    match self {
      Bucket::Table(facts) => Tuples::All(facts.keys()),
      Bucket::Fresh(tuples) => Tuples::Some(tuples.iter()),
      Bucket::Index(index, key) => {
        Tuples::Some(index.get(key).map_or([].iter(), |tuples| tuples.iter()))
      }
    }
  }
}

/// A recursive round's new facts, for the atom at a place in a body.
type Delta<'a> = Option<(usize, &'a Fresh)>;

/// What a join reads, the tables, and where it counts the facts it visits.
#[derive(Clone, Copy)]
struct Join<'a> {
  tables: &'a [Table],
  visits: &'a Cell<u64>,
}

/// A relation's new facts, from (relation, fact, rule) triples.
type New = Vec<(usize, Tuple, usize)>;

/// What a tick does for a rule of a component that it follows
/// ([`Runtime::follow`]): nothing; join just what came and went at the atom
/// in a place in its body; or run whole.
enum Plan {
  Nothing,
  At(usize),
  Whole,
}

pub struct Runtime {
  pub prog: Program,
  /// Each relation's facts and indexes.
  tables: Vec<Table>,
  /// The facts that the last tick's joins visited.
  visits: Cell<u64>,
  /// The components that a tick skips ([`skipped`]), unless the host asks
  /// for one of their relations, which the next tick derives: `asked`.
  skip: Vec<bool>,
  asked: Vec<usize>,
  /// For each component, the relations that its rules read ([`reads`]);
  /// and the versions that they had when it last ran, or none: it hasn't
  /// run since the runtime began, or since the rules failed, or a tick
  /// skipped it.
  reads: Vec<Vec<usize>>,
  ran: Vec<Option<Vec<u64>>>,
  /// For each relation, the version at which its keys last held, if any.
  held: Vec<Option<u64>>,
  /// For each rule, of a component that isn't recursive, how many ways it
  /// derives each fact, as its component last ran ([`Runtime::follow`]);
  /// and for each of its atoms, whether a tick follows a change to that
  /// atom's relation alone ([`followed`]).
  counts: Vec<HashMap<Tuple, u64>>,
  followed: Vec<Vec<bool>>,
  /// Whether the next tick derives every relation anew ([`Runtime::forget`]).
  anew: bool,
}

impl Runtime {
  pub fn new(prog: Program) -> Runtime {
    let tables = prog.rels.iter().map(|_| Table::default()).collect();
    let visits = Cell::new(0);
    let skip = skipped(&prog, &[]);
    let reads = reads(&prog);
    let ran = vec![None; prog.components.len()];
    let held = vec![None; prog.rels.len()];
    let counts = vec![HashMap::new(); prog.rules.len()];
    let followed = prog.rules.iter().map(followed).collect();
    Runtime {
      prog,
      tables,
      visits,
      skip,
      asked: Vec::new(),
      reads,
      ran,
      held,
      counts,
      followed,
      anew: false,
    }
  }

  /// Forgets what the components read when they last ran, and which keys
  /// held: the next tick, after its `on` blocks, clears every derived
  /// relation, runs every rule and checks every key, as every tick did
  /// before ticks kept what they derived. For tests, which compare the two.
  pub fn forget(&mut self) {
    self.ran.fill(None);
    self.held.fill(None);
    self.anew = true;
  }

  /// The facts that the last tick's joins visited: a measure of its work,
  /// which tests of how joins scale read, rather than times.
  pub fn visits(&self) -> u64 {
    self.visits.get()
  }

  /// A relation's facts, in order.
  pub fn facts(&self, rel: &str) -> Vec<Tuple> {
    match self.prog.rel(rel) {
      Some(rel) => self.tables[rel].facts.keys().cloned().collect(),
      None => Vec::new(),
    }
  }

  /// A relation's facts, by its number in the program, in order, and why
  /// each holds.
  pub fn rel_facts(&self, rel: usize) -> impl Iterator<Item = (&Tuple, Why)> + '_ {
    self.tables[rel]
      .facts
      .iter()
      .map(|(tuple, why)| (tuple, *why))
  }

  /// Drops a relation's facts that `drop` picks: for a layer above lem that
  /// cleans up between ticks. If it picks none, nothing changes, so the
  /// relation keeps its version.
  pub fn drop_facts(&mut self, rel: usize, mut drop: impl FnMut(&Tuple) -> bool) {
    self.tables[rel].drop_where(|t| drop(t));
  }

  /// The host's fact, written in lem, such as `Now(1000)`
  /// ([`Runtime::assert_tuple`]).
  pub fn assert(&mut self, text: &str) -> Result<(), String> {
    let (rel, values) = self.prog.fact(text)?;
    self.assert_tuple(rel, values.into())
  }

  /// Retracts the host's fact, written in lem; whether it held.
  pub fn retract(&mut self, text: &str) -> Result<bool, String> {
    let (rel, values) = self.prog.fact(text)?;
    self.retract_tuple(rel, &values.into())
  }

  /// The host's fact, in an `input` relation. In a relation with a key, it
  /// replaces the fact with its key. A fact that holds already changes
  /// nothing.
  pub fn assert_tuple(&mut self, rel: usize, tuple: Tuple) -> Result<(), String> {
    self.input(rel)?;
    if self.tables[rel].facts.contains_key(&tuple) {
      return Ok(());
    }
    let key = &self.prog.rels[rel].key;
    match key.is_empty() {
      true => self.tables[rel].put(tuple, Why::Host),
      false => self.tables[rel].put_keyed(tuple, key),
    }
    Ok(())
  }

  /// Retracts the host's fact; whether it held. One that doesn't hold
  /// changes nothing.
  pub fn retract_tuple(&mut self, rel: usize, tuple: &Tuple) -> Result<bool, String> {
    self.input(rel)?;
    if !self.tables[rel].facts.contains_key(tuple) {
      return Ok(false);
    }
    let key = &self.prog.rels[rel].key;
    match key.is_empty() {
      true => _ = self.tables[rel].remove(tuple),
      false => self.tables[rel].remove_keyed(tuple, key),
    }
    Ok(true)
  }

  /// Whether the host may change a relation: only an `input` one.
  fn input(&self, rel: usize) -> Result<(), String> {
    let r = &self.prog.rels[rel];
    if r.kind != DeclKind::Input {
      return Err(format!(
        "{} isn't an `input` relation, so the host can't change it",
        r.name
      ));
    }
    Ok(())
  }

  /// A message, written in lem, such as `Finish("job_1")`.
  pub fn msg(&mut self, text: &str) -> Result<Value, String> {
    self.prog.value(text, &Ty::Named("Msg".into()))
  }

  /// Reads a tick written as lines ([`read_step`]), as the tests' files and
  /// `pythia-ui lem` give them, and takes each step as it's read
  /// ([`Runtime::step`]): gives the message and the relations to show.
  pub fn read_script(&mut self, lines: &str) -> Result<Script, String> {
    let mut script = Script::default();
    for line in lines.lines() {
      if let Some(step) = read_step(&mut self.prog, line)? {
        self.step(step, &mut script)?;
      }
    }
    Ok(script)
  }

  /// Takes a step of a tick: asserts or retracts the host's fact at once,
  /// or notes the tick's message, or a relation to show after it. Steps
  /// come from lines ([`read_step`]) or from JSON ([`super::host`]), and
  /// every runtime takes them alike.
  pub fn step(&mut self, step: Step, script: &mut Script) -> Result<(), String> {
    match step {
      Step::Assert(rel, tuple) => self.assert_tuple(rel, tuple),
      Step::Retract(rel, tuple) => match self.retract_tuple(rel, &tuple)? {
        true => Ok(()),
        false => Err(format!(
          "{} doesn't hold, so it can't be retracted",
          self.fact_text(rel, &tuple)
        )),
      },
      Step::Msg(_) if script.msg.is_some() => Err("a tick has one message at most".to_string()),
      Step::Msg(msg) => {
        script.msg = Some(msg);
        Ok(())
      }
      Step::Show(rel) if self.prog.rel(&rel).is_none() => {
        Err(format!("there's no relation `{rel}`"))
      }
      Step::Show(rel) => {
        self.asked.extend(self.prog.rel(&rel));
        script.show.push(rel);
        Ok(())
      }
    }
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
  /// what [`Runtime::show`] gives ([`Runtime::run`]).
  pub fn script(&mut self, lines: &str) -> Result<String, String> {
    let script = self.read_script(lines)?;
    self.run(&script)
  }

  /// Runs the tick that a script describes, and gives what
  /// [`Runtime::show`] gives, or `(nothing to show)`.
  pub fn run(&mut self, script: &Script) -> Result<String, String> {
    let effects = self.tick(script.msg.as_ref())?;
    let out = self.show(script, &effects);
    Ok(match out.trim_end() {
      "" => "(nothing to show)".to_string(),
      out => out.to_string(),
    })
  }

  /// One tick: the message's `on` blocks, if there's a message, then the
  /// rules. Gives the effects that the `on` blocks sent, in order.
  pub fn tick(&mut self, msg: Option<&Value>) -> Result<Vec<Effect>, String> {
    self.visits.set(0);
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
        self.tables[rel].remove(&tuple);
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
        let table = &mut self.tables[rel];
        table.drop_where(|t| key.iter().all(|&i| t[i] == tuple[i]));
        table.put(tuple.clone(), Why::Action(a));
        added.insert((rel, k), (tuple, a));
      }
    }
    let ran = self.derive().and_then(|()| self.check_keys());
    self.settle();
    ran?;
    Ok(effects)
  }

  /// The end of a tick that ran the rules, whether or not they failed: each
  /// relation's log begins again.
  fn settle(&mut self) {
    for table in &mut self.tables {
      table.log = Some(Log::default());
    }
  }

  /// What the `on` blocks of a message would add and delete: (relation,
  /// fact, action) each.
  fn update(&self, msg: &Value) -> Result<(New, New), String> {
    let j = Join {
      tables: &self.tables,
      visits: &self.visits,
    };
    let (mut inserts, mut deletes) = (Vec::new(), Vec::new());
    for (a, action) in self.prog.actions.iter().enumerate() {
      let mut env = vec![None; action.vars.len()];
      if !match_pat(&action.msg, msg, &mut env, &mut Vec::new())? {
        continue;
      }
      solve(j, &action.body, &mut env, None, &mut |env| {
        match &action.change {
          Change::Insert(exprs) => {
            let values = exprs
              .iter()
              .map(|e| eval(e, env))
              .collect::<Result<Vec<_>, _>>()?;
            inserts.push((action.rel, values.into(), a));
          }
          Change::Delete(pats) => {
            for tuple in j.tables[action.rel].facts.keys() {
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

  /// Runs the rules from the input and state facts, component by component,
  /// in the checker's order: each only if what it reads has changed. A
  /// component that isn't recursive, and has run since the rules last
  /// failed, follows what changed ([`Runtime::follow`]), unless it starts
  /// over ([`Runtime::starts_over`]); one that fails to follow runs again
  /// whole, as any other does ([`Runtime::rerun`]). A component that the
  /// tick skips has no facts. If a component's rules fail, those after it
  /// have no facts, as when every tick cleared them first, and the next tick
  /// runs every rule.
  fn derive(&mut self) -> Result<(), String> {
    if std::mem::take(&mut self.anew) {
      for (table, rel) in self.tables.iter_mut().zip(&self.prog.rels) {
        if rel.kind == DeclKind::Rel {
          table.clear();
        }
      }
    }
    let asked = std::mem::take(&mut self.asked);
    let skip = match asked.iter().any(|&r| self.prog.rels[r].lazy) {
      true => skipped(&self.prog, &asked),
      false => self.skip.clone(),
    };
    for (c, skip) in skip.into_iter().enumerate() {
      let whole = self.prog.components[c].recursive || self.ran[c].is_none() || self.starts_over(c);
      let ran = match (skip, whole) {
        (true, _) => {
          self.ran[c] = None;
          self.clear(c);
          Ok(())
        }
        (false, true) => self.rerun(c),
        (false, false) => self.follow(c).or_else(|_| {
          self.ran[c] = None;
          self.rerun(c)
        }),
      };
      if let Err(e) = ran {
        self.ran.fill(None);
        for later in c + 1..self.prog.components.len() {
          self.clear(later);
        }
        return Err(e);
      }
    }
    Ok(())
  }

  /// Whether component `c` starts over this tick, running whole as in its
  /// first tick, rather than following what changed ([`Runtime::follow`]):
  /// when its relations have no facts to keep, and more of what it reads
  /// came or went since the last tick than stayed. Following such a change
  /// would join about as much as a whole run does, and keep more for each
  /// fact, where a whole run fills new tables with nothing to compare: it's
  /// semi-naive evaluation's first round, where every fact is new, which is
  /// naive. Both runtimes decide alike, from the logs and the sizes of the
  /// tables, so that their joins visit the same facts.
  fn starts_over(&self, c: usize) -> bool {
    let rels = &self.prog.components[c].rels;
    if !rels.iter().all(|&r| self.tables[r].facts.is_empty()) {
      return false;
    }
    let (mut changed, mut stayed) = (0, 0);
    for &r in &self.reads[c] {
      let log = self.log(r);
      changed += log.came.len + log.went.len;
      stayed += self.tables[r].facts.len() - log.came.len;
    }
    changed > stayed
  }

  /// Clears component `c`'s relations: those that have facts, so that one
  /// without keeps its version ([`Table::clear`]).
  fn clear(&mut self, c: usize) {
    for &r in &self.prog.components[c].rels {
      self.tables[r].clear();
    }
  }

  /// Runs component `c`'s rules, unless the relations that they read have
  /// the versions that they had when it last ran. Each of its relations gets
  /// a new table, a version on from the old one; but a relation whose facts
  /// come out the same keeps the old table, with its version and indexes,
  /// and why each fact holds now. A new table takes the old one's log, and
  /// notes there what came and went: what came in the order that the rules
  /// first derived it ([`Runtime::component`]).
  fn rerun(&mut self, c: usize) -> Result<(), String> {
    let read: Vec<u64> = self.reads[c]
      .iter()
      .map(|&r| self.tables[r].version)
      .collect();
    if self.ran[c].as_ref() == Some(&read) {
      return Ok(());
    }
    let olds: Vec<Table> = (self.prog.components[c].rels.iter())
      .map(|&r| {
        let version = self.tables[r].version + 1;
        let new = Table {
          version,
          ..Table::default()
        };
        std::mem::replace(&mut self.tables[r], new)
      })
      .collect();
    let derived = self.component(c)?;
    for (&r, mut old) in self.prog.components[c].rels.iter().zip(olds) {
      let new = &mut self.tables[r];
      if new.facts.len() == old.facts.len() && new.facts.keys().eq(old.facts.keys()) {
        for (why, &now) in old.facts.values_mut().zip(new.facts.values()) {
          *why = now;
        }
        self.tables[r] = old;
        continue;
      }
      let mut log = old.log.take();
      if let Some(log) = &mut log {
        for tuple in old.facts.keys().filter(|t| !new.facts.contains_key(*t)) {
          log.note(tuple, false);
        }
        let order = derived.get(&r).into_iter().flatten();
        for tuple in order.filter(|t| !old.facts.contains_key(*t)) {
          log.note(tuple, true);
        }
      }
      new.log = log;
    }
    self.ran[c] = Some(read);
    Ok(())
  }

  /// Runs a component's rules: once on every fact; then, if it's recursive,
  /// again for each of their atoms on its relations, joining just what the
  /// last round added there, until a round adds nothing. Of one that isn't,
  /// it counts how many ways each rule derives each fact
  /// ([`Runtime::follow`]). Gives the facts that it derived, by relation, in
  /// the order that it first derived them.
  fn component(&mut self, c: usize) -> Result<BTreeMap<usize, Vec<Tuple>>, String> {
    let prog = &self.prog;
    let component = &prog.components[c];
    let mut new = Vec::new();
    let j = Join {
      tables: &self.tables,
      visits: &self.visits,
    };
    for &r in &component.rules {
      solutions(prog, j, r, None, &mut new)?;
    }
    if !component.recursive {
      for &r in &component.rules {
        self.counts[r].clear();
      }
      for (_, tuple, r) in &new {
        *self.counts[*r].entry(tuple.clone()).or_default() += 1;
      }
    }
    let mut derived: BTreeMap<usize, Vec<Tuple>> = BTreeMap::new();
    let mut fresh = insert(&mut self.tables, new);
    while component.recursive && !fresh.is_empty() {
      for (&rel, tuples) in &fresh {
        derived
          .entry(rel)
          .or_default()
          .extend(tuples.iter().cloned());
      }
      let added: BTreeMap<usize, Fresh> = fresh
        .into_iter()
        .map(|(rel, tuples)| {
          let indexes = Indexes::default();
          (rel, Fresh { tuples, indexes })
        })
        .collect();
      let mut new = Vec::new();
      let j = Join {
        tables: &self.tables,
        visits: &self.visits,
      };
      for &r in &component.rules {
        for (at, lit) in prog.rules[r].body.iter().enumerate() {
          let Lit::Atom(atom) = lit else { continue };
          if let Some(added) = added.get(&atom.rel) {
            solutions(prog, j, r, Some((at, added)), &mut new)?;
          }
        }
      }
      fresh = insert(&mut self.tables, new);
    }
    for (rel, tuples) in fresh {
      derived.entry(rel).or_default().extend(tuples);
    }
    Ok(derived)
  }

  /// Follows what changed, since component `c` last ran, in what its rules
  /// read, unless that has the versions that it had then: in place, by how
  /// many ways each rule derives each fact (`counts`). A rule that reads
  /// one relation that changed, at an atom that it follows ([`followed`]),
  /// joins there just the facts that came, and those that went, with the
  /// rest of its body as it is; one that reads more that changed, or reads
  /// one in a negation, or at an atom that it doesn't follow, runs whole.
  /// Every rule joins before any count changes. A fact whose counts change
  /// holds if a rule derives it, by the lowest-numbered of them
  /// ([`Runtime::place`]).
  fn follow(&mut self, c: usize) -> Result<(), String> {
    let read: Vec<u64> = (self.reads[c].iter())
      .map(|&r| self.tables[r].version)
      .collect();
    if self.ran[c].as_ref() == Some(&read) {
      return Ok(());
    }
    let prog = &self.prog;
    let j = Join {
      tables: &self.tables,
      visits: &self.visits,
    };
    // Each rule's change: its counts anew (whole), or by how many ways more,
    // or fewer, it derives each fact.
    let mut changes = Vec::new();
    for &r in &prog.components[c].rules {
      let mut found = Vec::new();
      let change = match self.plan(r) {
        Plan::Nothing => continue,
        Plan::Whole => {
          solutions(prog, j, r, None, &mut found)?;
          (true, tallied(found, 1, Ordered::default()))
        }
        Plan::At(at) => {
          let Lit::Atom(atom) = &prog.rules[r].body[at] else {
            unreachable!("a rule joins what came and went at an atom");
          };
          let log = self.log(atom.rel);
          let mut by = Ordered::default();
          for (facts, sign) in [(&log.came, 1), (&log.went, -1)] {
            if !facts.is_empty() {
              let tuples = facts.iter().map(|(tuple, _)| tuple.clone()).collect();
              let fresh = Fresh {
                tuples,
                indexes: Indexes::default(),
              };
              solutions(prog, j, r, Some((at, &fresh)), &mut found)?;
              by = tallied(std::mem::take(&mut found), sign, by);
            }
          }
          (false, by)
        }
      };
      changes.push((r, change));
    }
    // The facts whose counts change, by relation, in the order that the
    // rules first gave them; and the counts.
    let mut touched: BTreeMap<usize, Ordered<()>> = BTreeMap::new();
    for (r, (whole, change)) in changes {
      let heads = touched.entry(self.prog.rules[r].head).or_default();
      let counts = &mut self.counts[r];
      if whole {
        counts
          .keys()
          .for_each(|tuple| heads.insert(tuple.clone(), ()));
        change
          .iter()
          .for_each(|(tuple, _)| heads.insert(tuple.clone(), ()));
        *counts = (change.iter())
          .map(|(tuple, n)| (tuple.clone(), n.unsigned_abs()))
          .collect();
        continue;
      }
      for (tuple, &by) in change.iter() {
        let n = counts.get(tuple).copied().unwrap_or(0);
        let now = n.checked_add_signed(by);
        let now = now.expect("a rule derives a fact no fewer than no ways");
        heads.insert(tuple.clone(), ());
        match now {
          0 => _ = counts.remove(tuple),
          _ => _ = counts.insert(tuple.clone(), now),
        }
      }
    }
    for (rel, heads) in touched {
      self.place(c, rel, heads);
    }
    self.ran[c] = Some(read);
    Ok(())
  }

  /// What a tick does for rule `r`, by the relations in its body whose logs
  /// aren't empty ([`Plan`]).
  fn plan(&self, r: usize) -> Plan {
    let (mut at, mut changes) = (0, 0);
    for (i, lit) in self.prog.rules[r].body.iter().enumerate() {
      let (Lit::Atom(atom) | Lit::Not(atom)) = lit else {
        continue;
      };
      if self.log(atom.rel).changed() {
        changes += match lit {
          Lit::Atom(_) if self.followed[r][i] => 1,
          _ => 2,
        };
        at = i;
      }
    }
    match changes {
      0 => Plan::Nothing,
      1 => Plan::At(at),
      _ => Plan::Whole,
    }
  }

  /// A relation's log, which a tick that follows what changed has: rules
  /// follow only after a tick has run them ([`Runtime::settle`]).
  fn log(&self, rel: usize) -> &Log {
    let log = self.tables[rel].log.as_ref();
    log.expect("a relation has a log once a tick has run the rules")
  }

  /// Brings relation `rel`, of component `c`, up to its rules' counts, at
  /// the facts `heads`, in their order: a fact holds if a rule derives it,
  /// by the lowest-numbered of them.
  fn place(&mut self, c: usize, rel: usize, heads: Ordered<()>) {
    let rules: Vec<usize> = (self.prog.components[c].rules.iter())
      .copied()
      .filter(|&r| self.prog.rules[r].head == rel)
      .collect();
    let table = &mut self.tables[rel];
    let mut changed = false;
    for (tuple, _) in heads.iter() {
      let by = rules.iter().find(|&&r| self.counts[r].contains_key(tuple));
      let held = table.facts.contains_key(tuple);
      match by {
        Some(&r) if held => _ = table.facts.insert(tuple.clone(), Why::Rule(r)),
        Some(&r) => {
          table.add(tuple.clone(), Why::Rule(r));
          changed = true;
        }
        None if held => {
          table.cut(tuple);
          changed = true;
        }
        None => {}
      }
    }
    if changed {
      table.changed();
    }
  }

  /// Where a fact comes from, for errors.
  pub fn why(&self, why: Why) -> String {
    match why {
      Why::Host => "from the host".to_string(),
      Why::Action(a) => format!("from the `on` block at {}", self.prog.actions[a].place),
      Why::Rule(r) => format!("from the rule at {}", self.prog.rules[r].place),
    }
  }

  /// A fact as lem writes it, its values by position.
  pub fn fact_text(&self, rel: usize, tuple: &Tuple) -> String {
    Value::Ctor(self.prog.rels[rel].name.clone(), tuple.clone()).to_string()
  }

  /// Facts that agree on their relation's key agree on the rest: checked
  /// where the relation's version has moved since its keys last held, by
  /// how many of its key's values more than one fact has
  /// ([`Table::key_count`]). Of a relation whose keys don't hold, the error
  /// names the first two facts that clash ([`Runtime::key_clash`]); if none
  /// do, the count that said so is wrong.
  fn check_keys(&mut self) -> Result<(), String> {
    for (rel, r) in self.prog.rels.iter().enumerate() {
      let version = self.tables[rel].version;
      if r.key.is_empty() || self.held[rel] == Some(version) {
        continue;
      }
      if self.tables[rel].key_count(&r.key).clashes > 0 {
        let miscounted = || {
          let name = &self.prog.rels[rel].name;
          format!("the facts of {name} by their key were miscounted (a bug in lem's runtime)")
        };
        return Err(self.key_clash(rel).unwrap_or_else(miscounted));
      }
      self.held[rel] = Some(version);
    }
    Ok(())
  }

  /// The first two of a relation's facts, in order, that agree on its key
  /// and not on the rest, as an error; or none.
  fn key_clash(&self, rel: usize) -> Option<String> {
    let key = &self.prog.rels[rel].key;
    let mut seen: BTreeMap<Vec<Value>, (&Tuple, Why)> = BTreeMap::new();
    for (tuple, why) in &self.tables[rel].facts {
      let k = project(tuple, key);
      if let Some((other, first)) = seen.get(&k) {
        let (x, y) = (self.fact_text(rel, other), self.fact_text(rel, tuple));
        let (at, other) = (self.why(*first), self.why(*why));
        return Some(format!("one key, two facts: {x} ({at}) and {y} ({other})"));
      }
      seen.insert(k, (tuple, *why));
    }
    None
  }
}

/// A tick written as lines ([`Runtime::read_script`]): its message, and the
/// relations whose facts to show after it.
#[derive(Clone, Debug, Default)]
pub struct Script {
  pub msg: Option<Value>,
  pub show: Vec<String>,
}

/// A step of a tick, as a line or JSON gives it: the host asserts or
/// retracts a fact, the tick's message, or a relation whose facts to show
/// after the tick ([`Runtime::step`]).
#[derive(Clone, Debug)]
pub enum Step {
  Assert(usize, Tuple),
  Retract(usize, Tuple),
  Msg(Value),
  Show(String),
}

/// Reads a line of a tick: `+ Fact` asserts the host's fact, `- Fact`
/// retracts it, `! Msg` is the tick's message, and `? Name` asks for a
/// relation's facts after the tick. A blank line is no step.
pub fn read_step(prog: &mut Program, line: &str) -> Result<Option<Step>, String> {
  let step = if let Some(fact) = line.strip_prefix("+ ") {
    let (rel, values) = prog.fact(fact)?;
    Step::Assert(rel, values.into())
  } else if let Some(fact) = line.strip_prefix("- ") {
    let (rel, values) = prog.fact(fact)?;
    Step::Retract(rel, values.into())
  } else if let Some(text) = line.strip_prefix("! ") {
    Step::Msg(prog.value(text, &Ty::Named("Msg".into()))?)
  } else if let Some(rel) = line.strip_prefix("? ") {
    Step::Show(rel.trim().to_string())
  } else if line.trim().is_empty() {
    return Ok(None);
  } else {
    return Err(format!(
      "a tick's lines start with `+ `, `- `, `! ` or `? `, not {line:?}"
    ));
  };
  Ok(Some(step))
}

/// Adds rules' new facts: by relation, those that didn't hold before.
fn insert(tables: &mut [Table], new: New) -> BTreeMap<usize, Vec<Tuple>> {
  let mut fresh: BTreeMap<usize, Vec<Tuple>> = BTreeMap::new();
  for (rel, tuple, r) in new {
    if tables[rel].derive(&tuple, r) {
      fresh.entry(rel).or_default().push(tuple);
    }
  }
  fresh
}

/// Counts a fact `by` more ways, or fewer, in `counts`: one that's counted
/// no ways goes.
fn tally(counts: &mut Ordered<i64>, tuple: &Tuple, by: i64) {
  let n = counts.get_mut(tuple).map(|n| {
    *n += by;
    *n
  });
  match n {
    None => counts.insert(tuple.clone(), by),
    Some(0) => _ = counts.remove(tuple),
    Some(_) => {}
  }
}

/// Counts the facts of rules' solutions, `found`, `by` more ways each, or
/// fewer, in `counts`.
fn tallied(found: New, by: i64, mut counts: Ordered<i64>) -> Ordered<i64> {
  for (_, tuple, _) in &found {
    tally(&mut counts, tuple, by);
  }
  counts
}

/// The variables that a pattern binds, into `out`.
fn pat_vars(pat: &Pat, out: &mut Vec<usize>) {
  match pat {
    Pat::Var(v) => out.push(*v),
    Pat::Ctor(_, pats) => pats.iter().for_each(|pat| pat_vars(pat, out)),
    Pat::Any | Pat::Const(_) | Pat::Expr(_) => {}
  }
}

/// Whether a pattern has `_`.
fn loose(pat: &Pat) -> bool {
  match pat {
    Pat::Any => true,
    Pat::Ctor(_, pats) => pats.iter().any(loose),
    Pat::Var(_) | Pat::Const(_) | Pat::Expr(_) => false,
  }
}

/// For each of a rule's atoms, whether a tick follows a change to its
/// relation alone by joining there just the facts that came and went
/// ([`Runtime::follow`]). A rule's solutions must count the same whatever
/// order its join takes: a solution counts each fact of an atom with `_`
/// that binds a variable that no other atom binds, and once an atom without
/// `_`, which matches one fact at most; an atom with `_` and no variable is
/// a test, which counts once, so a change there isn't followed. But an atom
/// with `_` whose variables other atoms bind counts one fact or each, by the
/// order of the join: then no change to the rule is followed.
fn followed(rule: &Rule) -> Vec<bool> {
  let atoms: Vec<&Atom> = (rule.body.iter())
    .map_while(|lit| match lit {
      Lit::Atom(atom) => Some(atom),
      _ => None,
    })
    .collect();
  let vars: Vec<Vec<usize>> = (atoms.iter())
    .map(|atom| {
      let mut out = Vec::new();
      atom.pats.iter().for_each(|pat| pat_vars(pat, &mut out));
      out
    })
    .collect();
  let own = |i: usize| {
    let alone = |v: &usize| (vars.iter().enumerate()).all(|(j, vs)| j == i || !vs.contains(v));
    vars[i].iter().any(alone)
  };
  let ok: Vec<bool> = (atoms.iter().enumerate())
    .map(|(i, atom)| !atom.pats.iter().any(loose) || own(i))
    .collect();
  let counts = (ok.iter().zip(&vars)).all(|(&yes, vs)| yes || vs.is_empty());
  ok.into_iter().map(|yes| counts && yes).collect()
}

/// What each component's rules read, as atoms or negations: the relations of
/// other components, the host's and the `on` blocks', in order.
fn reads(prog: &Program) -> Vec<Vec<usize>> {
  let comps = prog.components.iter();
  comps
    .map(|c| {
      let mut read: Vec<usize> = (c.rules.iter())
        .flat_map(|&r| &prog.rules[r].body)
        .filter_map(|lit| match lit {
          Lit::Atom(atom) | Lit::Not(atom) => Some(atom.rel),
          _ => None,
        })
        .filter(|rel| !c.rels.contains(rel))
        .collect();
      read.sort_unstable();
      read.dedup();
      read
    })
    .collect()
}

/// The components that a tick skips: those whose relations are all lazy,
/// and that nothing derived reads: no rule of a relation that's derived, no
/// `on` block, and not the host, which asks for the relations `asked`.
/// Their relations have no facts.
fn skipped(prog: &Program, asked: &[usize]) -> Vec<bool> {
  let mut read: Vec<bool> = prog.rels.iter().map(|rel| !rel.lazy).collect();
  asked.iter().for_each(|&r| read[r] = true);
  let mut todo: Vec<usize> = (0..prog.rels.len()).filter(|&r| read[r]).collect();
  let reads = |body: &[Lit], read: &mut Vec<bool>, todo: &mut Vec<usize>| {
    for lit in body {
      if let Lit::Atom(atom) | Lit::Not(atom) = lit {
        if !read[atom.rel] {
          read[atom.rel] = true;
          todo.push(atom.rel);
        }
      }
    }
  };
  for action in &prog.actions {
    reads(&action.body, &mut read, &mut todo);
  }
  while let Some(r) = todo.pop() {
    for rule in prog.rules.iter().filter(|rule| rule.head == r) {
      reads(&rule.body, &mut read, &mut todo);
    }
  }
  let comps = prog.components.iter();
  comps.map(|c| c.rels.iter().all(|&r| !read[r])).collect()
}

/// Adds rule `r`'s solutions to `new`, as (relation, fact, rule).
fn solutions(prog: &Program, j: Join, r: usize, delta: Delta, new: &mut New) -> Result<(), String> {
  let rule = &prog.rules[r];
  let mut env = vec![None; rule.vars.len()];
  solve(j, &rule.body, &mut env, delta, &mut |env| {
    let values = rule
      .args
      .iter()
      .map(|e| eval(e, env))
      .collect::<Result<Vec<_>, _>>();
    let values = values.map_err(|e| format!("{e}, in the rule at {}", rule.place))?;
    new.push((rule.head, Tuple::from(values), r));
    Ok(())
  })
}

/// Calls `out` with each solution of a body, as the variables' values: its
/// positive atoms, which come first, joined as [`join`] does; then the rest
/// in order.
fn solve(
  j: Join,
  body: &[Lit],
  env: &mut Env,
  delta: Delta,
  out: &mut dyn FnMut(&Env) -> Result<(), String>,
) -> Result<(), String> {
  let atoms = body
    .iter()
    .take_while(|lit| matches!(lit, Lit::Atom(_)))
    .count();
  assert!(atoms <= MAX_ATOMS, "the checker limits a body's atoms");
  // A bit for each atom: all of them are left to join.
  let left = match atoms {
    0 => 0,
    n => u64::MAX >> (MAX_ATOMS - n),
  };
  join(j, body, atoms, left, env, delta, out)
}

/// Joins the first `atoms` parts of a body, the atoms, whose bits are set in
/// `left`: of those that are ready, it looks each up by the columns bound so
/// far; fails if one has no facts; and else takes the one with the fewest,
/// the first written on a tie. An atom that binds no variable is a test,
/// which it takes as soon as it's ready: the join goes on once if a fact
/// matches it, rather than once for each fact. Then the rest of the body, in
/// order.
fn join(
  j: Join,
  body: &[Lit],
  atoms: usize,
  left: u64,
  env: &mut Env,
  delta: Delta,
  out: &mut dyn FnMut(&Env) -> Result<(), String>,
) -> Result<(), String> {
  if left == 0 {
    return rest(j, body, atoms, env, out);
  }
  let alone = left.count_ones() == 1;
  let mut best: Option<(usize, usize, Bucket)> = None;
  for at in (0..atoms).filter(|at| left & (1 << at) != 0) {
    let Lit::Atom(atom) = &body[at] else {
      unreachable!("a body's atoms come first");
    };
    if !atom.needs.iter().all(|&v| env[v].is_some()) {
      continue;
    }
    let cols: Vec<usize> = (0..atom.pats.len())
      .filter(|&c| known(&atom.pats[c], env))
      .collect();
    let fresh = delta.filter(|(place, _)| *place == at).map(|(_, f)| f);
    let bucket = bucket(j, atom, &cols, env, fresh)?;
    if !atom.pats.iter().any(|pat| binds(pat, env)) {
      if !matches_one(j, atom, &bucket, env)? {
        return Ok(());
      }
      return join(j, body, atoms, left & !(1 << at), env, delta, out);
    }
    if alone {
      best = Some((0, at, bucket));
      break;
    }
    let len = bucket.len();
    if len == 0 {
      return Ok(());
    }
    match &best {
      Some((least, ..)) if *least <= len => {}
      _ => best = Some((len, at, bucket)),
    }
  }
  let (_, at, bucket) = best.expect("the first atom left is ready");
  let Lit::Atom(atom) = &body[at] else {
    unreachable!("a body's atoms come first");
  };
  for tuple in bucket.tuples() {
    j.visits.set(j.visits.get() + 1);
    let mut trail = Vec::new();
    if match_all(&atom.pats, tuple, env, &mut trail)? {
      join(j, body, atoms, left & !(1 << at), env, delta, out)?;
    }
    trail.iter().for_each(|&v| env[v] = None);
  }
  Ok(())
}

/// The facts that might match an atom, by the columns `cols` whose values
/// are known: from what a round added, if `fresh` is this atom's, else from
/// its relation.
fn bucket<'a>(
  j: Join<'a>,
  atom: &Atom,
  cols: &[usize],
  env: &Env,
  fresh: Option<&'a Fresh>,
) -> Result<Bucket<'a>, String> {
  if cols.is_empty() {
    return Ok(match fresh {
      Some(fresh) => Bucket::Fresh(&fresh.tuples),
      None => Bucket::Table(&j.tables[atom.rel].facts),
    });
  }
  let key = cols
    .iter()
    .map(|&c| value_of(&atom.pats[c], env))
    .collect::<Result<Vec<_>, _>>()?;
  let index = match fresh {
    Some(fresh) => index(&fresh.indexes, cols, fresh.tuples.iter()),
    None => j.tables[atom.rel].index(cols),
  };
  Ok(Bucket::Index(index, key))
}

/// Whether a pattern binds a variable: one that isn't bound yet.
fn binds(pat: &Pat, env: &Env) -> bool {
  match pat {
    Pat::Var(v) => env[*v].is_none(),
    Pat::Ctor(_, pats) => pats.iter().any(|pat| binds(pat, env)),
    Pat::Any | Pat::Const(_) | Pat::Expr(_) => false,
  }
}

/// Whether a fact in a bucket matches an atom that binds no variable, or a
/// negation's, visiting them in order, up to the first that does.
fn matches_one(j: Join, atom: &Atom, bucket: &Bucket, env: &mut Env) -> Result<bool, String> {
  for tuple in bucket.tuples() {
    j.visits.set(j.visits.get() + 1);
    let mut trail = Vec::new();
    let matched = match_all(&atom.pats, tuple, env, &mut trail)?;
    trail.iter().for_each(|&v| env[v] = None);
    if matched {
      return Ok(true);
    }
  }
  Ok(false)
}

/// Whether a pattern's value is known, from the variables bound so far: in
/// an atom that's ready, an expression's is.
fn known(pat: &Pat, env: &Env) -> bool {
  match pat {
    Pat::Any => false,
    Pat::Var(v) => env[*v].is_some(),
    Pat::Const(_) | Pat::Expr(_) => true,
    Pat::Ctor(_, pats) => pats.iter().all(|pat| known(pat, env)),
  }
}

/// The rest of a body, after its atoms, in order: bindings, tests and
/// negations, each looked up by its key.
fn rest(
  j: Join,
  body: &[Lit],
  at: usize,
  env: &mut Env,
  out: &mut dyn FnMut(&Env) -> Result<(), String>,
) -> Result<(), String> {
  let Some(lit) = body.get(at) else {
    return out(env);
  };
  match lit {
    Lit::Atom(_) => unreachable!("a body's atoms come first"),
    Lit::Let(v, e) => {
      env[*v] = Some(eval(e, env)?);
      let result = rest(j, body, at + 1, env, out);
      env[*v] = None;
      result
    }
    Lit::Test(t) if test(t, env)? => rest(j, body, at + 1, env, out),
    Lit::NotTest(t) if !test(t, env)? => rest(j, body, at + 1, env, out),
    Lit::Test(_) | Lit::NotTest(_) => Ok(()),
    Lit::Not(atom) => {
      let bucket = bucket(j, atom, &atom.key, env, None)?;
      match matches_one(j, atom, &bucket, env)? {
        true => Ok(()),
        false => rest(j, body, at + 1, env, out),
      }
    }
  }
}

/// A pattern's value, when it's known: a key column's.
fn value_of(pat: &Pat, env: &Env) -> Result<Value, String> {
  Ok(match pat {
    Pat::Const(value) => value.clone(),
    Pat::Var(v) => env[*v]
      .clone()
      .ok_or("a key's variable isn't bound (a bug in lem's checker)")?,
    Pat::Expr(e) => eval(e, env)?,
    Pat::Ctor(name, pats) => {
      let args = pats
        .iter()
        .map(|pat| value_of(pat, env))
        .collect::<Result<Vec<_>, _>>()?;
      Value::Ctor(name.clone(), args.into())
    }
    Pat::Any => return Err("`_` can't be in a key (a bug in lem's checker)".to_string()),
  })
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
    Expr::Arith(op, _, a, b) => arith(*op, eval(a, env)?, eval(b, env)?)?,
    Expr::Text(_, e) => Value::Text(eval(e, env)?.text()),
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
