//! lem's checker: from parsed files to a [`Program`], which [`super::eval`]
//! runs.
//!
//! It loads lem's prelude first, and reads every file's declarations before
//! their rules, so that files may come in any order. Then it lowers each
//! rule to a flat one: everything inside a `for` gets its body. On the way
//! it checks names, fields and types, that an atom binds every variable
//! (safety), and that no relation depends on its own negation (strata). lem
//! itself gives blocks (`Head:`) no meaning: a layer above it may rewrite
//! them before this checks the files ([`check`]).
//!
//! Types are simple: a variable's is that of the first column it's in, and a
//! literal fits the types made from its own, as `"job_1"` fits `JobId`.
//! An Int widens to Num, and a field of type `Any` takes any value.

use super::parse::{parse, Cmp, Decl, Mod, Param, Sent, SentRef, SpanIdent, Term, TermRef};
use super::token::Span;
use super::value::{Ty, Value};
use std::collections::HashMap;
use std::panic::Location;
use std::rc::Rc;

/// What a declaration sentence declares, by its word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclKind {
  /// Relations that the host asserts and retracts.
  Input,
  /// Relations that only `on` blocks change, and that persist.
  State,
  /// Relations that rules derive, each tick.
  Rel,
  /// Relations that only `on` blocks add to, whose facts go to the host or
  /// the runtime, such as `Send(msg)`.
  Effect,
  /// Constructors of the program's `Id` type: names for things.
  Id,
  /// Constructors of the program's `Msg` type: what `on` blocks handle.
  Msg,
}

impl DeclKind {
  pub fn word(self) -> &'static str {
    match self {
      DeclKind::Input => "input",
      DeclKind::State => "state",
      DeclKind::Rel => "rel",
      DeclKind::Effect => "effect",
      DeclKind::Id => "id",
      DeclKind::Msg => "msg",
    }
  }

  pub fn from_word(word: &str) -> Option<DeclKind> {
    Some(match word {
      "input" => DeclKind::Input,
      "state" => DeclKind::State,
      "rel" => DeclKind::Rel,
      "effect" => DeclKind::Effect,
      "id" => DeclKind::Id,
      "msg" => DeclKind::Msg,
      _ => return None,
    })
  }
}

/// Where something is written: a file, by its place in the load, and a span.
#[derive(Clone, Debug)]
pub struct At {
  pub file: usize,
  pub span: Span,
}

/// An error, as lem shows it ([`super::render`]), and the checker's own line
/// that raised it.
#[derive(Clone, Debug)]
pub struct LoadError {
  pub text: String,
  pub loc: &'static Location<'static>,
}

/// An error before it's rendered.
struct CheckError {
  at: At,
  message: String,
  loc: &'static Location<'static>,
}

type Check<T> = Result<T, CheckError>;

#[derive(Clone, Debug)]
pub struct Field {
  /// None for a constructor's field that's only a type.
  pub name: Option<Rc<str>>,
  pub ty: Ty,
  pub default: Option<Value>,
}

/// A relation, as `input`, `state`, `rel` or `effect` declares it.
#[derive(Debug)]
pub struct Rel {
  pub name: Rc<str>,
  pub kind: DeclKind,
  pub fields: Vec<Field>,
  /// The key's columns: facts that agree on them must agree on the rest.
  pub key: Vec<usize>,
  pub at: At,
}

/// A constructor of an enum, of `Id` or of `Msg`.
#[derive(Debug)]
pub struct Ctor {
  pub name: Rc<str>,
  /// The type that it makes.
  pub ty: Rc<str>,
  pub fields: Vec<Field>,
  pub at: At,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
  Add,
  Sub,
  Mul,
  Div,
}

/// A value to compute, once the body has bound its variables.
#[derive(Clone, Debug)]
pub enum Expr {
  Var(usize),
  Const(Value),
  Ctor(Rc<str>, Vec<Expr>),
  Neg(Box<Expr>),
  Arith(Op, Box<Expr>, Box<Expr>),
  /// An Int, widened to Num.
  ToNum(Box<Expr>),
}

/// What a column, or a constructor's field, matches.
#[derive(Clone, Debug)]
pub enum Pat {
  Any,
  /// Binds the variable, or matches its value once it's bound.
  Var(usize),
  Const(Value),
  Ctor(Rc<str>, Vec<Pat>),
  /// A value computed from variables that earlier atoms bound.
  Expr(Expr),
}

#[derive(Clone, Debug)]
pub enum Test {
  Eq(Expr, Expr),
  Ne(Expr, Expr),
  Cmp(Cmp, Expr, Expr),
}

/// A part of a body, in the order that they run: atoms first, as written;
/// then bindings by `=`; then tests and negations, whose variables are
/// bound by then.
#[derive(Clone, Debug)]
pub enum Lit {
  Atom(usize, Vec<Pat>),
  /// `x = e`, where only `e` is bound: binds `x`.
  Let(usize, Expr),
  Test(Test),
  Not(usize, Vec<Pat>),
  NotTest(Test),
}

/// A rule, lowered: its head's relation, a value per column, and its body.
#[derive(Debug)]
pub struct Rule {
  pub at: At,
  pub head: usize,
  pub args: Vec<Expr>,
  pub body: Vec<Lit>,
  /// Its variables' names, by number.
  pub vars: Vec<Rc<str>>,
}

/// What a sentence of an `on` block does to its relation.
#[derive(Debug)]
pub enum Change {
  Insert(Vec<Expr>),
  Delete(Vec<Pat>),
}

/// A sentence of an `on` block: for a message that `msg` matches, and each
/// solution of the body, a change.
#[derive(Debug)]
pub struct Action {
  pub at: At,
  pub msg: Pat,
  pub body: Vec<Lit>,
  pub rel: usize,
  pub change: Change,
  pub vars: Vec<Rc<str>>,
}

#[derive(Clone, Copy, Debug)]
enum Name {
  Rel(usize),
  Ctor(usize),
}

/// A checked program.
#[derive(Debug)]
pub struct Program {
  /// The files, by name and text, for errors and for why a fact holds.
  pub files: Vec<(Rc<str>, Rc<str>)>,
  pub rels: Vec<Rel>,
  pub ctors: Vec<Ctor>,
  /// Relations and constructors, which share one namespace.
  names: HashMap<Rc<str>, Name>,
  /// Types by name: a type made with `type` has its base; an enum, None.
  types: HashMap<Rc<str>, Option<Ty>>,
  pub rules: Vec<Rule>,
  /// The rules, a list per stratum, in the order to run them.
  pub strata: Vec<Vec<usize>>,
  pub actions: Vec<Action>,
}

impl Program {
  pub fn rel(&self, name: &str) -> Option<usize> {
    match self.names.get(name) {
      Some(Name::Rel(i)) => Some(*i),
      _ => None,
    }
  }

  pub fn ctor(&self, name: &str) -> Option<&Ctor> {
    match self.names.get(name) {
      Some(Name::Ctor(i)) => Some(&self.ctors[*i]),
      _ => None,
    }
  }

  /// The type that a type made with `type` is made from, through any chain;
  /// any other type itself.
  pub fn base(&self, ty: &Ty) -> Ty {
    let mut ty = ty.clone();
    for _ in 0..64 {
      match &ty {
        Ty::Named(name) => match self.types.get(name) {
          Some(Some(base)) => ty = base.clone(),
          _ => break,
        },
        _ => break,
      }
    }
    ty
  }

  /// `file:line:col`.
  pub fn place(&self, at: &At) -> String {
    let (name, text) = &self.files[at.file];
    let (line, col, _) = super::line_col(text, at.span.start);
    format!("{name}:{line}:{col}")
  }

  /// A fact written in lem, as the host or a test gives it, such as
  /// `Job("job_1", "review", 0)`: its relation and values.
  pub fn fact(&mut self, text: &str) -> Result<(usize, Vec<Value>), String> {
    let term = one_term(text)?;
    let mut loader = Loader {
      prog: self,
      file: 0,
    };
    let result = match loader.head(&mut Scope::default(), &term) {
      Ok((rel, exprs)) => match exprs.iter().map(fold).collect() {
        Some(values) => Ok((rel, values)),
        None => loader.fail(term.span(), "a fact's values are literals and constructors"),
      },
      Err(e) => Err(e),
    };
    result.map_err(|e| super::render("fact", text, &e.at.span, &e.message))
  }

  /// A value written in lem, of type `ty`, such as the message
  /// `Finish("job_1")`.
  pub fn value(&mut self, text: &str, ty: &Ty) -> Result<Value, String> {
    let term = one_term(text)?;
    let mut loader = Loader {
      prog: self,
      file: 0,
    };
    let result = match loader.expr_fit(&mut Scope::default(), &term, ty, "this value") {
      Ok(expr) => match fold(&expr) {
        Some(value) => Ok(value),
        None => loader.fail(term.span(), "a value here is a literal or a constructor"),
      },
      Err(e) => Err(e),
    };
    result.map_err(|e| super::render("value", text, &e.at.span, &e.message))
  }
}

/// The one term that `text` holds.
fn one_term(text: &str) -> Result<TermRef, String> {
  let mod_ = parse(text).map_err(|e| e.render("value", text))?;
  match mod_.body.as_slice() {
    [sent] => match &**sent {
      Sent::Just(_, term) => Ok(term.clone()),
      _ => Err(format!("expected one term, not {text:?}")),
    },
    _ => Err(format!("expected one term, not {text:?}")),
  }
}

/// Loads files, as (name, text) pairs: parses each, then checks them
/// together.
pub fn load(files: &[(&str, &str)]) -> Result<Program, LoadError> {
  check(files, &parse_all(files)?)
}

/// Parses each of the files, given as (name, text) pairs.
pub fn parse_all(files: &[(&str, &str)]) -> Result<Vec<Mod>, LoadError> {
  let parse_one = |(name, text): &(&str, &str)| {
    parse(text).map_err(|e| LoadError {
      text: e.render(name, text),
      loc: e.loc,
    })
  };
  files.iter().map(parse_one).collect()
}

/// Checks parsed files together: `mods[i]` is the tree of `files[i]`, whose
/// name and text errors show. A layer above lem may rewrite the trees first;
/// as long as it keeps their spans, errors point into the files as written.
pub fn check(files: &[(&str, &str)], mods: &[Mod]) -> Result<Program, LoadError> {
  assert_eq!(files.len(), mods.len(), "a tree for each file");
  let prelude = parse(super::PRELUDE.1).expect("lem's prelude parses");
  let files: Vec<(&str, &str)> = std::iter::once(super::PRELUDE)
    .chain(files.iter().copied())
    .collect();
  let mods: Vec<&Mod> = std::iter::once(&prelude).chain(mods).collect();
  let mut types = HashMap::new();
  types.insert(Rc::from("Id"), None);
  types.insert(Rc::from("Msg"), None);
  let mut prog = Program {
    files: files
      .iter()
      .map(|(name, text)| (Rc::from(*name), Rc::from(*text)))
      .collect(),
    rels: Vec::new(),
    ctors: Vec::new(),
    names: HashMap::new(),
    types,
    rules: Vec::new(),
    strata: Vec::new(),
    actions: Vec::new(),
  };
  let mut loader = Loader {
    prog: &mut prog,
    file: 0,
  };
  match loader.load(&mods) {
    Ok(()) => Ok(prog),
    Err(e) => {
      let (name, text) = &loader.prog.files[e.at.file];
      Err(LoadError {
        text: super::render(name, text, &e.at.span, &e.message),
        loc: e.loc,
      })
    }
  }
}

/// A rule's variables, by number.
#[derive(Default)]
struct Scope {
  names: Vec<Rc<str>>,
  tys: Vec<Ty>,
}

impl Scope {
  fn get(&self, name: &str) -> Option<usize> {
    self.names.iter().position(|n| &**n == name)
  }

  fn add(&mut self, name: &Rc<str>, ty: Ty) -> usize {
    self.names.push(name.clone());
    self.tys.push(ty);
    self.names.len() - 1
  }
}

fn is_var(name: &str) -> bool {
  name.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
}

/// A body's items: the terms that commas join, and that groups hold.
fn body_items<'t>(term: &'t Term, out: &mut Vec<&'t Term>) {
  match term {
    Term::Bunch(_, tup) => tup.iter().for_each(|t| body_items(t, out)),
    Term::Group(_, t) if matches!(&**t, Term::Bunch(..)) => body_items(t, out),
    term => out.push(term),
  }
}

/// A sentence's head and body: `Head :- Body`, or `Head` alone.
fn split(term: &Term) -> (&Term, Option<&Term>) {
  match term {
    Term::Deduct(_, head, body) => (head, Some(body)),
    term => (term, None),
  }
}

fn is_literal(term: &Term) -> bool {
  match term {
    Term::IntLit(..) | Term::FloatLit(..) | Term::AtomLit(..) | Term::LogicLit(..) => true,
    Term::Minus(_, t) => matches!(&**t, Term::IntLit(..) | Term::FloatLit(..)),
    _ => false,
  }
}

/// The variables in a term, as names, but not the field names of `name=value`.
fn free_vars<'t>(term: &'t Term, out: &mut Vec<&'t str>) {
  match term {
    Term::Ident(_, name) if is_var(name) => out.push(name),
    Term::Apply(_, tup) => {
      free_vars(&tup[0], out);
      for arg in &tup[1..] {
        match &**arg {
          Term::Equal(_, l, r) if matches!(&**l, Term::Ident(_, n) if is_var(n)) => {
            free_vars(r, out)
          }
          arg => free_vars(arg, out),
        }
      }
    }
    Term::Group(_, t) | Term::Not(_, t) | Term::Minus(_, t) | Term::Query(_, t) => {
      free_vars(t, out)
    }
    Term::Attr(_, t, _) => free_vars(t, out),
    Term::Bunch(_, tup) => tup.iter().for_each(|t| free_vars(t, out)),
    Term::Deduct(_, l, r)
    | Term::Equal(_, l, r)
    | Term::NEqual(_, l, r)
    | Term::Compare(_, _, l, r)
    | Term::Add(_, l, r)
    | Term::Sub(_, l, r)
    | Term::Mul(_, l, r)
    | Term::Div(_, l, r) => {
      free_vars(l, out);
      free_vars(r, out);
    }
    _ => {}
  }
}

struct Loader<'p> {
  prog: &'p mut Program,
  /// The file being checked, by its place in the load.
  file: usize,
}

impl Loader<'_> {
  #[track_caller]
  fn fail<T>(&self, span: Span, message: impl Into<String>) -> Check<T> {
    Err(CheckError {
      at: At {
        file: self.file,
        span,
      },
      message: message.into(),
      loc: Location::caller(),
    })
  }

  fn at(&self, span: &Span) -> At {
    At {
      file: self.file,
      span: span.clone(),
    }
  }

  fn load(&mut self, mods: &[&Mod]) -> Check<()> {
    // Types' names first, since any declaration may name any type.
    for (file, mod_) in mods.iter().enumerate() {
      self.file = file;
      for sent in &mod_.body {
        if let Sent::Type(_, name, _) | Sent::Enum(_, name, _) = &**sent {
          if self.prog.types.contains_key(&name.text) || self.ty_name(&name.text).is_some() {
            return self.fail(
              name.span.clone(),
              format!("the type `{}` is declared twice", name.text),
            );
          }
          self.prog.types.insert(name.text.clone(), None);
        }
      }
    }
    // Then what types are made from, and constructors; then relations,
    // whose defaults may name any constructor.
    for relations in [false, true] {
      for (file, mod_) in mods.iter().enumerate() {
        self.file = file;
        for sent in &mod_.body {
          match &**sent {
            Sent::Type(_, name, ty) if !relations => {
              let base = self.ty(&ty.text, &ty.span)?;
              self.prog.types.insert(name.text.clone(), Some(base));
            }
            Sent::Enum(_, name, ctors) if !relations => {
              for ctor in ctors {
                self.ctor(ctor, &name.text)?;
              }
            }
            Sent::Decl(span, word, decls, key, flags) => {
              let kind = self.decl_kind(word)?;
              let constructors = matches!(kind, DeclKind::Id | DeclKind::Msg);
              if constructors && !relations {
                if !key.is_empty() || !flags.is_empty() {
                  return self.fail(
                    span.clone(),
                    format!(
                      "`{}` declares constructors, which have no key or flags",
                      kind.word()
                    ),
                  );
                }
                let ty: Rc<str> = if kind == DeclKind::Id { "Id" } else { "Msg" }.into();
                for decl in decls {
                  self.ctor(decl, &ty)?;
                }
              } else if !constructors && relations {
                self.no_flags(flags)?;
                for decl in decls {
                  self.relation(kind, decl, key)?;
                }
              }
            }
            _ => {}
          }
        }
      }
    }
    // Rules, and `on` blocks.
    for (file, mod_) in mods.iter().enumerate() {
      self.file = file;
      for sent in &mod_.body {
        match &**sent {
          Sent::On(_, pattern, body) => self.on(pattern, body, &[])?,
          Sent::Type(..) | Sent::Enum(..) | Sent::Decl(..) => {}
          _ => self.sents(std::slice::from_ref(sent), &[])?,
        }
      }
    }
    self.strata()
  }

  /// What a declaration's word declares.
  fn decl_kind(&self, word: &SpanIdent) -> Check<DeclKind> {
    match DeclKind::from_word(&word.text) {
      Some(kind) => Ok(kind),
      None => self.fail(
        word.span.clone(),
        format!(
          "lem's declarations are `input`, `state`, `rel`, `effect`, `id` and `msg`, not `{}`",
          word.text
        ),
      ),
    }
  }

  /// lem's declarations take no flags: a layer above lem reads its own,
  /// and removes them.
  fn no_flags(&self, flags: &[SpanIdent]) -> Check<()> {
    match flags.first() {
      Some(flag) => self.fail(
        flag.span.clone(),
        format!("there's no flag `{}` in lem", flag.text),
      ),
      None => Ok(()),
    }
  }

  /// A base type's name, as a type.
  fn ty_name(&self, name: &str) -> Option<Ty> {
    Some(match name {
      "Bool" => Ty::Bool,
      "Int" => Ty::Int,
      "Num" => Ty::Num,
      "Text" => Ty::Text,
      "Time" => Ty::Time,
      "Any" => Ty::Any,
      _ => return None,
    })
  }

  fn ty(&self, name: &Rc<str>, span: &Span) -> Check<Ty> {
    match self.ty_name(name) {
      Some(ty) => Ok(ty),
      None if self.prog.types.contains_key(name) => Ok(Ty::Named(name.clone())),
      None => self.fail(span.clone(), format!("there's no type `{name}`")),
    }
  }

  fn declare(&mut self, name: &Rc<str>, span: &Span, what: Name) -> Check<()> {
    if let Some(first) = self.prog.names.get(name) {
      let at = match first {
        Name::Rel(i) => &self.prog.rels[*i].at,
        Name::Ctor(i) => &self.prog.ctors[*i].at,
      };
      let first = self.prog.place(at);
      return self.fail(
        span.clone(),
        format!("`{name}` is declared twice, first at {first}"),
      );
    }
    self.prog.names.insert(name.clone(), what);
    Ok(())
  }

  /// A declaration's fields. `named` asks that each have a name, as a
  /// relation's do.
  fn fields(&mut self, params: &[Param], named: bool, owner: &str) -> Check<Vec<Field>> {
    let mut fields: Vec<Field> = Vec::new();
    for param in params {
      let name = param.name.as_ref().map(|n| n.text.clone());
      if named && name.is_none() {
        return self.fail(
          param.span.clone(),
          format!(
            "{owner}'s fields need names: write `name: {}`",
            param.ty.text
          ),
        );
      }
      if let Some(name) = &name {
        if fields.iter().any(|f| f.name.as_ref() == Some(name)) {
          let span = param
            .name
            .as_ref()
            .map_or(param.span.clone(), |n| n.span.clone());
          return self.fail(span, format!("{owner} has two fields named `{name}`"));
        }
      }
      let ty = self.ty(&param.ty.text, &param.ty.span)?;
      let default = match &param.default {
        None => None,
        Some(term) => {
          let what = format!("{owner}'s `{}`", name.as_deref().unwrap_or("field"));
          Some(self.constant(term, &ty, &what)?)
        }
      };
      fields.push(Field { name, ty, default });
    }
    Ok(fields)
  }

  fn ctor(&mut self, decl: &Decl, ty: &Rc<str>) -> Check<()> {
    let fields = self.fields(&decl.params, false, &decl.name.text)?;
    let i = self.prog.ctors.len();
    self.declare(&decl.name.text, &decl.name.span, Name::Ctor(i))?;
    self.prog.ctors.push(Ctor {
      name: decl.name.text.clone(),
      ty: ty.clone(),
      fields,
      at: self.at(&decl.span),
    });
    Ok(())
  }

  fn relation(&mut self, kind: DeclKind, decl: &Decl, key: &[SpanIdent]) -> Check<()> {
    let fields = self.fields(&decl.params, true, &decl.name.text)?;
    let mut columns = Vec::new();
    for column in key {
      match fields
        .iter()
        .position(|f| f.name.as_ref() == Some(&column.text))
      {
        Some(i) if !columns.contains(&i) => columns.push(i),
        Some(_) => {
          return self.fail(
            column.span.clone(),
            format!("`{}` is in the key twice", column.text),
          )
        }
        None => {
          return self.fail(
            column.span.clone(),
            format!("{} has no field `{}`", decl.name.text, column.text),
          )
        }
      }
    }
    let i = self.prog.rels.len();
    self.declare(&decl.name.text, &decl.name.span, Name::Rel(i))?;
    self.prog.rels.push(Rel {
      name: decl.name.text.clone(),
      kind,
      fields,
      key: columns,
      at: self.at(&decl.span),
    });
    Ok(())
  }

  fn rel_of(&self, name: &Rc<str>, span: &Span) -> Check<usize> {
    match self.prog.names.get(name) {
      Some(Name::Rel(i)) => Ok(*i),
      Some(Name::Ctor(_)) => self.fail(
        span.clone(),
        format!("`{name}` is a constructor, not a relation"),
      ),
      None => self.fail(span.clone(), format!("there's no relation `{name}`")),
    }
  }

  fn ctor_of(&self, name: &Rc<str>, span: &Span) -> Check<usize> {
    match self.prog.names.get(name) {
      Some(Name::Ctor(i)) => Ok(*i),
      Some(Name::Rel(_)) => self.fail(
        span.clone(),
        format!("`{name}` is a relation, not a constructor"),
      ),
      None => self.fail(span.clone(), format!("there's no constructor `{name}`")),
    }
  }

  /// An atom's name, its arguments, and the name's span: `Name(args)`, or
  /// `Name` alone.
  fn atom_parts<'t>(
    &self,
    term: &'t Term,
    want: &str,
  ) -> Check<(&'t Rc<str>, &'t [TermRef], Span)> {
    match term {
      Term::Ident(span, name) if !is_var(name) => Ok((name, &[], span.clone())),
      Term::Apply(_, tup) => match &*tup[0] {
        Term::Ident(span, name) if !is_var(name) => Ok((name, &tup[1..], span.clone())),
        Term::Ident(span, name) => {
          self.fail(span.clone(), format!("lem has no function `{name}` yet"))
        }
        head => self.fail(head.span(), format!("expected {want}'s name")),
      },
      term => self.fail(term.span(), format!("expected {want}, such as `Job(j)`")),
    }
  }

  /// Each field's argument: those by position first, filling `positional`'s
  /// fields in order, then those by `name=value`.
  fn arguments<'t>(
    &self,
    fields: &[Field],
    positional: &[usize],
    args: &'t [TermRef],
    owner: &str,
  ) -> Check<Vec<Option<&'t Term>>> {
    let mut cols = vec![None; fields.len()];
    let (mut next, mut named) = (0, false);
    for arg in args {
      if let Term::Equal(_, l, r) = &**arg {
        if let Term::Ident(span, name) = &**l {
          if is_var(name) {
            named = true;
            let Some(i) = fields.iter().position(|f| f.name.as_ref() == Some(name)) else {
              return self.fail(span.clone(), format!("{owner} has no field `{name}`"));
            };
            if cols[i].is_some() {
              return self.fail(span.clone(), format!("{owner}'s `{name}` is given twice"));
            }
            cols[i] = Some(&**r);
            continue;
          }
        }
      }
      if named {
        return self.fail(
          arg.span(),
          "arguments by position come before those by name",
        );
      }
      let Some(&i) = positional.get(next) else {
        let n = positional.len();
        let s = if n == 1 { "" } else { "s" };
        return self.fail(
          arg.span(),
          format!("{owner} takes {n} argument{s} by position"),
        );
      };
      next += 1;
      cols[i] = Some(&**arg);
    }
    Ok(cols)
  }

  /// A literal's own type and value.
  fn natural(&self, term: &Term) -> Check<(Ty, Value)> {
    Ok(match term {
      Term::IntLit(span, s) => {
        let n = match s.strip_prefix("0x") {
          Some(hex) => i64::from_str_radix(hex, 16).ok(),
          None => s.parse().ok(),
        };
        let Some(n) = n else {
          return self.fail(span.clone(), "a number must fit in 64 bits");
        };
        (Ty::Int, Value::Int(n))
      }
      Term::FloatLit(_, s) => (Ty::Num, Value::Num(s.parse().unwrap_or(f64::NAN))),
      Term::AtomLit(_, s) => (Ty::Text, Value::Text(s.clone())),
      Term::LogicLit(_, b) => (Ty::Bool, Value::Bool(*b)),
      Term::Minus(_, t) => match self.natural(t)? {
        (ty, Value::Int(n)) => (ty, Value::Int(-n)),
        (ty, Value::Num(x)) => (ty, Value::Num(-x)),
        _ => return self.fail(term.span(), "`-` negates numbers"),
      },
      term => return self.fail(term.span(), "expected a literal"),
    })
  }

  /// A literal, as a value of `ty`: `what` names where it goes.
  fn literal(&self, term: &Term, ty: &Ty, what: &str) -> Check<Value> {
    let (natural, value) = self.natural(term)?;
    Ok(match (self.prog.base(ty), value) {
      (Ty::Any, value) => value,
      (Ty::Num, Value::Int(n)) => Value::Num(n as f64),
      (Ty::Time, value @ Value::Int(_)) => value,
      (base, value) if base == natural => value,
      _ => {
        return self.fail(
          term.span(),
          format!("{what} is {}, but this is {}", ty.a(), natural.a()),
        )
      }
    })
  }

  /// A value that needs no variables, for a default.
  fn constant(&mut self, term: &Term, ty: &Ty, what: &str) -> Check<Value> {
    let expr = self.expr_fit(&mut Scope::default(), term, ty, what)?;
    match fold(&expr) {
      Some(value) => Ok(value),
      None => self.fail(term.span(), "a default must be a literal or a constructor"),
    }
  }

  /// A term as a value, and its type.
  fn expr(&mut self, scope: &mut Scope, term: &Term) -> Check<(Expr, Ty)> {
    match term {
      Term::Ident(span, name) if is_var(name) => match scope.get(name) {
        Some(v) => Ok((Expr::Var(v), scope.tys[v].clone())),
        None => self.fail(
          span.clone(),
          format!("`{name}` isn't bound: an atom in the body must bind it"),
        ),
      },
      Term::Ident(..) | Term::Apply(..) => self.ctor_expr(scope, term),
      Term::IntLit(..) | Term::FloatLit(..) | Term::AtomLit(..) | Term::LogicLit(..) => {
        let (ty, value) = self.natural(term)?;
        Ok((Expr::Const(value), ty))
      }
      Term::Group(_, t) => self.expr(scope, t),
      Term::Minus(_, t) => {
        let (e, ty) = self.expr(scope, t)?;
        match self.prog.base(&ty) {
          Ty::Int | Ty::Num => Ok((Expr::Neg(Box::new(e)), ty)),
          _ => self.fail(term.span(), format!("`-` negates numbers, not {}", ty.a())),
        }
      }
      Term::Add(_, l, r) => self.arith(scope, Op::Add, term, l, r),
      Term::Sub(_, l, r) => self.arith(scope, Op::Sub, term, l, r),
      Term::Mul(_, l, r) => self.arith(scope, Op::Mul, term, l, r),
      Term::Div(_, l, r) => self.arith(scope, Op::Div, term, l, r),
      Term::PlaceIdent(span) => {
        self.fail(span.clone(), "`_` matches anything, so it can't be a value")
      }
      term => self.fail(term.span(), "expected a value here"),
    }
  }

  fn arith(
    &mut self,
    scope: &mut Scope,
    op: Op,
    term: &Term,
    l: &Term,
    r: &Term,
  ) -> Check<(Expr, Ty)> {
    let (a, ta) = self.expr(scope, l)?;
    let (b, tb) = self.expr(scope, r)?;
    let (ba, bb) = (self.prog.base(&ta), self.prog.base(&tb));
    let ty = match (op, &ba, &bb) {
      (Op::Div, Ty::Int | Ty::Num, Ty::Int | Ty::Num) => Ty::Num,
      (_, Ty::Int, Ty::Int) => Ty::Int,
      (_, Ty::Int | Ty::Num, Ty::Int | Ty::Num) => Ty::Num,
      (Op::Add | Op::Sub, Ty::Time, Ty::Int) | (Op::Add, Ty::Int, Ty::Time) => Ty::Time,
      (Op::Sub, Ty::Time, Ty::Time) => Ty::Int,
      (Op::Add, Ty::Text, Ty::Text) => Ty::Text,
      _ => {
        let op = op.spelling();
        return self.fail(
          term.span(),
          format!("`{op}` doesn't take {} and {}", ta.a(), tb.a()),
        );
      }
    };
    let num = |e: Expr, base: &Ty| match base {
      Ty::Int if ty == Ty::Num => Expr::ToNum(Box::new(e)),
      _ => e,
    };
    let (a, b) = (num(a, &ba), num(b, &bb));
    Ok((Expr::Arith(op, Box::new(a), Box::new(b)), ty))
  }

  fn ctor_expr(&mut self, scope: &mut Scope, term: &Term) -> Check<(Expr, Ty)> {
    let (name, args, span) = self.atom_parts(term, "a constructor")?;
    let ctor = &self.prog.ctors[self.ctor_of(name, &span)?];
    let (cname, ty, fields) = (ctor.name.clone(), ctor.ty.clone(), ctor.fields.clone());
    let positional: Vec<usize> = (0..fields.len()).collect();
    let cols = self.arguments(&fields, &positional, args, &cname)?;
    let mut exprs = Vec::new();
    for (i, (field, col)) in fields.iter().zip(cols).enumerate() {
      exprs.push(match (col, &field.default) {
        (Some(t), _) => self.expr_fit(scope, t, &field.ty, &field_what(&cname, field, i))?,
        (None, Some(value)) => Expr::Const(value.clone()),
        (None, None) => return self.fail(span, format!("{cname} needs {}", field_name(field, i))),
      });
    }
    Ok((Expr::Ctor(cname, exprs), Ty::Named(ty)))
  }

  /// A term as a value of type `ty`; `what` names where it goes, for errors.
  /// A literal, or a constructor written there, fits the types made from
  /// its own type, and an Int widens to Num.
  fn expr_fit(&mut self, scope: &mut Scope, term: &Term, ty: &Ty, what: &str) -> Check<Expr> {
    if is_literal(term) {
      return Ok(Expr::Const(self.literal(term, ty, what)?));
    }
    let (e, got) = self.expr(scope, term)?;
    let base = self.prog.base(ty);
    let written =
      matches!(term, Term::Ident(_, n) if !is_var(n)) || matches!(term, Term::Apply(..));
    if *ty == Ty::Any || got == *ty || (written && got == base) {
      Ok(e)
    } else if got == Ty::Int && base == Ty::Num {
      Ok(Expr::ToNum(Box::new(e)))
    } else {
      self.fail(
        term.span(),
        format!("{what} is {}, but this is {}", ty.a(), got.a()),
      )
    }
  }

  /// What a term in a column of type `ty` matches. `bind` lets it bind new
  /// variables, which a negation's can't.
  fn pat(&mut self, scope: &mut Scope, term: &Term, ty: &Ty, bind: bool, what: &str) -> Check<Pat> {
    match term {
      Term::PlaceIdent(_) => Ok(Pat::Any),
      Term::Group(_, t) => self.pat(scope, t, ty, bind, what),
      Term::Ident(span, name) if is_var(name) => match scope.get(name) {
        Some(v) => {
          let got = &scope.tys[v];
          if got != ty && *ty != Ty::Any && *got != Ty::Any {
            let message = format!("{what} is {}, but `{name}` is {}", ty.a(), got.a());
            return self.fail(span.clone(), message);
          }
          Ok(Pat::Var(v))
        }
        None if bind => Ok(Pat::Var(scope.add(name, ty.clone()))),
        None => {
          let message = format!("`{name}` isn't bound by an atom: write `_` to match anything");
          self.fail(span.clone(), message)
        }
      },
      Term::Ident(..) | Term::Apply(..) => {
        let (name, args, span) = self.atom_parts(term, "a constructor")?;
        let ctor = &self.prog.ctors[self.ctor_of(name, &span)?];
        let (cname, cty, fields) = (
          ctor.name.clone(),
          Ty::Named(ctor.ty.clone()),
          ctor.fields.clone(),
        );
        if *ty != Ty::Any && *ty != cty && self.prog.base(ty) != cty {
          return self.fail(
            term.span(),
            format!("{what} is {}, but this is {}", ty.a(), cty.a()),
          );
        }
        let positional: Vec<usize> = (0..fields.len()).collect();
        let cols = self.arguments(&fields, &positional, args, &cname)?;
        let mut pats = Vec::new();
        for (i, (field, col)) in fields.iter().zip(cols).enumerate() {
          pats.push(match col {
            Some(t) => self.pat(scope, t, &field.ty, bind, &field_what(&cname, field, i))?,
            None => Pat::Any,
          });
        }
        Ok(Pat::Ctor(cname, pats))
      }
      term if is_literal(term) => Ok(Pat::Const(self.literal(term, ty, what)?)),
      term => Ok(Pat::Expr(self.expr_fit(scope, term, ty, what)?)),
    }
  }

  /// A body's atom: its relation, and what each column matches. `bind` lets
  /// it bind variables, which a negation's can't.
  fn atom(&mut self, scope: &mut Scope, term: &Term, bind: bool) -> Check<(usize, Vec<Pat>)> {
    let (name, args, span) = self.atom_parts(term, "an atom")?;
    let rel = self.rel_of(name, &span)?;
    let r = &self.prog.rels[rel];
    let (rname, fields) = (r.name.clone(), r.fields.clone());
    let positional: Vec<usize> = (0..fields.len()).collect();
    let cols = self.arguments(&fields, &positional, args, &rname)?;
    let mut pats = Vec::new();
    for (i, (field, col)) in fields.iter().zip(cols).enumerate() {
      pats.push(match col {
        Some(t) => self.pat(scope, t, &field.ty, bind, &field_what(&rname, field, i))?,
        None => Pat::Any,
      });
    }
    Ok((rel, pats))
  }

  fn test(&mut self, scope: &mut Scope, term: &Term) -> Check<Test> {
    match term {
      Term::Group(_, t) => self.test(scope, t),
      Term::Equal(_, l, r) => {
        let (a, b) = self.pair(scope, term, l, r, false)?;
        Ok(Test::Eq(a, b))
      }
      Term::NEqual(_, l, r) => {
        let (a, b) = self.pair(scope, term, l, r, false)?;
        Ok(Test::Ne(a, b))
      }
      Term::Compare(_, cmp, l, r) => {
        let (a, b) = self.pair(scope, term, l, r, true)?;
        Ok(Test::Cmp(*cmp, a, b))
      }
      term => self.fail(term.span(), "expected an atom, a comparison or `not` here"),
    }
  }

  /// A comparison's two sides, of one type: a literal takes the other side's,
  /// and an Int widens to the other side's Num.
  fn pair(
    &mut self,
    scope: &mut Scope,
    term: &Term,
    l: &Term,
    r: &Term,
    ordered: bool,
  ) -> Check<(Expr, Expr)> {
    let ((a, ta), (b, tb)) = match (is_literal(l), is_literal(r)) {
      (true, false) => {
        let (b, tb) = self.expr(scope, r)?;
        let a = Expr::Const(self.literal(l, &tb, "the other side")?);
        ((a, tb.clone()), (b, tb))
      }
      (false, true) => {
        let (a, ta) = self.expr(scope, l)?;
        let b = Expr::Const(self.literal(r, &ta, "the other side")?);
        ((a, ta.clone()), (b, ta))
      }
      _ => (self.expr(scope, l)?, self.expr(scope, r)?),
    };
    let widen = |e: Expr| Expr::ToNum(Box::new(e));
    let (a, b, ty) = if ta == tb {
      (a, b, ta)
    } else {
      match (self.prog.base(&ta), self.prog.base(&tb)) {
        (Ty::Int, Ty::Num) => (widen(a), b, Ty::Num),
        (Ty::Num, Ty::Int) => (a, widen(b), Ty::Num),
        _ => {
          return self.fail(
            term.span(),
            format!("this compares {} with {}", ta.a(), tb.a()),
          )
        }
      }
    };
    if ordered && !matches!(self.prog.base(&ty), Ty::Int | Ty::Num | Ty::Time | Ty::Text) {
      let message = format!("only numbers, times and text are ordered, not {}", ty.a());
      return self.fail(term.span(), message);
    }
    Ok((a, b))
  }

  /// `x = e`, where `e` is bound and `x` isn't: binds `x`.
  fn binding(&mut self, scope: &mut Scope, term: &Term) -> Check<Option<Lit>> {
    let Term::Equal(_, l, r) = term else {
      return Ok(None);
    };
    for (var, value) in [(l, r), (r, l)] {
      if let Term::Ident(_, name) = &**var {
        if is_var(name) && scope.get(name).is_none() {
          let mut names = Vec::new();
          free_vars(value, &mut names);
          if names.iter().all(|n| scope.get(n).is_some()) {
            let (e, ty) = self.expr(scope, value)?;
            return Ok(Some(Lit::Let(scope.add(name, ty), e)));
          }
        }
      }
    }
    Ok(None)
  }

  /// A body, in the order that it runs: atoms as written, then bindings by
  /// `=`, then tests and negations.
  fn body(&mut self, scope: &mut Scope, items: &[&Term]) -> Check<Vec<Lit>> {
    let is_atom =
      |t: &Term| matches!(t, Term::Apply(..)) || matches!(t, Term::Ident(_, n) if !is_var(n));
    let mut lits = Vec::new();
    let mut rest = Vec::new();
    for &item in items {
      if is_atom(item) {
        let (rel, pats) = self.atom(scope, item, true)?;
        lits.push(Lit::Atom(rel, pats));
      } else {
        rest.push(item);
      }
    }
    loop {
      let before = rest.len();
      let mut i = 0;
      while i < rest.len() {
        match self.binding(scope, rest[i])? {
          Some(lit) => {
            lits.push(lit);
            rest.remove(i);
          }
          None => i += 1,
        }
      }
      if rest.len() == before {
        break;
      }
    }
    for item in rest {
      lits.push(match item {
        Term::Not(_, inner) => {
          let mut inner: &Term = inner;
          while let Term::Group(_, t) = inner {
            inner = t;
          }
          if is_atom(inner) {
            let (rel, pats) = self.atom(scope, inner, false)?;
            Lit::Not(rel, pats)
          } else {
            Lit::NotTest(self.test(scope, inner)?)
          }
        }
        item => Lit::Test(self.test(scope, item)?),
      });
    }
    Ok(lits)
  }

  /// Rules and `for`s, inside the `for`s whose bodies `body` holds.
  fn sents<'t>(&mut self, sents: &'t [SentRef], body: &[&'t Term]) -> Check<()> {
    for sent in sents {
      match &**sent {
        Sent::Just(span, term) => {
          let (head, own) = split(term);
          self.rule(head, own, body, span)?;
        }
        Sent::Block(span, ..) => {
          return self.fail(span.clone(), "lem itself gives a block no meaning");
        }
        Sent::For(_, cond, inner) => {
          let mut items = body.to_vec();
          body_items(cond, &mut items);
          self.sents(inner, &items)?;
        }
        Sent::Del(span, _) => {
          return self.fail(span.clone(), "`del` deletes facts only in `on` blocks")
        }
        Sent::On(span, ..) => {
          return self.fail(span.clone(), "an `on` block can't be inside another block")
        }
        Sent::Type(span, ..) | Sent::Enum(span, ..) | Sent::Decl(span, ..) => {
          return self.fail(span.clone(), "a declaration can't be inside a block");
        }
      }
    }
    Ok(())
  }

  /// A rule, `head :- own`, inside the `for`s whose bodies `body` holds.
  fn rule<'t>(
    &mut self,
    head: &'t Term,
    own: Option<&'t Term>,
    body: &[&'t Term],
    span: &Span,
  ) -> Check<()> {
    let mut items = body.to_vec();
    if let Some(own) = own {
      body_items(own, &mut items);
    }
    let mut scope = Scope::default();
    let lits = self.body(&mut scope, &items)?;
    let (rel, args) = self.head(&mut scope, head)?;
    let why = match self.prog.rels[rel].kind {
      DeclKind::Input => Some("the host sends `input` facts, so rules don't derive them"),
      DeclKind::State => Some("`state` facts change only in `on` blocks"),
      DeclKind::Effect => Some("only `on` blocks send effects"),
      _ => None,
    };
    if let Some(why) = why {
      return self.fail(head.span(), why);
    }
    self.prog.rules.push(Rule {
      at: self.at(span),
      head: rel,
      args,
      body: lits,
      vars: scope.names,
    });
    Ok(())
  }

  /// A head: its relation, and a value per column.
  fn head(&mut self, scope: &mut Scope, term: &Term) -> Check<(usize, Vec<Expr>)> {
    let (name, args, span) = self.atom_parts(term, "a head")?;
    let rel = self.rel_of(name, &span)?;
    let r = &self.prog.rels[rel];
    let (rname, fields) = (r.name.clone(), r.fields.clone());
    let positional: Vec<usize> = (0..fields.len()).collect();
    let cols = self.arguments(&fields, &positional, args, &rname)?;
    let mut exprs = Vec::new();
    for (i, (field, col)) in fields.iter().zip(&cols).enumerate() {
      let what = field_what(&rname, field, i);
      exprs.push(match (col, &field.default) {
        (Some(t), _) => self.expr_fit(scope, t, &field.ty, &what)?,
        (None, Some(value)) => Expr::Const(value.clone()),
        (None, None) => return self.fail(span, format!("{rname} needs {}", field_name(field, i))),
      });
    }
    Ok((rel, exprs))
  }

  /// An `on` block's sentences: `pattern` is its message, and `body` what
  /// the `for`s around them add.
  fn on<'t>(&mut self, pattern: &'t Term, sents: &'t [SentRef], body: &[&'t Term]) -> Check<()> {
    for sent in sents {
      match &**sent {
        Sent::Just(span, term) => {
          let (head, own) = split(term);
          self.action(pattern, head, own, body, false, span)?;
        }
        Sent::Del(span, term) => {
          let (head, own) = split(term);
          self.action(pattern, head, own, body, true, span)?;
        }
        Sent::For(_, cond, inner) => {
          let mut items = body.to_vec();
          body_items(cond, &mut items);
          self.on(pattern, inner, &items)?;
        }
        Sent::Block(span, ..) => {
          return self.fail(
            span.clone(),
            "what an `on` block changes has no block of its own",
          );
        }
        Sent::On(span, ..) => {
          return self.fail(span.clone(), "an `on` block can't be inside another block")
        }
        Sent::Type(span, ..) | Sent::Enum(span, ..) | Sent::Decl(span, ..) => {
          return self.fail(span.clone(), "a declaration can't be inside a block");
        }
      }
    }
    Ok(())
  }

  fn action<'t>(
    &mut self,
    pattern: &'t Term,
    head: &'t Term,
    own: Option<&'t Term>,
    body: &[&'t Term],
    del: bool,
    span: &Span,
  ) -> Check<()> {
    let mut scope = Scope::default();
    let msg = self.pat(
      &mut scope,
      pattern,
      &Ty::Named("Msg".into()),
      true,
      "`on`'s message",
    )?;
    if !matches!(msg, Pat::Ctor(..)) {
      return self.fail(
        pattern.span(),
        "`on` takes a message, as in `on Finish(j):`",
      );
    }
    let mut items = body.to_vec();
    if let Some(own) = own {
      body_items(own, &mut items);
    }
    let lits = self.body(&mut scope, &items)?;
    let (name, _, nspan) = self.atom_parts(head, "a head")?;
    let rel = self.rel_of(name, &nspan)?;
    let kind = self.prog.rels[rel].kind;
    let change = match (del, kind) {
      (false, DeclKind::State | DeclKind::Effect) => Change::Insert(self.head(&mut scope, head)?.1),
      (true, DeclKind::State) => Change::Delete(self.atom(&mut scope, head, false)?.1),
      (false, _) => {
        let message = format!(
          "an `on` block adds `state` facts and sends effects, and {name} is {}",
          kind_a(kind)
        );
        return self.fail(head.span(), message);
      }
      (true, _) => {
        let message = format!(
          "`del` deletes `state` facts, and {name} is {}",
          kind_a(kind)
        );
        return self.fail(head.span(), message);
      }
    };
    self.prog.actions.push(Action {
      at: self.at(span),
      msg,
      body: lits,
      rel,
      change,
      vars: scope.names,
    });
    Ok(())
  }

  /// Orders the rules into strata, so that a relation's rules run after
  /// those of every relation that it negates. A relation that depends on its
  /// own negation is an error.
  fn strata(&mut self) -> Check<()> {
    let n = self.prog.rels.len();
    // (from, to, whether negated, rule): the head depends on the atom.
    let mut edges = Vec::new();
    for (i, rule) in self.prog.rules.iter().enumerate() {
      for lit in &rule.body {
        match lit {
          Lit::Atom(rel, _) => edges.push((*rel, rule.head, false, i)),
          Lit::Not(rel, _) => edges.push((*rel, rule.head, true, i)),
          _ => {}
        }
      }
    }
    let mut reach = vec![vec![false; n]; n];
    for &(from, to, ..) in &edges {
      reach[from][to] = true;
    }
    for k in 0..n {
      let via = reach[k].clone();
      for row in reach.iter_mut() {
        if row[k] {
          for (cell, &r) in row.iter_mut().zip(&via) {
            *cell |= r;
          }
        }
      }
    }
    for &(from, to, negated, i) in &edges {
      if negated && (from == to || reach[to][from]) {
        let at = self.prog.rules[i].at.clone();
        self.file = at.file;
        let (head, atom) = (&self.prog.rels[to].name, &self.prog.rels[from].name);
        let message = format!(
          "{head} depends on `not {atom}`, which depends on {head}, so lem can't order them"
        );
        return self.fail(at.span, message);
      }
    }
    let mut level = vec![0; n];
    for _ in 0..=n {
      let mut changed = false;
      for &(from, to, negated, _) in &edges {
        let want = level[from] + usize::from(negated);
        if want > level[to] {
          level[to] = want;
          changed = true;
        }
      }
      if !changed {
        break;
      }
    }
    let top = level.iter().copied().max().unwrap_or(0);
    self.prog.strata = vec![Vec::new(); top + 1];
    for (i, rule) in self.prog.rules.iter().enumerate() {
      self.prog.strata[level[rule.head]].push(i);
    }
    Ok(())
  }
}

impl Op {
  pub fn spelling(self) -> &'static str {
    match self {
      Op::Add => "+",
      Op::Sub => "-",
      Op::Mul => "*",
      Op::Div => "/",
    }
  }
}

/// "Job's `due`", or "Ring's field 1" for a field without a name.
fn field_what(owner: &str, field: &Field, i: usize) -> String {
  match &field.name {
    Some(name) => format!("{owner}'s `{name}`"),
    None => format!("{owner}'s field {}", i + 1),
  }
}

/// "`call`", or "field 1".
fn field_name(field: &Field, i: usize) -> String {
  match &field.name {
    Some(name) => format!("`{name}`"),
    None => format!("field {}", i + 1),
  }
}

fn kind_a(kind: DeclKind) -> &'static str {
  match kind {
    DeclKind::Input => "an `input` relation",
    DeclKind::State => "a `state` relation",
    DeclKind::Rel => "a `rel` relation",
    DeclKind::Effect => "an effect",
    DeclKind::Id | DeclKind::Msg => "a constructor",
  }
}

/// A value that an expression has without variables, if it has one.
fn fold(expr: &Expr) -> Option<Value> {
  match expr {
    Expr::Const(value) => Some(value.clone()),
    Expr::Ctor(name, args) => {
      let args: Option<Vec<Value>> = args.iter().map(fold).collect();
      Some(Value::Ctor(name.clone(), args?.into()))
    }
    _ => None,
  }
}
