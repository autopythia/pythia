//! The compiled form: a checked program as JSON, which lem's runtime in
//! JavaScript runs (`lem.js`, beside this file), and which Rust reads back
//! ([`read`]), so that every tick test runs on it too. It's data, not code:
//! the relations and constructors with their types, and the rules and `on`
//! blocks, each body as its plan ([`Lit`]), with the places that errors
//! name. It needs no source file.
//!
//! The form, version 1, is an object, a member to a line and an entry of
//! its lists to a line, so that it reads and diffs well:
//!
//! - `lem`: 1, the version;
//! - `files`: the files that it was compiled from, lem's prelude first;
//! - `types`: each type made with `type`, by name, and what it's made from;
//!   and each enum, with `Id` and `Msg`, as null;
//! - `rels`: each relation's `name`, `kind` (`input`, `state`, `rel` or
//!   `effect`), `fields`, each `[name, type]` or `[name, type, default]`,
//!   and `key`, its columns;
//! - `ctors`: each constructor's `name`, `type`, the type that it makes,
//!   and `fields`, whose names may be null;
//! - `rules`: each rule's place, `at`; its `head`, a relation by number;
//!   `args`, an expression per column; `body`; and `vars`, its variables'
//!   names, by number;
//! - `components`: the order to run the rules in ([`Component`]);
//! - `actions`: the sentences of `on` blocks: `at`, `msg`, a pattern;
//!   `body`; `rel`; `vars`; and `insert`, an expression per column, or
//!   `delete`, a pattern per column.
//!
//! A body is a list of parts: `["atom", rel, pats, key, needs]`, `["let",
//! var, expr]`, `["test", op, a, b]`, `["not", rel, pats, key, needs]` and
//! `["nottest", op, a, b]`, where `op` is `=`, `/=`, `<`, `<=`, `>` or `>=`.
//! A pattern is `["any"]`, `["var", n]`, `["const", value]`, `["ctor",
//! name, pats]` or `["expr", expr]`. An expression is `["var", n]`,
//! `["const", value]`, `["ctor", name, exprs]`, `["neg", expr]`, `["arith",
//! op, type, a, b]`, `["num", expr]`, an Int as a Num, or `["text", type,
//! expr]`. A type is a name: `Bool`, `Int`, `Num`, `Text`, `Time`, `Any`,
//! or one that the program declares.
//!
//! A value is JSON as JavaScript holds it: a Bool as `true` or `false`, a
//! number as a number, a Text as a string, and a constructor as an array of
//! its name and arguments. An Int is written in digits, and a Num as Rust's
//! `{:?}` writes it, always with a point or an exponent (`2.0`, `1e16`), so
//! that Rust reads each back as what it was.

use super::check::{
  Action, At, Atom, Change, Component, Ctor, DeclKind, Expr, Field, Lit, Op, Pat, Program, Rel,
  Rule, Test, MAX_ATOMS,
};
use super::parse::Cmp;
use super::value::{int_fits, Ty, Value};
use crate::json::{self, object, Value as Json};
use std::rc::Rc;

/// The compiled form's version, which [`read`] and `lem.js` check.
pub const VERSION: u64 = 1;

/// The program as its compiled form, JSON text.
pub fn compile(prog: &Program) -> String {
  let files = prog.files.iter().map(|(name, _)| Json::from(&**name));
  let types = prog.types().into_iter().map(|(name, base)| {
    let base = base.map_or(Json::Null, |ty| ty_json(&ty));
    (name.to_string(), base)
  });
  let mut out = format!("{{\"lem\":{VERSION},\n");
  member(&mut out, "files", &Json::Array(files.collect()));
  member(&mut out, "types", &Json::Object(types.collect()));
  list(&mut out, "rels", prog.rels.iter().map(rel_json));
  list(&mut out, "ctors", prog.ctors.iter().map(ctor_json));
  list(&mut out, "rules", prog.rules.iter().map(rule_json));
  list(
    &mut out,
    "components",
    prog.components.iter().map(component_json),
  );
  list(&mut out, "actions", prog.actions.iter().map(action_json));
  out.truncate(out.len() - 2);
  out.push_str("}\n");
  out
}

/// `"name":value,` and a line break.
fn member(out: &mut String, name: &str, value: &Json) {
  out.push_str(&format!("{}:{},\n", text(&Json::from(name)), text(value)));
}

/// `"name":[`, then each item on a line of its own, then `],`.
fn list(out: &mut String, name: &str, items: impl Iterator<Item = Json>) {
  let items: Vec<String> = items.map(|item| text(&item)).collect();
  out.push_str(&format!("{}:[", text(&Json::from(name))));
  if !items.is_empty() {
    out.push('\n');
    out.push_str(&items.join(",\n"));
  }
  out.push_str("],\n");
}

fn text(value: &Json) -> String {
  String::from_utf8(value.to_vec()).expect("JSON text is UTF-8")
}

fn array(items: impl IntoIterator<Item = Json>) -> Json {
  Json::Array(items.into_iter().collect())
}

fn numbers<'a>(numbers: impl IntoIterator<Item = &'a usize>) -> Json {
  array(numbers.into_iter().map(|&n| Json::from(n)))
}

fn names(names: &[Rc<str>]) -> Json {
  array(names.iter().map(|name| Json::from(&**name)))
}

fn ty_json(ty: &Ty) -> Json {
  Json::from(ty.to_string())
}

/// A value as the compiled form writes it: JSON, as JavaScript holds it.
pub fn value_json(value: &Value) -> Json {
  match value {
    Value::Bool(b) => Json::from(*b),
    Value::Int(n) => Json::Number(n.to_string()),
    Value::Num(x) => Json::Number(format!("{x:?}")),
    Value::Text(text) => Json::from(&**text),
    Value::Ctor(name, args) => {
      array(std::iter::once(Json::from(&**name)).chain(args.iter().map(value_json)))
    }
  }
}

fn fields_json(fields: &[Field]) -> Json {
  array(fields.iter().map(|field| {
    let name = field.name.as_deref().map_or(Json::Null, Json::from);
    let mut parts = vec![name, ty_json(&field.ty)];
    parts.extend(field.default.as_ref().map(value_json));
    array(parts)
  }))
}

fn rel_json(rel: &Rel) -> Json {
  object([
    ("name", Json::from(&*rel.name)),
    ("kind", Json::from(rel.kind.word())),
    ("fields", fields_json(&rel.fields)),
    ("key", numbers(&rel.key)),
  ])
}

fn ctor_json(ctor: &Ctor) -> Json {
  object([
    ("name", Json::from(&*ctor.name)),
    ("type", Json::from(&*ctor.ty)),
    ("fields", fields_json(&ctor.fields)),
  ])
}

fn rule_json(rule: &Rule) -> Json {
  object([
    ("at", Json::from(&*rule.place)),
    ("head", Json::from(rule.head)),
    ("args", array(rule.args.iter().map(expr_json))),
    ("body", body_json(&rule.body)),
    ("vars", names(&rule.vars)),
  ])
}

fn component_json(component: &Component) -> Json {
  object([
    ("rels", numbers(&component.rels)),
    ("rules", numbers(&component.rules)),
    ("recursive", Json::from(component.recursive)),
  ])
}

fn action_json(action: &Action) -> Json {
  let change = match &action.change {
    Change::Insert(exprs) => ("insert", array(exprs.iter().map(expr_json))),
    Change::Delete(pats) => ("delete", array(pats.iter().map(pat_json))),
  };
  object([
    ("at", Json::from(&*action.place)),
    ("msg", pat_json(&action.msg)),
    ("body", body_json(&action.body)),
    ("rel", Json::from(action.rel)),
    change,
    ("vars", names(&action.vars)),
  ])
}

fn body_json(body: &[Lit]) -> Json {
  array(body.iter().map(|lit| match lit {
    Lit::Atom(atom) => atom_json("atom", atom),
    Lit::Let(v, e) => array([Json::from("let"), Json::from(*v), expr_json(e)]),
    Lit::Test(t) => test_json("test", t),
    Lit::Not(atom) => atom_json("not", atom),
    Lit::NotTest(t) => test_json("nottest", t),
  }))
}

fn atom_json(tag: &str, atom: &Atom) -> Json {
  array([
    Json::from(tag),
    Json::from(atom.rel),
    array(atom.pats.iter().map(pat_json)),
    numbers(&atom.key),
    numbers(&atom.needs),
  ])
}

fn test_json(tag: &str, test: &Test) -> Json {
  let (op, a, b) = match test {
    Test::Eq(a, b) => ("=", a, b),
    Test::Ne(a, b) => ("/=", a, b),
    Test::Cmp(cmp, a, b) => (cmp_spelling(*cmp), a, b),
  };
  array([Json::from(tag), Json::from(op), expr_json(a), expr_json(b)])
}

fn cmp_spelling(cmp: Cmp) -> &'static str {
  match cmp {
    Cmp::Lt => "<",
    Cmp::LtEq => "<=",
    Cmp::Gt => ">",
    Cmp::GtEq => ">=",
  }
}

fn pat_json(pat: &Pat) -> Json {
  match pat {
    Pat::Any => array([Json::from("any")]),
    Pat::Var(v) => array([Json::from("var"), Json::from(*v)]),
    Pat::Const(value) => array([Json::from("const"), value_json(value)]),
    Pat::Ctor(name, pats) => array([
      Json::from("ctor"),
      Json::from(&**name),
      array(pats.iter().map(pat_json)),
    ]),
    Pat::Expr(e) => array([Json::from("expr"), expr_json(e)]),
  }
}

fn expr_json(e: &Expr) -> Json {
  match e {
    Expr::Var(v) => array([Json::from("var"), Json::from(*v)]),
    Expr::Const(value) => array([Json::from("const"), value_json(value)]),
    Expr::Ctor(name, args) => array([
      Json::from("ctor"),
      Json::from(&**name),
      array(args.iter().map(expr_json)),
    ]),
    Expr::Neg(e) => array([Json::from("neg"), expr_json(e)]),
    Expr::Arith(op, ty, a, b) => array([
      Json::from("arith"),
      Json::from(op.spelling()),
      ty_json(ty),
      expr_json(a),
      expr_json(b),
    ]),
    Expr::ToNum(e) => array([Json::from("num"), expr_json(e)]),
    Expr::Text(ty, e) => array([Json::from("text"), ty_json(ty), expr_json(e)]),
  }
}

/// Reads a program's compiled form back ([`compile`]), and checks that it
/// holds together, so that the runtime can run it: every relation,
/// constructor, type, variable and column that it names is there, arities
/// agree, and a body's atoms come first. The program has no source: its
/// places are in its rules and actions ([`Rule::place`]).
pub fn read(text: &str) -> Result<Program, String> {
  let form = json::parse(text).map_err(|e| format!("the compiled form isn't JSON: {e}"))?;
  if form.get("lem").and_then(Json::as_u64) != Some(VERSION) {
    return Err(format!(
      "this isn't the compiled form of a lem program, version {VERSION}"
    ));
  }
  let files = items(&form, "files")?
    .iter()
    .map(|file| string(file, "a file").map(Rc::from))
    .collect::<Result<Vec<Rc<str>>, String>>()?;
  let Some(Json::Object(members)) = form.get("types") else {
    return Err("the compiled form's `types` is an object".to_string());
  };
  let mut decls = Decls {
    types: members
      .iter()
      .map(|(name, _)| Rc::from(name.as_str()))
      .collect(),
    arity: Vec::new(),
  };
  for ctor in items(&form, "ctors")? {
    let name = string(get(ctor, "name")?, "a constructor's `name`")?;
    decls
      .arity
      .push((Rc::from(name), items(ctor, "fields")?.len()));
  }
  let mut types = Vec::new();
  for (name, base) in members {
    let base = match base {
      Json::Null => None,
      base => Some(decls.ty(base)?),
    };
    types.push((Rc::from(name.as_str()), base));
  }
  let rels = items(&form, "rels")?
    .iter()
    .map(|rel| decls.rel(rel))
    .collect::<Result<Vec<Rel>, String>>()?;
  let ctors = items(&form, "ctors")?
    .iter()
    .map(|ctor| decls.ctor(ctor))
    .collect::<Result<Vec<Ctor>, String>>()?;
  let mut prog = Program::assemble(files, rels, ctors, types)?;
  for (i, rule) in items(&form, "rules")?.iter().enumerate() {
    let rule = read_rule(&prog, &decls, rule).map_err(|e| format!("rule {i}: {e}"))?;
    prog.rules.push(rule);
  }
  for (i, action) in items(&form, "actions")?.iter().enumerate() {
    let action = read_action(&prog, &decls, action).map_err(|e| format!("action {i}: {e}"))?;
    prog.actions.push(action);
  }
  for component in items(&form, "components")? {
    let rels = indexes(get(component, "rels")?, prog.rels.len(), "a relation")?;
    let rules = indexes(get(component, "rules")?, prog.rules.len(), "a rule")?;
    let Some(recursive) = get(component, "recursive")?.as_bool() else {
      return Err("a component's `recursive` is true or false".to_string());
    };
    prog.components.push(Component {
      rels,
      rules,
      recursive,
    });
  }
  Ok(prog)
}

/// The member `name` of an object.
fn get<'j>(value: &'j Json, name: &str) -> Result<&'j Json, String> {
  value
    .get(name)
    .ok_or_else(|| format!("expected a member `{name}`"))
}

/// The items of the array that's the member `name` of an object.
fn items<'j>(value: &'j Json, name: &str) -> Result<&'j [Json], String> {
  match get(value, name)? {
    Json::Array(items) => Ok(items),
    other => Err(format!("`{name}` is an array, not {}", other.kind())),
  }
}

fn list_of<'j>(value: &'j Json, what: &str) -> Result<&'j [Json], String> {
  match value {
    Json::Array(items) => Ok(items),
    other => Err(format!("{what} is an array, not {}", other.kind())),
  }
}

fn string<'j>(value: &'j Json, what: &str) -> Result<&'j str, String> {
  value
    .as_str()
    .ok_or_else(|| format!("{what} is a string, not {}", value.kind()))
}

/// A number below `len`: a relation's, a rule's, a column's or a variable's.
fn index(value: &Json, len: usize, what: &str) -> Result<usize, String> {
  match value.as_u64().and_then(|n| usize::try_from(n).ok()) {
    Some(n) if n < len => Ok(n),
    _ => Err(format!(
      "{what} is a number below {len}, not {}",
      text(value)
    )),
  }
}

fn indexes(value: &Json, len: usize, what: &str) -> Result<Vec<usize>, String> {
  list_of(value, what)?
    .iter()
    .map(|n| index(n, len, what))
    .collect()
}

/// What reading values and types needs before the program is assembled:
/// the types' names, and each constructor's number of fields.
struct Decls {
  types: Vec<Rc<str>>,
  arity: Vec<(Rc<str>, usize)>,
}

impl Decls {
  fn ty(&self, value: &Json) -> Result<Ty, String> {
    let name = string(value, "a type")?;
    Ok(match name {
      "Bool" => Ty::Bool,
      "Int" => Ty::Int,
      "Num" => Ty::Num,
      "Text" => Ty::Text,
      "Time" => Ty::Time,
      "Any" => Ty::Any,
      name if self.types.iter().any(|n| &**n == name) => Ty::Named(name.into()),
      name => return Err(format!("there's no type `{name}`")),
    })
  }

  /// A value, as [`value_json`] writes it.
  fn value(&self, value: &Json) -> Result<Value, String> {
    Ok(match value {
      Json::Bool(b) => Value::Bool(*b),
      Json::Number(text) if text.contains(['.', 'e', 'E']) => {
        let x: f64 = text.parse().map_err(|_| format!("{text} isn't a number"))?;
        Value::num(x).ok_or_else(|| format!("{text} isn't finite"))?
      }
      Json::Number(text) => match text.parse::<i64>() {
        Ok(n) if int_fits(n) => Value::Int(n),
        _ => return Err(format!("{text} isn't a whole number within ±(2^53 − 1)")),
      },
      Json::String(text) => Value::Text(text.as_str().into()),
      Json::Array(items) => {
        let name = string(items.first().unwrap_or(&Json::Null), "a constructor's name")?;
        match self.arity.iter().find(|(n, _)| &**n == name) {
          Some((_, n)) if *n == items.len() - 1 => {}
          Some((_, n)) => return Err(format!("{name} takes {n} values")),
          None => return Err(format!("there's no constructor `{name}`")),
        }
        let args = items[1..]
          .iter()
          .map(|item| self.value(item))
          .collect::<Result<Vec<Value>, String>>()?;
        Value::Ctor(name.into(), args.into())
      }
      other => return Err(format!("a value isn't {}", other.kind())),
    })
  }

  fn fields(&self, value: &Json) -> Result<Vec<Field>, String> {
    let mut fields = Vec::new();
    for field in list_of(value, "`fields`")? {
      let parts = list_of(field, "a field")?;
      let (name, ty, default) = match parts {
        [name, ty] => (name, ty, None),
        [name, ty, default] => (name, ty, Some(self.value(default)?)),
        _ => return Err("a field is [name, type] or [name, type, default]".to_string()),
      };
      let name = match name {
        Json::Null => None,
        name => Some(Rc::from(string(name, "a field's name")?)),
      };
      let ty = self.ty(ty)?;
      fields.push(Field { name, ty, default });
    }
    Ok(fields)
  }

  fn rel(&self, value: &Json) -> Result<Rel, String> {
    let name = string(get(value, "name")?, "a relation's `name`")?;
    let kind = match DeclKind::from_word(string(get(value, "kind")?, "a relation's `kind`")?) {
      Some(kind @ (DeclKind::Input | DeclKind::State | DeclKind::Rel | DeclKind::Effect)) => kind,
      _ => return Err(format!("{name}'s `kind` is input, state, rel or effect")),
    };
    let fields = self.fields(get(value, "fields")?)?;
    let key = indexes(get(value, "key")?, fields.len(), "a column")?;
    Ok(Rel {
      name: name.into(),
      kind,
      fields,
      key,
      at: At::nowhere(),
    })
  }

  fn ctor(&self, value: &Json) -> Result<Ctor, String> {
    let name = string(get(value, "name")?, "a constructor's `name`")?;
    let ty = string(get(value, "type")?, "a constructor's `type`")?;
    if !self.types.iter().any(|n| &**n == ty) {
      return Err(format!("there's no type `{ty}`, which {name} makes"));
    }
    Ok(Ctor {
      name: name.into(),
      ty: ty.into(),
      fields: self.fields(get(value, "fields")?)?,
      at: At::nowhere(),
    })
  }
}

fn read_rule(prog: &Program, decls: &Decls, value: &Json) -> Result<Rule, String> {
  let vars = names_of(get(value, "vars")?)?;
  let body = Body {
    prog,
    decls,
    vars: vars.len(),
  };
  let head = index(get(value, "head")?, prog.rels.len(), "a relation")?;
  let args = body.exprs(get(value, "args")?, prog.rels[head].fields.len())?;
  Ok(Rule {
    at: At::nowhere(),
    place: string(get(value, "at")?, "`at`")?.into(),
    head,
    args,
    body: body.lits(get(value, "body")?)?,
    vars,
  })
}

fn read_action(prog: &Program, decls: &Decls, value: &Json) -> Result<Action, String> {
  let vars = names_of(get(value, "vars")?)?;
  let body = Body {
    prog,
    decls,
    vars: vars.len(),
  };
  let rel = index(get(value, "rel")?, prog.rels.len(), "a relation")?;
  let arity = prog.rels[rel].fields.len();
  let change = match (value.get("insert"), value.get("delete")) {
    (Some(exprs), None) => Change::Insert(body.exprs(exprs, arity)?),
    (None, Some(pats)) => Change::Delete(body.pats(pats, arity)?),
    _ => return Err("an action has `insert` or `delete`".to_string()),
  };
  Ok(Action {
    at: At::nowhere(),
    place: string(get(value, "at")?, "`at`")?.into(),
    msg: body.pat(get(value, "msg")?)?,
    body: body.lits(get(value, "body")?)?,
    rel,
    change,
    vars,
  })
}

fn names_of(value: &Json) -> Result<Vec<Rc<str>>, String> {
  list_of(value, "`vars`")?
    .iter()
    .map(|name| string(name, "a variable's name").map(Rc::from))
    .collect()
}

/// What reading a body needs: the program, and its rule's number of
/// variables.
struct Body<'p> {
  prog: &'p Program,
  decls: &'p Decls,
  vars: usize,
}

impl Body<'_> {
  fn var(&self, value: &Json) -> Result<usize, String> {
    index(value, self.vars, "a variable")
  }

  /// The parts of a body, its atoms first.
  fn lits(&self, value: &Json) -> Result<Vec<Lit>, String> {
    let mut lits = Vec::new();
    for lit in list_of(value, "`body`")? {
      let parts = list_of(lit, "a part of a body")?;
      let tag = string(parts.first().unwrap_or(&Json::Null), "a part's tag")?;
      let lit = match (tag, parts.len()) {
        ("atom" | "not", 5) => {
          let rel = index(&parts[1], self.prog.rels.len(), "a relation")?;
          let arity = self.prog.rels[rel].fields.len();
          let atom = Atom {
            rel,
            pats: self.pats(&parts[2], arity)?,
            key: indexes(&parts[3], arity, "a column")?,
            needs: list_of(&parts[4], "`needs`")?
              .iter()
              .map(|v| self.var(v))
              .collect::<Result<Vec<usize>, String>>()?,
          };
          match tag {
            "atom" if lits.iter().any(|lit| !matches!(lit, Lit::Atom(_))) => {
              return Err("a body's atoms come first".to_string());
            }
            "atom" => Lit::Atom(atom),
            _ => Lit::Not(atom),
          }
        }
        ("let", 3) => Lit::Let(self.var(&parts[1])?, self.expr(&parts[2])?),
        ("test" | "nottest", 4) => {
          let (a, b) = (self.expr(&parts[2])?, self.expr(&parts[3])?);
          let test = match string(&parts[1], "a test's operator")? {
            "=" => Test::Eq(a, b),
            "/=" => Test::Ne(a, b),
            "<" => Test::Cmp(Cmp::Lt, a, b),
            "<=" => Test::Cmp(Cmp::LtEq, a, b),
            ">" => Test::Cmp(Cmp::Gt, a, b),
            ">=" => Test::Cmp(Cmp::GtEq, a, b),
            op => return Err(format!("there's no test `{op}`")),
          };
          match tag {
            "test" => Lit::Test(test),
            _ => Lit::NotTest(test),
          }
        }
        _ => return Err(format!("there's no part `{tag}` of {} items", parts.len())),
      };
      lits.push(lit);
    }
    if lits
      .iter()
      .filter(|lit| matches!(lit, Lit::Atom(_)))
      .count()
      > MAX_ATOMS
    {
      return Err(format!("a body has at most {MAX_ATOMS} atoms"));
    }
    Ok(lits)
  }

  /// `n` patterns.
  fn pats(&self, value: &Json, n: usize) -> Result<Vec<Pat>, String> {
    let pats = list_of(value, "patterns")?;
    if pats.len() != n {
      return Err(format!("{} patterns where {n} go", pats.len()));
    }
    pats.iter().map(|pat| self.pat(pat)).collect()
  }

  fn pat(&self, value: &Json) -> Result<Pat, String> {
    let parts = list_of(value, "a pattern")?;
    let tag = string(parts.first().unwrap_or(&Json::Null), "a pattern's tag")?;
    Ok(match (tag, parts.len()) {
      ("any", 1) => Pat::Any,
      ("var", 2) => Pat::Var(self.var(&parts[1])?),
      ("const", 2) => Pat::Const(self.decls.value(&parts[1])?),
      ("ctor", 3) => {
        let (name, n) = self.ctor(&parts[1])?;
        Pat::Ctor(name, self.pats(&parts[2], n)?)
      }
      ("expr", 2) => Pat::Expr(self.expr(&parts[1])?),
      _ => {
        return Err(format!(
          "there's no pattern `{tag}` of {} items",
          parts.len()
        ))
      }
    })
  }

  /// A constructor's name, and its number of fields.
  fn ctor(&self, value: &Json) -> Result<(Rc<str>, usize), String> {
    let name = string(value, "a constructor's name")?;
    match self.prog.ctor(name) {
      Some(ctor) => Ok((ctor.name.clone(), ctor.fields.len())),
      None => Err(format!("there's no constructor `{name}`")),
    }
  }

  /// `n` expressions.
  fn exprs(&self, value: &Json, n: usize) -> Result<Vec<Expr>, String> {
    let exprs = list_of(value, "expressions")?;
    if exprs.len() != n {
      return Err(format!("{} expressions where {n} go", exprs.len()));
    }
    exprs.iter().map(|e| self.expr(e)).collect()
  }

  fn expr(&self, value: &Json) -> Result<Expr, String> {
    let parts = list_of(value, "an expression")?;
    let tag = string(parts.first().unwrap_or(&Json::Null), "an expression's tag")?;
    let sub = |i: usize| self.expr(&parts[i]).map(Box::new);
    Ok(match (tag, parts.len()) {
      ("var", 2) => Expr::Var(self.var(&parts[1])?),
      ("const", 2) => Expr::Const(self.decls.value(&parts[1])?),
      ("ctor", 3) => {
        let (name, n) = self.ctor(&parts[1])?;
        Expr::Ctor(name, self.exprs(&parts[2], n)?)
      }
      ("neg", 2) => Expr::Neg(sub(1)?),
      ("arith", 5) => {
        let op = match string(&parts[1], "an operator")? {
          "+" => Op::Add,
          "-" => Op::Sub,
          "*" => Op::Mul,
          "/" => Op::Div,
          op => return Err(format!("there's no operator `{op}`")),
        };
        let ty = match self.decls.ty(&parts[2])? {
          ty @ (Ty::Int | Ty::Time | Ty::Num | Ty::Text) => ty,
          ty => {
            return Err(format!(
              "arithmetic gives an Int, Time, Num or Text, not {ty}"
            ))
          }
        };
        Expr::Arith(op, ty, sub(3)?, sub(4)?)
      }
      ("num", 2) => Expr::ToNum(sub(1)?),
      ("text", 3) => Expr::Text(self.decls.ty(&parts[1])?, sub(2)?),
      _ => {
        return Err(format!(
          "there's no expression `{tag}` of {} items",
          parts.len()
        ))
      }
    })
  }
}
