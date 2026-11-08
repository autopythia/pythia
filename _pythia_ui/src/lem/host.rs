//! What the host and lem say to each other, as JSON: the host's facts, by
//! relation, with fields by name; messages; and the steps of a tick, as the
//! tests give them to every runtime ([`Step`]). `lem.js` reads the same
//! JSON the same way, with the same errors.
//!
//! A value goes by its declared type: a Bool as `true` or `false`; an Int
//! or a Time as a whole number within ±(2^53 − 1); a Num as a finite
//! number; a Text, or a type made from it, as a string; and a constructor
//! as its name, or as an array of its name and its fields by position,
//! where fields with defaults may be left off the end. A value of type
//! `Any` is a Bool, a whole number, a string, which is a Text there, or a
//! constructor's array.
//!
//! A step is `["+", relation, fields]`, which asserts the host's fact;
//! `["-", relation, fields]`, which retracts it; `["!", message]`; or
//! `["?", relation]`, a relation to show. The tests' steps come from their
//! lines ([`script_json`]), and a line that only Rust can read, because
//! it's wrong, becomes `["error", why]`.

use super::check::{Field, Program};
use super::compile;
use super::eval::{read_step, Runtime, Script, Step, Tuple};
use super::value::{Ty, Value, MAX_INT};
use crate::json::Value as Json;

/// A value of type `ty`, from JSON; `what` names it, for errors.
pub fn value(prog: &Program, ty: &Ty, json: &Json, what: &str) -> Result<Value, String> {
  let base = prog.base(ty);
  let wrong = || Err(format!("{what} is {}, not {}", ty.a(), json.kind()));
  match (&base, json) {
    (Ty::Bool, Json::Bool(b)) => Ok(Value::Bool(*b)),
    (Ty::Int | Ty::Time, Json::Number(text)) => whole(text)
      .map(Value::Int)
      .ok_or_else(|| format!("{what} is {}, a whole number within ±(2^53 − 1)", ty.a())),
    (Ty::Num, Json::Number(text)) => text
      .parse()
      .ok()
      .and_then(Value::num)
      .ok_or_else(|| format!("{what} is {}, a finite number", ty.a())),
    (Ty::Text, Json::String(text)) => Ok(Value::Text(text.as_str().into())),
    (Ty::Any, Json::Bool(b)) => Ok(Value::Bool(*b)),
    (Ty::Any, Json::Number(text)) => whole(text)
      .map(Value::Int)
      .ok_or_else(|| format!("{what} takes any value but a Num")),
    (Ty::Any, Json::String(text)) => Ok(Value::Text(text.as_str().into())),
    (Ty::Any, Json::Array(items)) => ctor(prog, None, items, what),
    (Ty::Named(name), Json::String(_)) => ctor(prog, Some(name), std::slice::from_ref(json), what),
    (Ty::Named(name), Json::Array(items)) => ctor(prog, Some(name), items, what),
    _ => wrong(),
  }
}

/// A whole number within ±(2^53 − 1), from a JSON number's text, as
/// JavaScript reads it: to the nearest double, so `2.0` is 2.
fn whole(text: &str) -> Option<i64> {
  let x: f64 = text.parse().ok()?;
  match x.fract() == 0.0 && x.abs() <= MAX_INT as f64 {
    true => Some(x as i64),
    false => None,
  }
}

/// A constructor from its array: its name, then its fields by position;
/// of the type `ty`, if one is asked for.
fn ctor(prog: &Program, ty: Option<&str>, items: &[Json], what: &str) -> Result<Value, String> {
  let Some(name) = items.first().and_then(Json::as_str) else {
    return Err(format!(
      "{what} is a constructor: its name, or an array of its name and its fields"
    ));
  };
  let Some(c) = prog.ctor(name) else {
    return Err(format!("there's no constructor `{name}`"));
  };
  if let Some(ty) = ty.filter(|ty| *ty != &*c.ty) {
    let (want, made) = (Ty::Named(ty.into()).a(), Ty::Named(c.ty.clone()).a());
    return Err(format!("{what} is {want}, but {name} makes {made}"));
  }
  let given = &items[1..];
  if given.len() > c.fields.len() {
    let n = match c.fields.len() {
      1 => "1 value".to_string(),
      n => format!("{n} values"),
    };
    return Err(format!("{name} takes {n}, not {}", given.len()));
  }
  let mut args = Vec::new();
  for (i, field) in c.fields.iter().enumerate() {
    args.push(match (given.get(i), &field.default) {
      (Some(json), _) => value(prog, &field.ty, json, &field_what(name, field, i))?,
      (None, Some(default)) => default.clone(),
      (None, None) => return Err(format!("{name} needs {}", field_name(field, i))),
    });
  }
  Ok(Value::Ctor(c.name.clone(), args.into()))
}

/// "Job's `due`", or "Ring's field 1".
fn field_what(owner: &str, field: &Field, i: usize) -> String {
  match &field.name {
    Some(name) => format!("{owner}'s `{name}`"),
    None => format!("{owner}'s field {}", i + 1),
  }
}

/// "`due`", or "field 1".
fn field_name(field: &Field, i: usize) -> String {
  match &field.name {
    Some(name) => format!("`{name}`"),
    None => format!("field {}", i + 1),
  }
}

/// A value of type `ty` as JSON ([`value`] reads it back).
pub fn value_json(prog: &Program, ty: &Ty, value: &Value) -> Json {
  let Value::Ctor(name, args) = value else {
    return compile::value_json(value);
  };
  if args.is_empty() && prog.base(ty) != Ty::Any {
    return Json::from(&**name);
  }
  let fields = prog.ctor(name).map_or(&[][..], |c| &c.fields);
  let args = fields
    .iter()
    .zip(args.iter())
    .map(|(field, arg)| value_json(prog, &field.ty, arg));
  Json::Array(std::iter::once(Json::from(&**name)).chain(args).collect())
}

/// A fact of the relation named `rel`, from its fields by name; fields
/// with defaults may be left out.
pub fn fact(prog: &Program, rel: &str, json: &Json) -> Result<(usize, Tuple), String> {
  let Some(r) = prog.rel(rel) else {
    return Err(format!("there's no relation `{rel}`"));
  };
  let Json::Object(members) = json else {
    return Err(format!(
      "a fact of {rel} is an object of its fields, not {}",
      json.kind()
    ));
  };
  let fields = &prog.rels[r].fields;
  if let Some((name, _)) = members.iter().find(|(name, _)| {
    !fields
      .iter()
      .any(|f| f.name.as_deref() == Some(name.as_str()))
  }) {
    return Err(format!("{rel} has no field `{name}`"));
  }
  let mut values = Vec::new();
  for (i, field) in fields.iter().enumerate() {
    let name = field.name.as_deref().unwrap_or_default();
    values.push(match (json.get(name), &field.default) {
      (Some(json), _) => value(prog, &field.ty, json, &field_what(rel, field, i))?,
      (None, Some(default)) => default.clone(),
      (None, None) => return Err(format!("{rel} needs `{name}`")),
    });
  }
  Ok((r, values.into()))
}

/// A fact as JSON, its fields by name ([`fact`] reads it back).
pub fn fact_json(prog: &Program, rel: usize, tuple: &Tuple) -> Json {
  let fields = &prog.rels[rel].fields;
  Json::Object(
    fields
      .iter()
      .zip(tuple.iter())
      .map(|(field, value)| {
        let name = field.name.as_deref().unwrap_or_default().to_string();
        (name, value_json(prog, &field.ty, value))
      })
      .collect(),
  )
}

/// A step from JSON.
pub fn step(prog: &Program, json: &Json) -> Result<Step, String> {
  let items = match json {
    Json::Array(items) => &items[..],
    _ => &[],
  };
  let msg_ty = Ty::Named("Msg".into());
  match items {
    [Json::String(tag), Json::String(rel), fields] if tag == "+" => {
      let (rel, tuple) = fact(prog, rel, fields)?;
      Ok(Step::Assert(rel, tuple))
    }
    [Json::String(tag), Json::String(rel), fields] if tag == "-" => {
      let (rel, tuple) = fact(prog, rel, fields)?;
      Ok(Step::Retract(rel, tuple))
    }
    [Json::String(tag), msg] if tag == "!" => {
      Ok(Step::Msg(value(prog, &msg_ty, msg, "a message")?))
    }
    [Json::String(tag), Json::String(rel)] if tag == "?" => Ok(Step::Show(rel.clone())),
    [Json::String(tag), Json::String(why)] if tag == "error" => Err(why.clone()),
    _ => Err(
      "a step is [\"+\", relation, fields], [\"-\", relation, fields], [\"!\", message] or \
       [\"?\", relation]"
        .to_string(),
    ),
  }
}

/// A step as JSON ([`step`] reads it back).
pub fn step_json(prog: &Program, step: &Step) -> Json {
  let msg_ty = Ty::Named("Msg".into());
  let parts = match step {
    Step::Assert(rel, tuple) | Step::Retract(rel, tuple) => {
      let tag = if matches!(step, Step::Assert(..)) {
        "+"
      } else {
        "-"
      };
      let rel_name = Json::from(&*prog.rels[*rel].name);
      vec![Json::from(tag), rel_name, fact_json(prog, *rel, tuple)]
    }
    Step::Msg(msg) => vec![Json::from("!"), value_json(prog, &msg_ty, msg)],
    Step::Show(rel) => vec![Json::from("?"), Json::from(rel.as_str())],
  };
  Json::Array(parts)
}

/// A tick written as lines ([`read_step`]), as JSON steps: a step a line,
/// until a line that lem can't read, which becomes `["error", why]` and
/// ends them, as it ends the tick where Rust reads it.
pub fn script_json(prog: &mut Program, lines: &str) -> Json {
  let mut steps = Vec::new();
  for line in lines.lines() {
    match read_step(prog, line) {
      Ok(Some(step)) => steps.push(step_json(prog, &step)),
      Ok(None) => {}
      Err(why) => {
        steps.push(Json::Array(vec![Json::from("error"), Json::from(why)]));
        break;
      }
    }
  }
  Json::Array(steps)
}

/// Reads a tick given as JSON steps, and takes each as it's read
/// ([`Runtime::step`]): gives the message and the relations to show.
pub fn read_steps(runtime: &mut Runtime, steps: &Json) -> Result<Script, String> {
  let Json::Array(steps) = steps else {
    return Err(format!("a tick is an array of steps, not {}", steps.kind()));
  };
  let mut script = Script::default();
  for json in steps {
    let step = step(&runtime.prog, json)?;
    runtime.step(step, &mut script)?;
  }
  Ok(script)
}

/// Runs a tick given as JSON steps ([`read_steps`]), and gives what
/// [`Runtime::run`] gives.
pub fn run(runtime: &mut Runtime, steps: &Json) -> Result<String, String> {
  let script = read_steps(runtime, steps)?;
  runtime.run(&script)
}
