//! What the tests of lem, and of the layers above it, share: their files in
//! the crate's `lem/`, cases in those files, round trips through the
//! printer, and reports in TAP, as zinterp's `parse_test` writes them
//! (pythia's `_pythia/src/zinterp/parse_test.rs`): `ok 1 - "title"`, or
//! `not ok 2 - "title"` and why, then `1..N`. Trees compare as S-expressions
//! without spans ([`mod_sexp`]). And what the runtimes in JavaScript share:
//! their tests, which Rust writes ([`write_web`]), and a check of their
//! scripts ([`unsafe_in_script`]).

use super::check::LoadError;
use super::parse::{parse, Decl, Mod, Param, Sent, SentRef, Term};
use super::print::print;
use super::token::Span;
use crate::json::Value as Json;
use std::fmt::Write;
use std::path::Path;

/// lem's files: `lem/` in the crate.
const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/lem");

pub(crate) fn read(name: &str) -> String {
  let path = Path::new(DIR).join(name);
  std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A file's tree, a sentence a line, as S-expressions without spans.
pub(crate) fn mod_sexp(mod_: &Mod) -> String {
  let mut out = String::new();
  for sent in &mod_.body {
    sent_sexp(&mut out, sent);
    out.push('\n');
  }
  out
}

fn sent_sexp(out: &mut String, sent: &Sent) {
  let block = |out: &mut String, word: &str, head: &Term, body: &[SentRef]| {
    let _ = write!(out, "({word} ");
    term_sexp(out, head);
    for sent in body {
      out.push(' ');
      sent_sexp(out, sent);
    }
    out.push(')');
  };
  match sent {
    Sent::Just(_, term) => term_sexp(out, term),
    Sent::Block(_, head, body) => block(out, "block", head, body),
    Sent::For(_, cond, body) => block(out, "for", cond, body),
    Sent::On(_, pattern, body) => block(out, "on", pattern, body),
    Sent::Del(_, head) => list(out, "del", [&**head]),
    Sent::Type(_, name, ty) => {
      let _ = write!(out, "(type {} {})", name.text, ty.text);
    }
    Sent::Enum(_, name, ctors) => {
      let _ = write!(out, "(enum {}", name.text);
      for ctor in ctors {
        out.push(' ');
        decl_sexp(out, ctor);
      }
      out.push(')');
    }
    Sent::Decl(_, word, decls, key, flags) => {
      out.push('(');
      out.push_str(&word.text);
      for decl in decls {
        out.push(' ');
        decl_sexp(out, decl);
      }
      if !key.is_empty() {
        out.push_str(" (key");
        for column in key {
          let _ = write!(out, " {}", column.text);
        }
        out.push(')');
      }
      for flag in flags {
        let _ = write!(out, " {}", flag.text);
      }
      out.push(')');
    }
  }
}

fn decl_sexp(out: &mut String, decl: &Decl) {
  out.push('(');
  out.push_str(&decl.name.text);
  for param in &decl.params {
    out.push(' ');
    param_sexp(out, param);
  }
  out.push(')');
}

fn param_sexp(out: &mut String, param: &Param) {
  match &param.name {
    Some(name) => {
      let _ = write!(out, "({}: {}", name.text, param.ty.text);
    }
    None => {
      let _ = write!(out, "({}", param.ty.text);
    }
  }
  if let Some(default) = &param.default {
    out.push_str(" = ");
    term_sexp(out, default);
  }
  out.push(')');
}

fn list<'a>(out: &mut String, head: &str, items: impl IntoIterator<Item = &'a Term>) {
  out.push('(');
  out.push_str(head);
  for item in items {
    out.push(' ');
    term_sexp(out, item);
  }
  out.push(')');
}

fn term_sexp(out: &mut String, term: &Term) {
  match term {
    Term::Ident(_, s) | Term::IntLit(_, s) | Term::FloatLit(_, s) => out.push_str(s),
    Term::PlaceIdent(_) => out.push('_'),
    Term::AtomLit(_, s) => {
      let _ = write!(out, "{s:?}");
    }
    Term::LogicLit(_, b) => out.push_str(if *b { "True" } else { "False" }),
    Term::Group(_, t) => list(out, "group", [&**t]),
    Term::Bunch(_, tup) => list(out, "and", tup.iter().map(|t| &**t)),
    Term::Not(_, t) => list(out, "not", [&**t]),
    Term::Minus(_, t) => list(out, "neg", [&**t]),
    Term::Query(_, t) => list(out, "query", [&**t]),
    Term::Attr(_, t, name) => {
      out.push_str("(. ");
      term_sexp(out, t);
      let _ = write!(out, " {})", name.text);
    }
    Term::Deduct(_, l, r) => list(out, ":-", [&**l, &**r]),
    Term::Equal(_, l, r) => list(out, "=", [&**l, &**r]),
    Term::NEqual(_, l, r) => list(out, "/=", [&**l, &**r]),
    Term::Compare(_, cmp, l, r) => list(out, cmp.spelling(), [&**l, &**r]),
    Term::Add(_, l, r) => list(out, "+", [&**l, &**r]),
    Term::Sub(_, l, r) => list(out, "-", [&**l, &**r]),
    Term::Mul(_, l, r) => list(out, "*", [&**l, &**r]),
    Term::Div(_, l, r) => list(out, "/", [&**l, &**r]),
    Term::Apply(_, tup) => {
      out.push('(');
      for (i, t) in tup.iter().enumerate() {
        if i > 0 {
          out.push(' ');
        }
        term_sexp(out, t);
      }
      out.push(')');
    }
  }
}

/// Whether each term's span holds its children's, as error carets need.
fn spans_nest(term: &Term, outer: &Span) -> Result<(), String> {
  let span = term.span();
  if span.start < outer.start || span.end > outer.end || span.start > span.end {
    return Err(format!("{span:?}, outside {outer:?}: {term:?}"));
  }
  let children: Vec<&Term> = match term {
    Term::Group(_, t) | Term::Not(_, t) | Term::Minus(_, t) | Term::Query(_, t) => vec![t],
    Term::Attr(_, t, name) => {
      if name.span.start < span.start || name.span.end > span.end {
        return Err(format!("the name at {:?}, outside {span:?}", name.span));
      }
      vec![t]
    }
    Term::Deduct(_, l, r)
    | Term::Equal(_, l, r)
    | Term::NEqual(_, l, r)
    | Term::Compare(_, _, l, r)
    | Term::Add(_, l, r)
    | Term::Sub(_, l, r)
    | Term::Mul(_, l, r)
    | Term::Div(_, l, r) => vec![l, r],
    Term::Bunch(_, tup) | Term::Apply(_, tup) => tup.iter().map(|t| &**t).collect(),
    _ => vec![],
  };
  children.into_iter().try_for_each(|t| spans_nest(t, &span))
}

fn sent_spans_nest(sent: &Sent) -> Result<(), String> {
  let span = sent.span();
  match sent {
    Sent::Just(_, head) | Sent::Del(_, head) => spans_nest(head, &span),
    Sent::Block(_, head, body) | Sent::For(_, head, body) | Sent::On(_, head, body) => {
      spans_nest(head, &span)?;
      body.iter().try_for_each(|sent| sent_spans_nest(sent))
    }
    Sent::Decl(_, _, decls, ..) => decls.iter().try_for_each(|decl| {
      if decl.span.start < span.start || decl.span.end > span.end {
        return Err(format!("{:?}, outside {span:?}", decl.span));
      }
      decl
        .params
        .iter()
        .try_for_each(|param| match &param.default {
          Some(default) => spans_nest(default, &param.span),
          None => Ok(()),
        })
    }),
    Sent::Type(..) | Sent::Enum(..) => Ok(()),
  }
}

/// Parses a file, prints it, and parses and prints that again: the same
/// tree, and the same text. Each term's span also holds its children's, as
/// error carets need.
pub(crate) fn round_trip(name: &str, src: &str) -> Result<(), String> {
  let tree = parse(src).map_err(|e| e.render(name, src))?;
  tree
    .body
    .iter()
    .try_for_each(|sent| sent_spans_nest(sent))?;
  let text = print(&tree);
  let again = parse(&text).map_err(|e| {
    let error = e.render("printed", &text);
    format!("the printed text doesn't parse:\n{error}\n{text}")
  })?;
  let (want, got) = (mod_sexp(&tree), mod_sexp(&again));
  if got != want {
    return Err(format!("the printed text parses to\n{got}not to\n{want}"));
  }
  let text_again = print(&again);
  if text_again != text {
    return Err(format!("printed again, it's\n{text_again}not\n{text}"));
  }
  Ok(())
}

/// The cases of a file in `lem/`: a title (after `=== `), then lines up to
/// `---`, then what they should give, up to the next case.
pub(crate) fn cases(name: &str) -> Vec<(String, String, String)> {
  let text = read(name);
  let mut lines = text
    .lines()
    .skip_while(|line| !line.starts_with("=== "))
    .peekable();
  let mut cases = Vec::new();
  while let Some(title) = lines.next() {
    let title = title.trim_start_matches("=== ").to_string();
    let mut src = String::new();
    for line in lines.by_ref().take_while(|line| *line != "---") {
      src.push_str(line);
      src.push('\n');
    }
    let mut want = String::new();
    while let Some(line) = lines.next_if(|line| !line.starts_with("=== ")) {
      want.push_str(line);
      want.push('\n');
    }
    cases.push((title, src, want.trim_end().to_string()));
  }
  cases
}

/// Reports results in TAP, each a title and, for a failure, why it failed;
/// and fails if any did.
pub(crate) fn report(results: Vec<(String, Option<String>)>) {
  let mut tap = String::new();
  let failed = results.iter().filter(|(_, why)| why.is_some()).count();
  for (rank, (title, why)) in results.iter().enumerate() {
    match why {
      None => {
        let _ = writeln!(tap, "ok {} - {title:?}", rank + 1);
      }
      Some(why) => {
        let _ = writeln!(tap, "not ok {} - {title:?}", rank + 1);
        for line in why.lines() {
          let _ = writeln!(tap, "# {line}");
        }
      }
    }
  }
  let _ = writeln!(tap, "1..{}", results.len());
  print!("{tap}");
  assert_eq!(failed, 0, "\n{tap}");
}

/// Checks a file of error cases: each case's source, loaded as `t.lem`,
/// fails with its error, raised from the code in one of `dirs`.
pub(crate) fn error_cases(name: &str, dirs: &[&str], load: impl Fn(&str) -> Option<LoadError>) {
  let cases = cases(name);
  assert!(cases.len() > 5, "too few cases in {name}");
  let mut results = Vec::new();
  for (title, src, want) in cases {
    let why = match load(&src) {
      None => Some("it loads".to_string()),
      Some(e) if e.text != want => Some(format!("the error is\n{}\nnot\n{want}", e.text)),
      Some(e)
        if !dirs
          .iter()
          .any(|dir| e.loc.file().replace('\\', "/").contains(dir)) =>
      {
        Some(format!("raised at {}, outside {dirs:?}", e.loc))
      }
      Some(_) => None,
    };
    results.push((title, why));
  }
  report(results);
}

/// Runs a file of ticks: each case is a tick, written as lines, that runs
/// on what the cases before it left, and gives what `tick` gives for it,
/// or `error: …`.
pub(crate) fn tick_cases(name: &str, mut tick: impl FnMut(&str) -> Result<String, String>) {
  let cases = cases(name);
  assert!(cases.len() > 3, "too few cases in {name}");
  let mut results = Vec::new();
  for (title, lines, want) in cases {
    let got = tick(&lines).unwrap_or_else(|e| format!("error: {e}"));
    let why = (got != want).then(|| format!("the tick gives\n{got}\nnot\n{want}"));
    results.push((title, why));
  }
  report(results);
}

/// Writes `var NAME = value;` to `target/web/FILE`, for the tests in
/// JavaScript, which `make js-test` runs: in ASCII, each other character as
/// `\\uXXXX`, so that every engine reads it alike. `by` names the test that
/// writes it.
pub(crate) fn write_web(file: &str, name: &str, by: &str, value: &Json) {
  let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/target/web");
  std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{dir}: {e}"));
  let mut json = String::new();
  for c in String::from_utf8(value.to_vec())
    .expect("JSON text is UTF-8")
    .chars()
  {
    match c.is_ascii() {
      true => json.push(c),
      false => {
        for unit in c.encode_utf16(&mut [0; 2]) {
          let _ = write!(json, "\\u{unit:04x}");
        }
      }
    }
  }
  let text = format!("// Tests in JavaScript, written by {by}.\nvar {name} = {json};\n");
  let path = format!("{dir}/{file}");
  std::fs::write(&path, text).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// What a runtime's script mustn't have: anything but ASCII, so that every
/// engine reads it alike; ways to write HTML; and ways to run code that it's
/// given. Gives what it finds.
pub(crate) fn unsafe_in_script(js: &str) -> Vec<String> {
  let mut found: Vec<String> = Vec::new();
  if !js.is_ascii() {
    found.push("a character beyond ASCII".to_string());
  }
  for word in [
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
  ] {
    if js.contains(word) {
      found.push(word.to_string());
    }
  }
  for name in ["eval", "Function"] {
    if calls(js, name) {
      found.push(format!("{name}("));
    }
  }
  found
}

/// Whether `js` calls `name`: the name, not the end of a longer one, then
/// `(`.
fn calls(js: &str, name: &str) -> bool {
  js.match_indices(name).any(|(at, _)| {
    let before = js[..at].chars().next_back();
    let after = js[at + name.len()..].trim_start().chars().next();
    let joined = before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$');
    !joined && after == Some('(')
  })
}
