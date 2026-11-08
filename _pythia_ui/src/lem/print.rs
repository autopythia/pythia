//! Prints lem's trees as text, as zinterp's `Printer` does: a sentence a
//! line, blocks indented two spaces, arguments as `name=value`, and groups
//! where the tree has them, so that the text parses to the same tree.

use super::parse::{Decl, Mod, Param, Sent, Term};
use std::fmt::Write;

#[derive(Default)]
pub struct Printer {
  buf: String,
}

impl Printer {
  pub fn new() -> Printer {
    Printer::default()
  }

  /// The text printed so far.
  pub fn finish(self) -> String {
    self.buf
  }

  pub fn print_mod(&mut self, mod_: &Mod) {
    for sent in &mod_.body {
      self.sent(sent, 0);
    }
  }

  pub fn sent(&mut self, sent: &Sent, level: usize) {
    self.buf.push_str(&"  ".repeat(level));
    let body = match sent {
      Sent::Just(_, term) => {
        self.term(term);
        None
      }
      Sent::Block(_, head, body) => {
        self.term(head);
        Some(body)
      }
      Sent::For(_, cond, body) => {
        self.buf.push_str("for ");
        self.term(cond);
        Some(body)
      }
      Sent::On(_, pattern, body) => {
        self.buf.push_str("on ");
        self.term(pattern);
        Some(body)
      }
      Sent::Del(_, head) => {
        self.buf.push_str("del ");
        self.term(head);
        None
      }
      Sent::Type(_, name, ty) => {
        let _ = write!(self.buf, "type {} = {}", name.text, ty.text);
        None
      }
      Sent::Enum(_, name, ctors) => {
        let _ = writeln!(self.buf, "enum {}:", name.text);
        for ctor in ctors {
          self.buf.push_str(&"  ".repeat(level + 1));
          self.decl(ctor);
          self.buf.push('\n');
        }
        return;
      }
      Sent::Decl(_, word, decls, key, flags) => {
        self.buf.push_str(&word.text);
        for (i, decl) in decls.iter().enumerate() {
          self.buf.push_str(if i == 0 { " " } else { ", " });
          self.decl(decl);
        }
        for (i, column) in key.iter().enumerate() {
          self.buf.push_str(if i == 0 { " key " } else { ", " });
          self.buf.push_str(&column.text);
        }
        for flag in flags {
          self.buf.push(' ');
          self.buf.push_str(&flag.text);
        }
        None
      }
    };
    match body {
      None => self.buf.push('\n'),
      Some(body) => {
        self.buf.push_str(":\n");
        for sent in body {
          self.sent(sent, level + 1);
        }
      }
    }
  }

  pub fn decl(&mut self, decl: &Decl) {
    self.buf.push_str(&decl.name.text);
    if decl.params.is_empty() {
      return;
    }
    self.buf.push('(');
    for (i, param) in decl.params.iter().enumerate() {
      if i > 0 {
        self.buf.push_str(", ");
      }
      self.param(param);
    }
    self.buf.push(')');
  }

  fn param(&mut self, param: &Param) {
    if let Some(name) = &param.name {
      let _ = write!(self.buf, "{}: ", name.text);
    }
    self.buf.push_str(&param.ty.text);
    if let Some(default) = &param.default {
      self.buf.push_str(" = ");
      self.term(default);
    }
  }

  pub fn term(&mut self, term: &Term) {
    match term {
      Term::Ident(_, s) | Term::IntLit(_, s) | Term::FloatLit(_, s) => self.buf.push_str(s),
      Term::PlaceIdent(_) => self.buf.push('_'),
      Term::AtomLit(_, s) => quote(&mut self.buf, s),
      Term::LogicLit(_, b) => self.buf.push_str(if *b { "True" } else { "False" }),
      Term::Group(_, t) => {
        self.buf.push('(');
        self.term(t);
        self.buf.push(')');
      }
      Term::Bunch(_, tup) => {
        for (i, t) in tup.iter().enumerate() {
          if i > 0 {
            self.buf.push_str(", ");
          }
          self.term(t);
        }
      }
      Term::Not(_, t) => {
        self.buf.push_str("not ");
        self.term(t);
      }
      Term::Minus(_, t) => {
        // `--` would start a comment.
        self.buf.push('-');
        if matches!(&**t, Term::Minus(..)) {
          self.buf.push(' ');
        }
        self.term(t);
      }
      Term::Query(_, t) => {
        self.term(t);
        self.buf.push('?');
      }
      Term::Attr(_, t, name) => {
        self.term(t);
        self.buf.push('.');
        self.buf.push_str(&name.text);
      }
      Term::Deduct(_, l, r) => self.infix(l, " :- ", r),
      Term::Equal(_, l, r) => self.infix(l, " = ", r),
      Term::NEqual(_, l, r) => self.infix(l, " /= ", r),
      Term::Compare(_, cmp, l, r) => {
        self.term(l);
        let _ = write!(self.buf, " {} ", cmp.spelling());
        self.term(r);
      }
      Term::Add(_, l, r) => self.infix(l, " + ", r),
      Term::Sub(_, l, r) => self.infix(l, " - ", r),
      Term::Mul(_, l, r) => self.infix(l, " * ", r),
      Term::Div(_, l, r) => self.infix(l, " / ", r),
      Term::Apply(_, tup) => {
        self.term(&tup[0]);
        self.buf.push('(');
        for (i, arg) in tup[1..].iter().enumerate() {
          if i > 0 {
            self.buf.push_str(", ");
          }
          match &**arg {
            Term::Equal(_, name, value) if matches!(&**name, Term::Ident(..)) => {
              self.infix(name, "=", value)
            }
            arg => self.term(arg),
          }
        }
        self.buf.push(')');
      }
    }
  }

  fn infix(&mut self, l: &Term, op: &str, r: &Term) {
    self.term(l);
    self.buf.push_str(op);
    self.term(r);
  }
}

/// `s` as lem text: in `"`, with escapes for `"`, `\` and control
/// characters.
pub fn quote(buf: &mut String, s: &str) {
  buf.push('"');
  for c in s.chars() {
    match c {
      '"' => buf.push_str("\\\""),
      '\\' => buf.push_str("\\\\"),
      '\n' => buf.push_str("\\n"),
      '\r' => buf.push_str("\\r"),
      '\t' => buf.push_str("\\t"),
      c if c < ' ' || c == '\x7f' => {
        let _ = write!(buf, "\\u{:04x}", c as u32);
      }
      c => buf.push(c),
    }
  }
  buf.push('"');
}

/// A file's tree, printed.
pub fn print(mod_: &Mod) -> String {
  let mut printer = Printer::new();
  printer.print_mod(mod_);
  printer.finish()
}
