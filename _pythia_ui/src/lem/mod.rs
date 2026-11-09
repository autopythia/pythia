//! lem: a typed, Datalog-like language with Elm's update. Rules derive facts
//! from facts, in strata, with negation and recursion; `on` blocks turn
//! messages into state changes and effects, as Elm's update does. It's the
//! core of pythia's GUI (`doc/pythia-ui-gui-design.md` in pythia), as a
//! prototype; layers above it add widgets.
//!
//! - [`token`]: the tokenizer;
//! - [`parse`]: the Pratt parser, after zinterp's in pythia's `_pythia/`;
//! - [`print`]: the printer, whose text parses to the same tree;
//! - [`value`]: values and types;
//! - [`check`]: the checker, which lowers files to a [`check::Program`];
//! - [`eval`]: the runtime, which runs a program's ticks;
//! - [`compile`]: the compiled form, a checked program as JSON, which lem's
//!   runtime in JavaScript, `lem.js`, runs too;
//! - [`host`]: the host's facts and messages, and ticks, as JSON.
//!
//! Every program loads lem's prelude ([`PRELUDE`]) first. The tests' files
//! are in the crate's `lem/test/`.
//!
//! Every error renders as Elm's do ([`render`]): the file, line and column,
//! what's wrong, then the line with carets under the span.

pub mod check;
pub mod clash;
#[cfg(test)]
mod clash_test;
pub mod compile;
#[cfg(test)]
mod compile_test;
pub mod eval;
#[cfg(test)]
mod eval_test;
pub mod host;
pub mod parse;
#[cfg(test)]
mod parse_test;
pub mod print;
#[cfg(test)]
pub(crate) mod testing;
pub mod token;
pub mod value;

use token::Span;

/// lem's prelude, which every program loads first: by its path in the
/// crate, and its text.
pub const PRELUDE: (&str, &str) = ("src/lem/prelude.lem", include_str!("prelude.lem"));

/// The 1-based line and column, in characters, of the byte offset `at`, and
/// that line's text, without its line break.
pub fn line_col(src: &str, at: usize) -> (usize, usize, &str) {
  let at = floor_char_boundary(src, at.min(src.len()));
  let start = src[..at].rfind('\n').map_or(0, |i| i + 1);
  let end = src[at..].find('\n').map_or(src.len(), |i| at + i);
  let line = 1 + src[..start].matches('\n').count();
  let col = 1 + src[start..at].chars().count();
  (line, col, src[start..end].trim_end_matches('\r'))
}

/// `name:line:col: message:`, then the span's first line, with carets under
/// the span (at least one, at most to the end of the line).
pub fn render(name: &str, src: &str, span: &Span, message: &str) -> String {
  let (line, col, text) = line_col(src, span.start);
  let lead: String = text
    .chars()
    .take(col - 1)
    .map(|c| if c == '\t' { '\t' } else { ' ' })
    .collect();
  let end = floor_char_boundary(src, span.end.min(src.len()));
  let shown = src[span.start.min(end)..end]
    .split('\n')
    .next()
    .unwrap_or("")
    .chars()
    .count();
  let width = shown.clamp(1, (text.chars().count() + 1).saturating_sub(col - 1).max(1));
  format!(
    "{name}:{line}:{col}: {message}:\n{text}\n{lead}{}",
    "^".repeat(width)
  )
}

/// The largest char boundary at or before `at` (str::floor_char_boundary is
/// newer than Rust 1.78).
fn floor_char_boundary(src: &str, mut at: usize) -> usize {
  while !src.is_char_boundary(at) {
    at -= 1;
  }
  at
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn errors_point_at_their_span() {
    let src = "a\n  Button(on=\"go\") :- X\n";
    let at = src.find("\"go\"").unwrap();
    assert_eq!(
      render("f.lem", src, &(at..at + 4), "this is Text"),
      "f.lem:2:13: this is Text:\n  Button(on=\"go\") :- X\n            ^^^^"
    );
    // An empty span at the end of a line still gets a caret.
    let end = src.find(" :-").unwrap();
    assert_eq!(line_col(src, end), (2, 18, "  Button(on=\"go\") :- X"));
    assert!(render("f.lem", src, &(end..end), "here").ends_with("\n                 ^"));
    // Tabs before the span stay tabs, so that the carets line up.
    let tabbed = "\tx é y";
    assert_eq!(
      render("t", tabbed, &(6..7), "y"),
      "t:1:6: y:\n\tx é y\n\t    ^"
    );
  }
}
