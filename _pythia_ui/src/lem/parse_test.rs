//! lem's file-driven parse tests, reported in TAP ([`super::testing`]):
//!
//! - each file that `lem/test/parse.txt` lists parses, and prints as text
//!   that parses to the same tree, which prints the same again;
//! - each case in `lem/test/parse_errors.txt` fails with its error, as lem
//!   shows it, raised from lem's own code.
//!
//! The tests of binding powers and blocks read trees as S-expressions.

use super::parse::parse;
use super::testing::{cases, mod_sexp, read, report, round_trip};

#[test]
fn listed_files_parse_and_print_back() {
  let list = read("test/parse.txt");
  let keys: Vec<&str> = list
    .lines()
    .map(str::trim)
    .filter(|line| !line.is_empty() && !line.starts_with('#'))
    .collect();
  assert!(!keys.is_empty(), "test/parse.txt lists no files");
  let mut results = vec![(
    super::PRELUDE.0.to_string(),
    round_trip(super::PRELUDE.0, super::PRELUDE.1).err(),
  )];
  for key in keys {
    results.push((key.to_string(), round_trip(key, &read(key)).err()));
  }
  report(results);
}

#[test]
fn parse_errors_read_as_written() {
  let cases = cases("test/parse_errors.txt");
  assert!(cases.len() > 10, "too few cases in test/parse_errors.txt");
  let mut results = Vec::new();
  for (title, src, want) in cases {
    let why = match parse(&src) {
      Ok(tree) => Some(format!("it parses, as\n{}", mod_sexp(&tree))),
      Err(e) if e.render("t.lem", &src) != want => {
        let got = e.render("t.lem", &src);
        Some(format!("the error is\n{got}\nnot\n{want}"))
      }
      Err(e) if !e.loc.file().replace('\\', "/").contains("src/lem/") => {
        Some(format!("raised at {}, outside src/lem/", e.loc))
      }
      Err(_) => None,
    };
    results.push((title, why));
  }
  report(results);
}

/// One source's tree, without its last line break.
fn tree(src: &str) -> String {
  let mod_ = parse(src).unwrap_or_else(|e| panic!("{}", e.render("t.lem", src)));
  mod_sexp(&mod_).trim_end().to_string()
}

#[test]
fn binding_powers_follow_zinterp() {
  for (src, want) in [
    ("X :- A, B", "(:- X (and A B))"),
    ("X :- not A, B", "(:- X (and (not A) B))"),
    ("X :- not a = b", "(:- X (not (= a b)))"),
    ("X :- (A, B), C", "(:- X (and (group (and A B)) C))"),
    ("A, B :- C", "(:- (and A B) C)"),
    ("X(n=a + b * c - d)", "(X (= n (- (+ a (* b c)) d)))"),
    (
      "X(n=-a * b, m=a / -b)",
      "(X (= n (* (neg a) b)) (= m (/ a (neg b))))",
    ),
    ("X(f=(e - t) / n)", "(X (= f (/ (group (- e t)) n)))"),
    ("X :- a - b - c = d", "(:- X (= (- (- a b) c) d))"),
    ("X :- x /= y, x <= 2", "(:- X (and (/= x y) (<= x 2)))"),
    ("X :- a.b.c(d)", "(:- X ((. (. a b) c) d))"),
    ("Q(x)?", "(query (Q x))"),
  ] {
    assert_eq!(tree(src), want, "{src}");
  }
}

#[test]
fn blocks_nest_by_indent() {
  let src = "A :- B():\n  for C(x), D(x, _):\n    E(x)\n\n  F\non M(x):\n  S(x) :- T\n  del U(x)\n";
  assert_eq!(
    tree(src),
    "(block (:- A (B)) (for (and (C x) (D x _)) (E x)) F)\n(on (M x) (:- (S x) T) (del (U x)))"
  );
}

#[test]
fn declarations_and_words_by_their_place() {
  let src = "type T = Text\nenum E:\n  A, B(Int)\n  C(x: Num = 1)\n\
             state S(key: T, on: Int) key key keep\nmsg M, N(on: Msg)\non M:\n  X(on=key)\n";
  assert_eq!(
    tree(src),
    "(type T Text)\n(enum E (A) (B (Int)) (C (x: Num = 1)))\n\
     (state (S (key: T) (on: Int)) (key key) keep)\n(msg (M) (N (on: Msg)))\n(on M (X (= on key)))"
  );
}

#[test]
fn any_word_declares_and_any_flag_follows() {
  // The parser gives no word or flag a meaning: the checker, or a layer
  // above lem, does.
  let src = "widget W(text: Text)
surface S keep x
state D(a: Int) key a z
";
  assert_eq!(
    tree(src),
    "(widget (W (text: Text)))\n(surface (S) keep x)\n(state (D (a: Int)) (key a) z)"
  );
}
