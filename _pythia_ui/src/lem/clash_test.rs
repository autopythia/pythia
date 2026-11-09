//! The warnings of [`super::clash`]: each case of
//! `lem/test/clash_warnings.txt` is a file, `t.lem`, that loads after lem's
//! prelude, and the warnings that its keys give, or `(none)`.

use super::check::load;
use super::clash::clashes;
use super::testing::{cases, report};

#[test]
fn clash_warnings_read_as_written() {
  let cases = cases("test/clash_warnings.txt");
  assert!(cases.len() > 5, "too few cases");
  let mut results = Vec::new();
  for (title, src, want) in cases {
    let got = match load(&[("t.lem", &src)]) {
      Ok(prog) => match clashes(&prog, &[], &[]) {
        found if found.is_empty() => "(none)".to_string(),
        found => found.join("\n"),
      },
      Err(e) => format!("error: {}", e.text),
    };
    let why = (got != want).then(|| format!("the check gives\n{got}\nnot\n{want}"));
    results.push((title, why));
  }
  report(results);
}
