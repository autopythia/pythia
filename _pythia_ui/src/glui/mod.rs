//! glui: widgets and themes for glem ([`crate::glem`]), the design language
//! of pythia's GUI (`doc/pythia-ui-gui-design.md` in pythia). It's a library
//! in lem, with no syntax or meaning of its own: declarations that apps use,
//! in files that a program loads after glem's prelude ([`FILES`]):
//!
//! - `widgets.lem`: the surfaces, the widget kinds and their flags, the
//!   messages that widgets send, and `Disabled`, which children inherit;
//! - `theme.lem`: what every theme gives, the colors, sizes and times that a
//!   front end draws the widgets with, which rules never read;
//! - `themes/`: the themes, `dark`, the default, and `light`, after wui's
//!   log styles ([`THEMES`]). A program loads one ([`load`]); another can
//!   come from a file ([`theme`]), and [`complete`] checks that a theme
//!   gives every value.
//!
//! Front ends will draw each kind with a recipe. For now the outline shows
//! the widgets, with glui's notes ([`notes`]).

#[cfg(test)]
mod tests;

use crate::glem;
use crate::lem::check::LoadError;
use crate::lem::eval;
use crate::lem::value::Value;

/// glui's files, which a program loads after glem's prelude: (name, text).
pub const FILES: [(&str, &str); 2] = [
  ("src/glui/widgets.lem", include_str!("widgets.lem")),
  ("src/glui/theme.lem", include_str!("theme.lem")),
];

/// glui's themes, the default first: (name, file, text).
pub const THEMES: [(&str, &str, &str); 2] = [
  (
    "dark",
    "src/glui/themes/dark.lem",
    include_str!("themes/dark.lem"),
  ),
  (
    "light",
    "src/glui/themes/light.lem",
    include_str!("themes/light.lem"),
  ),
];

/// A theme, by its name in [`THEMES`] or its file, `FILE.lem`: the file's
/// name, and its text.
pub fn theme(theme: &str) -> Result<(String, String), String> {
  if let Some((_, file, text)) = THEMES.iter().find(|(name, ..)| *name == theme) {
    return Ok((file.to_string(), text.to_string()));
  }
  if !theme.ends_with(".lem") {
    return Err(format!(
      "there's no theme {theme:?}: the themes are dark and light, or a FILE.lem"
    ));
  }
  match std::fs::read_to_string(theme) {
    Ok(text) => Ok((theme.to_string(), text)),
    Err(error) => Err(format!("{theme}: {error}")),
  }
}

/// Loads files, given as (name, text) pairs, after glem's prelude, glui's
/// files and a theme, (name, text), as [`theme`] gives it.
pub fn load(theme: (&str, &str), files: &[(&str, &str)]) -> Result<glem::Program, LoadError> {
  let mut all = FILES.to_vec();
  all.push(theme);
  all.extend_from_slice(files);
  glem::load(&all)
}

/// Whether the theme that a program loaded gives every paint a color and
/// every metric a size, and whether every tone has its paint; else what's
/// missing. It runs a tick, which derives the theme's facts.
pub fn complete(runtime: &mut glem::Runtime) -> Result<(), String> {
  runtime.tick(None)?;
  let mut lacks = Vec::new();
  for (rel, ty) in [
    ("Color", "Paint"),
    ("Size", "Metric"),
    ("TonePaint", "Tone"),
  ] {
    let given = runtime.lem.facts(rel);
    for ctor in runtime.lem.prog.ctors.iter().filter(|c| &*c.ty == ty) {
      if !given
        .iter()
        .any(|tuple| matches!(&tuple[0], Value::Ctor(name, _) if *name == ctor.name))
      {
        lacks.push(format!("{rel}({})", ctor.name));
      }
    }
  }
  match lacks.is_empty() {
    true => Ok(()),
    false => Err(format!("the theme gives no {}", lacks.join(", "))),
  }
}

/// glui's notes on a widget, for the outline ([`glem::Notes`]): why it's
/// disabled, if it is.
pub fn notes(runtime: &eval::Runtime, id: &Value) -> Vec<String> {
  let why: Vec<String> = runtime
    .facts("Disabled")
    .iter()
    .filter(|tuple| tuple[0] == *id)
    .map(|tuple| match &tuple[1] {
      Value::Text(why) => why.to_string(),
      why => why.to_string(),
    })
    .collect();
  if why.is_empty() {
    return Vec::new();
  }
  vec![format!("disabled: {}", why.join("; "))]
}
