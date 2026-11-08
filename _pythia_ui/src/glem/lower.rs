//! glem's rewrite: a glem program as plain lem, which lem then checks and
//! runs. Spans stay as written, so that lem's errors point into the files.
//!
//! - `widget W(fields)` becomes `rel W(id: Id, fields, parent: Id,
//!   order: Any)`, and `surface S(fields)` becomes `rel S(id: Id, fields)`.
//!   A widget's own fields come before `parent` and `order`, so that its
//!   arguments by position are its id, then its own fields.
//! - A block's head is a widget or a surface. Each widget in the block gets
//!   `parent=`, the head's id, and `order=`, its place among the block's
//!   widgets as written, unless it gives them itself; the widgets of a `for`
//!   share a place. Everything in a block or a `for` gets its body, so lem
//!   sees neither, outside `on` blocks.
//! - Each widget kind gets a rule for `Parent`, which glem's prelude
//!   declares.
//! - `keep`, a flag of `state` declarations, is read and removed.

use crate::lem::check::LoadError;
use crate::lem::parse::{is_lower, Decl, Mod, Param, Sent, SentRef, SpanIdent, Term, TermRef};
use crate::lem::token::Span;
use std::collections::HashMap;
use std::panic::Location;
use std::rc::Rc;

/// What a `widget` or `surface` declaration declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
  /// A widget, which has a parent, and an order among its siblings.
  Widget,
  /// A surface, such as a window: the root of a tree.
  Surface,
}

impl Kind {
  pub fn word(self) -> &'static str {
    match self {
      Kind::Widget => "widget",
      Kind::Surface => "surface",
    }
  }
}

/// A program as plain lem, and what glem's runtime needs to know of it.
pub struct Lowered {
  /// The rewritten trees, one for each file.
  pub mods: Vec<Mod>,
  /// The widget kinds and surfaces, by name, in the order declared.
  pub kinds: Vec<(Rc<str>, Kind)>,
  /// The `state` relations declared `keep`.
  pub keep: Vec<Rc<str>>,
}

/// Rewrites parsed files as plain lem: `mods[i]` is the tree of `files[i]`,
/// whose name and text errors show.
pub fn lower(files: &[(&str, &str)], mods: &[Mod]) -> Result<Lowered, LoadError> {
  let mut lowerer = Lowerer::default();
  lowerer.run(mods).map_err(|e| {
    let (name, text) = files[e.file];
    LoadError {
      text: crate::lem::render(name, text, &e.span, &e.message),
      loc: e.loc,
    }
  })
}

/// An error before it's rendered.
struct LowerError {
  file: usize,
  span: Span,
  message: String,
  loc: &'static Location<'static>,
}

type Lower<T> = Result<T, LowerError>;

/// What a block or a `for` gives the sentences inside it.
#[derive(Clone, Default)]
struct Ctx {
  /// The bodies of the blocks and `for`s around them.
  body: Vec<TermRef>,
  /// The id of the widget or surface whose block this is.
  parent: Option<TermRef>,
  /// Whether they're inside a block or a `for`.
  inside: bool,
}

#[derive(Default)]
struct Lowerer {
  kinds: Vec<(Rc<str>, Kind)>,
  by_name: HashMap<Rc<str>, Kind>,
  /// Each widget kind's declaration, by file and span, for its `Parent` rule.
  widgets: Vec<(usize, Span, Rc<str>)>,
  keep: Vec<Rc<str>>,
  /// The file being rewritten.
  file: usize,
}

impl Lowerer {
  #[track_caller]
  fn fail<T>(&self, span: Span, message: impl Into<String>) -> Lower<T> {
    Err(LowerError {
      file: self.file,
      span,
      message: message.into(),
      loc: Location::caller(),
    })
  }

  fn run(&mut self, mods: &[Mod]) -> Lower<Lowered> {
    // The kinds first, since a block in any file may hold any kind.
    for (file, mod_) in mods.iter().enumerate() {
      self.file = file;
      for sent in &mod_.body {
        if let Sent::Decl(_, word, decls, _, flags) = &**sent {
          let kind = match &*word.text {
            "widget" => Kind::Widget,
            "surface" => Kind::Surface,
            _ => continue,
          };
          if let Some(flag) = flags.first() {
            let message = format!("a {} takes no flags, such as `{}`", kind.word(), flag.text);
            return self.fail(flag.span.clone(), message);
          }
          for decl in decls {
            self.kind(kind, decl, file)?;
          }
        }
      }
    }
    let mut out = Vec::new();
    for (file, mod_) in mods.iter().enumerate() {
      self.file = file;
      let mut body = Vec::new();
      // Each sentence outside a block starts counting widgets at 0.
      for sent in &mod_.body {
        self.sents(
          std::slice::from_ref(sent),
          &Ctx::default(),
          &mut 0,
          &mut body,
        )?;
      }
      out.push(Mod {
        span: mod_.span.clone(),
        body,
      });
    }
    for (file, span, name) in &self.widgets {
      out[*file].body.push(parent_rule(span, name));
    }
    Ok(Lowered {
      mods: out,
      kinds: self.kinds.clone(),
      keep: self.keep.clone(),
    })
  }

  /// A widget kind's or a surface's declaration.
  fn kind(&mut self, kind: Kind, decl: &Decl, file: usize) -> Lower<()> {
    let implicit: &[&str] = match kind {
      Kind::Widget => &["id", "parent", "order"],
      Kind::Surface => &["id"],
    };
    let mut names = decl.params.iter().filter_map(|p| p.name.as_ref());
    if let Some(name) = names
      .find(|n| implicit.contains(&&*n.text))
      .map(|n| &n.text)
    {
      let message = format!("every {} has a `{name}` of its own", kind.word());
      return self.fail(decl.span.clone(), message);
    }
    // A name declared twice is lem's error to give.
    self.kinds.push((decl.name.text.clone(), kind));
    self.by_name.insert(decl.name.text.clone(), kind);
    if kind == Kind::Widget {
      self
        .widgets
        .push((file, decl.span.clone(), decl.name.text.clone()));
    }
    Ok(())
  }

  /// Sentences inside `ctx`, rewritten into `out`. `pos` counts the widgets
  /// of the block around them.
  fn sents(
    &mut self,
    sents: &[SentRef],
    ctx: &Ctx,
    pos: &mut i64,
    out: &mut Vec<SentRef>,
  ) -> Lower<()> {
    for sent in sents {
      match &**sent {
        Sent::Just(span, term) => {
          let (rule, _) = self.rule(Some(sent), span, term, ctx, pos)?;
          out.push(rule);
        }
        Sent::Block(span, term, inner) => {
          let (rule, id) = self.rule(None, span, term, ctx, pos)?;
          let (head, body) = split(term);
          let Some(id) = id else {
            let message = "only a widget or a surface has a block of what's inside it";
            return self.fail(head.span(), message);
          };
          out.push(rule);
          let mut inner_ctx = Ctx {
            body: ctx.body.clone(),
            parent: Some(id),
            inside: true,
          };
          if let Some(body) = body {
            body_items(body, &mut inner_ctx.body);
          }
          self.sents(inner, &inner_ctx, &mut 0, out)?;
        }
        Sent::For(_, cond, inner) => {
          let mut inner_ctx = Ctx {
            body: ctx.body.clone(),
            parent: ctx.parent.clone(),
            inside: true,
          };
          body_items(cond, &mut inner_ctx.body);
          self.sents(inner, &inner_ctx, pos, out)?;
        }
        // lem refuses `del` outside `on` blocks.
        Sent::Del(..) => out.push(sent.clone()),
        Sent::On(span, ..) if ctx.inside => {
          return self.fail(span.clone(), "an `on` block can't be inside another block");
        }
        Sent::Type(span, ..) | Sent::Enum(span, ..) | Sent::Decl(span, ..) if ctx.inside => {
          return self.fail(span.clone(), "a declaration can't be inside a block");
        }
        Sent::Decl(..) => out.push(self.decl(sent)?),
        Sent::On(..) | Sent::Type(..) | Sent::Enum(..) => out.push(sent.clone()),
      }
    }
    Ok(())
  }

  /// A rule, or a block's head, inside `ctx`, as plain lem: with a widget's
  /// `parent` and `order`, and the body of each block and `for` around it.
  /// For a widget or a surface, also the term of its id, for its block.
  /// `same` is the sentence to give back if nothing changes.
  fn rule(
    &mut self,
    same: Option<&SentRef>,
    span: &Span,
    term: &TermRef,
    ctx: &Ctx,
    pos: &mut i64,
  ) -> Lower<(SentRef, Option<TermRef>)> {
    let (head, body) = split(term);
    let mut new_head = None;
    let mut id = None;
    if let Some((name, name_span, args)) = atom(head) {
      if let Some(&kind) = self.by_name.get(name) {
        id = args.iter().find_map(|arg| named_value(arg, "id"));
        id = id.or_else(|| args.first().filter(|arg| !is_named(arg)).cloned());
        if id.is_none() {
          return self.fail(name_span.clone(), format!("{name} needs an `id`"));
        }
        match kind {
          Kind::Surface if ctx.parent.is_some() => {
            let message = format!("{name} is a surface, so it can't be inside a block");
            return self.fail(name_span.clone(), message);
          }
          Kind::Surface => {}
          Kind::Widget => {
            let mut tup = vec![Rc::new(Term::Ident(name_span.clone(), name.clone()))];
            tup.extend(args.iter().cloned());
            if !args.iter().any(|arg| named_value(arg, "parent").is_some()) {
              let Some(parent) = &ctx.parent else {
                let message =
                  format!("{name} needs a `parent`: put it in a widget's block, or give `parent=`");
                return self.fail(name_span.clone(), message);
              };
              tup.push(named(name_span, "parent", parent.clone()));
            }
            if !args.iter().any(|arg| named_value(arg, "order").is_some()) {
              let order = Rc::new(Term::IntLit(name_span.clone(), pos.to_string().into()));
              tup.push(named(name_span, "order", order));
            }
            *pos += 1;
            new_head = Some(Rc::new(Term::Apply(head.span(), tup)));
          }
        }
      }
    }
    if let (Some(same), None, true) = (same, &new_head, ctx.body.is_empty()) {
      return Ok((same.clone(), id));
    }
    let head = new_head.unwrap_or_else(|| head.clone());
    let mut items = ctx.body.clone();
    if let Some(body) = body {
      body_items(body, &mut items);
    }
    let term = match items.len() {
      0 => head,
      1 => Rc::new(Term::Deduct(term.span(), head, items.remove(0))),
      _ => {
        let body = Rc::new(Term::Bunch(term.span(), items));
        Rc::new(Term::Deduct(term.span(), head, body))
      }
    };
    Ok((Rc::new(Sent::Just(span.clone(), term)), id))
  }

  /// A declaration as plain lem: a widget kind or a surface as a `rel`, and
  /// a `state` declaration without `keep`.
  fn decl(&mut self, sent: &SentRef) -> Lower<SentRef> {
    let Sent::Decl(span, word, decls, key, flags) = &**sent else {
      unreachable!("a declaration");
    };
    if matches!(&*word.text, "widget" | "surface") {
      let widget = &*word.text == "widget";
      let rel = SpanIdent {
        span: word.span.clone(),
        text: "rel".into(),
      };
      let decls = decls.iter().map(|decl| lower_decl(decl, widget)).collect();
      return Ok(Rc::new(Sent::Decl(
        span.clone(),
        rel,
        decls,
        key.clone(),
        Vec::new(),
      )));
    }
    let Some(keep) = flags.iter().find(|flag| &*flag.text == "keep") else {
      return Ok(sent.clone());
    };
    if &*word.text != "state" {
      return self.fail(
        keep.span.clone(),
        "`keep` is a flag of `state` declarations",
      );
    }
    self
      .keep
      .extend(decls.iter().map(|decl| decl.name.text.clone()));
    let flags = flags
      .iter()
      .filter(|flag| &*flag.text != "keep")
      .cloned()
      .collect();
    Ok(Rc::new(Sent::Decl(
      span.clone(),
      word.clone(),
      decls.clone(),
      key.clone(),
      flags,
    )))
  }
}

/// A sentence's head and body: `Head :- Body`, or `Head` alone.
fn split(term: &TermRef) -> (&TermRef, Option<&TermRef>) {
  match &**term {
    Term::Deduct(_, head, body) => (head, Some(body)),
    _ => (term, None),
  }
}

/// A body's items: the terms that commas join, and that groups hold.
fn body_items(term: &TermRef, out: &mut Vec<TermRef>) {
  match &**term {
    Term::Bunch(_, tup) => tup.iter().for_each(|t| body_items(t, out)),
    Term::Group(_, t) if matches!(&**t, Term::Bunch(..)) => body_items(t, out),
    _ => out.push(term.clone()),
  }
}

/// An atom's name, the name's span, and its arguments: `Name(args)`, or
/// `Name` alone.
fn atom(term: &Term) -> Option<(&Rc<str>, &Span, &[TermRef])> {
  match term {
    Term::Ident(span, name) if !is_lower(name) => Some((name, span, &[])),
    Term::Apply(_, tup) => match &*tup[0] {
      Term::Ident(span, name) if !is_lower(name) => Some((name, span, &tup[1..])),
      _ => None,
    },
    _ => None,
  }
}

/// The value of an argument `name=value`.
fn named_value(arg: &TermRef, name: &str) -> Option<TermRef> {
  match &**arg {
    Term::Equal(_, l, r) if matches!(&**l, Term::Ident(_, n) if &**n == name) => Some(r.clone()),
    _ => None,
  }
}

/// Whether an argument is given by name, as `name=value`.
fn is_named(arg: &TermRef) -> bool {
  matches!(&**arg, Term::Equal(_, l, _) if matches!(&**l, Term::Ident(_, n) if is_lower(n)))
}

/// The argument `name=value`.
fn named(span: &Span, name: &str, value: TermRef) -> TermRef {
  let field = Rc::new(Term::Ident(span.clone(), name.into()));
  Rc::new(Term::Equal(span.clone(), field, value))
}

/// A widget kind or a surface as a `rel`: `id`, its own fields, then a
/// widget's `parent` and `order`.
fn lower_decl(decl: &Decl, widget: bool) -> Decl {
  let span = &decl.name.span;
  let field = |name: &str, ty: &str| Param {
    span: span.clone(),
    name: Some(SpanIdent {
      span: span.clone(),
      text: name.into(),
    }),
    ty: SpanIdent {
      span: span.clone(),
      text: ty.into(),
    },
    default: None,
  };
  let mut params = vec![field("id", "Id")];
  params.extend(decl.params.iter().cloned());
  if widget {
    params.push(field("parent", "Id"));
    params.push(field("order", "Any"));
  }
  Decl {
    span: decl.span.clone(),
    name: decl.name.clone(),
    params,
  }
}

/// `Parent(id, parent) :- W(id=id, parent=parent)`, for the widget kind `W`
/// declared at `span`.
fn parent_rule(span: &Span, kind: &Rc<str>) -> SentRef {
  let name = |text: &str| Rc::new(Term::Ident(span.clone(), text.into()));
  let head = Term::Apply(
    span.clone(),
    vec![name("Parent"), name("id"), name("parent")],
  );
  let atom = vec![
    Rc::new(Term::Ident(span.clone(), kind.clone())),
    named(span, "id", name("id")),
    named(span, "parent", name("parent")),
  ];
  let term = Term::Deduct(
    span.clone(),
    Rc::new(head),
    Rc::new(Term::Apply(span.clone(), atom)),
  );
  Rc::new(Sent::Just(span.clone(), Rc::new(term)))
}
