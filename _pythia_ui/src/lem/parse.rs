//! lem's parser: a Pratt parser after zinterp's (`_pythia/src/zinterp/parse.rs`
//! in pythia), with its names. [`Term`] and [`Sent`] variants carry their
//! [`Span`] first; [`Parser::term`] calls [`Parser::term_nud`] on the first
//! token, then [`Parser::term_led`] while the next token's [`Parser::lbp`] is
//! greater than the context's; and a [`ParseSpanError`] carries the parser's
//! own line that raised it.
//!
//! zinterp's binding powers stay: `:-` 200, `=` and `/=` 220, `(` 400, `?`
//! 1200 and `.` 1600. lem's fit between them: `,` 205 (a conjunction), `<`
//! `<=` `>` `>=` 220, `+` and `-` 300, and `*` and `/` 320. Prefix `not` reads
//! its operand at 215, so `not A, B` is `(not A), B` and `not x = y` is
//! `not (x = y)`; prefix `-` reads at 330. Arguments read at 205, so that a
//! comma ends one (zinterp's comma has no binding power yet). Comparisons
//! don't chain.
//!
//! Sentences are lines, as in Python, and one that ends with `:` opens a
//! block of more-indented lines. `type`, `enum` and `on` start sentences of
//! their own. Any other sentence that starts with two names is a
//! declaration: a word, such as `state`, then declarations, the `key`
//! columns, and flags, such as `keep`. Which words and flags mean something
//! is up to the checker (and to a layer above lem, which may add its own).
//! These words are names that the parser knows by their place, since a field
//! may be called `on` or `key`.

use super::token::{tokenize, Lit, RawIndent, Span, SpanToken, Token};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::panic::Location;
use std::rc::Rc;

pub type RawBp = u16;
pub type TermRef = Rc<Term>;
pub type SentRef = Rc<Sent>;

/// How deeply terms and blocks may nest.
pub const MAX_DEPTH: u16 = 128;

/// What's wrong, at a [`ParseSpanError`]'s span.
#[derive(Clone, Debug, PartialEq)]
pub enum ParseError {
  /// A character that no token starts with.
  BadChar(char),
  /// A `\` that isn't the last thing on its line.
  StrayBackslash,
  /// Text whose closing quote is missing: `"` closes on its own line.
  UnclosedStr,
  /// An escape that zinterp's strings don't have, or a broken `\u`.
  BadEscape,
  /// A control character in text, which needs an escape.
  ControlInStr,
  /// A number that's too large, or that runs into a name, as `12px` does.
  BadNumber,
  /// A bracket still open at the end of the file.
  Unclosed(char),
  /// Something else was expected here: what, and the token found.
  Expected(&'static str, Token),
  /// zinterp's operator or keyword, which lem doesn't use.
  Reserved(&'static str),
  /// A line indented more than the one before, which doesn't open a block.
  Indent,
  /// A line indented less than the one before, but not as any line above.
  Dedent,
  /// A `:` at the end of a line, without more-indented lines after it.
  EmptyBlock,
  /// A comparison of a comparison, such as `a < b < c`.
  Chained,
  /// Brackets or blocks nested more than [`MAX_DEPTH`] deep.
  TooDeep,
  /// A declaration's `key` after its flags, or a second `key`.
  KeyAfterFlags,
}

impl Display for ParseError {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    match self {
      ParseError::BadChar(c) => write!(f, "{c:?} isn't part of any token"),
      ParseError::StrayBackslash => {
        write!(
          f,
          "a `\\` joins the next line to this one, so it must end the line"
        )
      }
      ParseError::UnclosedStr => {
        write!(
          f,
          "this text has no closing `\"` on its line (`\"\"\"` spans lines)"
        )
      }
      ParseError::BadEscape => write!(
        f,
        "the escapes are \\\" \\\\ \\/ \\b \\f \\n \\r \\t and \\u with 4 hex digits"
      ),
      ParseError::ControlInStr => write!(f, "a control character in text needs an escape"),
      ParseError::BadNumber => {
        write!(f, "a number must fit in 64 bits, and end before any letter")
      }
      ParseError::Unclosed(c) => {
        let close = match c {
          '(' => ')',
          '[' => ']',
          _ => '}',
        };
        write!(f, "this `{c}` has no matching `{close}`")
      }
      ParseError::Expected(want, found) => {
        write!(f, "expected {want}, not {}", describe(found))
      }
      ParseError::Reserved(s) => match *s {
        "!" => write!(f, "`!` is reserved: lem negates with `not`"),
        "==" => write!(f, "`==` is reserved: lem compares with `=`"),
        ":=" => write!(f, "`:=` is reserved: a rule is `Head :- Body`"),
        "[" | "]" | "{" | "}" => write!(f, "`{s}` is reserved: lem has no lists or sets yet"),
        s if s.starts_with(|c: char| c.is_ascii_alphabetic()) => {
          write!(f, "`{s}` is a reserved word")
        }
        s => write!(f, "`{s}` is a reserved operator"),
      },
      ParseError::Indent => write!(
        f,
        "this line is indented more than the line before, which doesn't open a block"
      ),
      ParseError::Dedent => write!(f, "this line's indent matches no line above it"),
      ParseError::EmptyBlock => write!(
        f,
        "this `:` opens a block, so the lines after it must be indented more"
      ),
      ParseError::Chained => write!(f, "comparisons don't chain: write `a < b, b < c`"),
      ParseError::TooDeep => write!(f, "this nests more than {MAX_DEPTH} deep"),
      ParseError::KeyAfterFlags => write!(f, "`key` comes once, before any flags"),
    }
  }
}

/// A [`ParseError`] at a span. `loc` is the parser's own line that raised it,
/// as zinterp's errors carry, for debugging the parser.
#[derive(Clone, Debug)]
pub struct ParseSpanError {
  pub span: Span,
  pub err: ParseError,
  pub loc: &'static Location<'static>,
}

impl ParseSpanError {
  /// The error at `span`, from the parser's line that calls this (or that
  /// calls the `#[track_caller]` function that does). A method rather than
  /// zinterp's `From`, since `.into()` would put the line in core, on some
  /// Rust versions.
  #[track_caller]
  pub fn new(span: Span, err: ParseError) -> ParseSpanError {
    ParseSpanError {
      span,
      err,
      loc: Location::caller(),
    }
  }

  /// The error as lem shows it ([`super::render`]), for the file `name`
  /// whose text is `src`.
  pub fn render(&self, name: &str, src: &str) -> String {
    super::render(name, src, &self.span, &self.err.to_string())
  }
}

#[track_caller]
fn error<T>(span: Span, err: ParseError) -> Result<T, ParseSpanError> {
  Err(ParseSpanError::new(span, err))
}

/// Describes a token for an error message: "`:-`", "the name `x`".
pub fn describe(tok: &Token) -> String {
  match tok {
    Token::Indent(_) => "an indent".to_string(),
    Token::NL => "the end of the line".to_string(),
    Token::_Eof => "the end of the file".to_string(),
    Token::IntLit(s) | Token::FloatLit(s) => format!("the number {s}"),
    Token::AtomLit(s) => format!("the text {s:?}"),
    Token::PlaceIdent(s) | Token::Ident(s) => format!("the name `{s}`"),
    tok => format!("`{}`", tok.spelling()),
  }
}

/// A name, where it's written.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanIdent {
  pub span: Span,
  pub text: Lit,
}

/// An order comparison's operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
  Lt,
  LtEq,
  Gt,
  GtEq,
}

impl Cmp {
  pub fn spelling(self) -> &'static str {
    match self {
      Cmp::Lt => "<",
      Cmp::LtEq => "<=",
      Cmp::Gt => ">",
      Cmp::GtEq => ">=",
    }
  }
}

#[derive(Debug)]
pub enum Term {
  /// A name: a variable (lowercase), or a relation, type or constructor
  /// (capitalized).
  Ident(Span, Lit),
  /// `_`, which matches anything.
  PlaceIdent(Span),
  /// Text, with its escapes decoded.
  AtomLit(Span, Lit),
  /// `True` or `False`.
  LogicLit(Span, bool),
  /// A whole number, as written.
  IntLit(Span, Lit),
  /// A number with a fraction, as written.
  FloatLit(Span, Lit),
  /// `(t)`.
  Group(Span, TermRef),
  /// `a, b, …`: a conjunction.
  Bunch(Span, Vec<TermRef>),
  /// `not t`.
  Not(Span, TermRef),
  /// `-t`.
  Minus(Span, TermRef),
  /// `t?`, a query.
  Query(Span, TermRef),
  /// `t.name`.
  Attr(Span, TermRef, SpanIdent),
  /// `head :- body`.
  Deduct(Span, TermRef, TermRef),
  /// `a = b`; in arguments, `name=value`.
  Equal(Span, TermRef, TermRef),
  /// `a /= b`.
  NEqual(Span, TermRef, TermRef),
  /// `a < b` and the other orderings.
  Compare(Span, Cmp, TermRef, TermRef),
  Add(Span, TermRef, TermRef),
  Sub(Span, TermRef, TermRef),
  Mul(Span, TermRef, TermRef),
  Div(Span, TermRef, TermRef),
  /// `f(a, …)`: the head, then the arguments.
  Apply(Span, Vec<TermRef>),
}

impl Term {
  pub fn span(&self) -> Span {
    match self {
      Term::Ident(span, ..)
      | Term::PlaceIdent(span)
      | Term::AtomLit(span, ..)
      | Term::LogicLit(span, ..)
      | Term::IntLit(span, ..)
      | Term::FloatLit(span, ..)
      | Term::Group(span, ..)
      | Term::Bunch(span, ..)
      | Term::Not(span, ..)
      | Term::Minus(span, ..)
      | Term::Query(span, ..)
      | Term::Attr(span, ..)
      | Term::Deduct(span, ..)
      | Term::Equal(span, ..)
      | Term::NEqual(span, ..)
      | Term::Compare(span, ..)
      | Term::Add(span, ..)
      | Term::Sub(span, ..)
      | Term::Mul(span, ..)
      | Term::Div(span, ..)
      | Term::Apply(span, ..) => span.clone(),
    }
  }

  fn is_comparison(&self) -> bool {
    matches!(self, Term::Equal(..) | Term::NEqual(..) | Term::Compare(..))
  }
}

/// A declared field: `name: Type`, or a type alone, with a default after `=`.
#[derive(Clone, Debug)]
pub struct Param {
  pub span: Span,
  pub name: Option<SpanIdent>,
  pub ty: SpanIdent,
  pub default: Option<TermRef>,
}

/// `Name` or `Name(fields)`: a relation or a constructor.
#[derive(Clone, Debug)]
pub struct Decl {
  pub span: Span,
  pub name: SpanIdent,
  pub params: Vec<Param>,
}

#[derive(Debug)]
pub enum Sent {
  /// A rule, `Head :- Body`, or a fact, `Head`.
  Just(Span, TermRef),
  /// `Head:` or `Head :- Body:`, then the head's children.
  Block(Span, TermRef, Vec<SentRef>),
  /// `for Body:`, then sentences that each get the body.
  For(Span, TermRef, Vec<SentRef>),
  /// `on Message:`, then what the message does.
  On(Span, TermRef, Vec<SentRef>),
  /// `del Head` or `del Head :- Body`, in an `on` block.
  Del(Span, TermRef),
  /// `type Name = Type`.
  Type(Span, SpanIdent, SpanIdent),
  /// `enum Name:`, then its constructors, one or more a line.
  Enum(Span, SpanIdent, Vec<Decl>),
  /// A declaration: its word, such as `state`; one or more declarations;
  /// the `key` columns; and flags, such as `keep`.
  Decl(Span, SpanIdent, Vec<Decl>, Vec<SpanIdent>, Vec<SpanIdent>),
}

impl Sent {
  pub fn span(&self) -> Span {
    match self {
      Sent::Just(span, ..)
      | Sent::Block(span, ..)
      | Sent::For(span, ..)
      | Sent::On(span, ..)
      | Sent::Del(span, ..)
      | Sent::Type(span, ..)
      | Sent::Enum(span, ..)
      | Sent::Decl(span, ..) => span.clone(),
    }
  }
}

/// A file's sentences.
#[derive(Debug)]
pub struct Mod {
  pub span: Span,
  pub body: Vec<SentRef>,
}

#[derive(Clone, Copy, Debug)]
pub struct TermCtx {
  pub depth: u16,
  pub bp: RawBp,
}

/// The indentation that a block's lines need: the same as the first's, or,
/// for the first, more than the line that opened the block.
#[derive(Clone, Copy, Debug)]
pub enum SentIndent {
  Eq(RawIndent),
  Gt(RawIndent),
}

#[derive(Clone, Copy, Debug)]
pub struct SentCtx {
  pub depth: u16,
  pub indent: SentIndent,
}

impl SentCtx {
  fn term(&self, bp: RawBp) -> TermCtx {
    TermCtx {
      depth: self.depth + 1,
      bp,
    }
  }
}

/// Whether a name starts with a lowercase letter, as variables, declaration
/// words and flags do.
pub fn is_lower(name: &str) -> bool {
  name.starts_with(|c: char| c.is_ascii_lowercase())
}

/// Parses one file.
pub fn parse(buf: &str) -> Result<Mod, ParseSpanError> {
  Parser::new(buf)?.mod_()
}

pub struct Parser {
  tokens: Vec<SpanToken>,
  /// The next token's index. The last token is `_Eof`, which stays.
  pos: usize,
  /// Where the last token taken ends: where a term's span ends.
  end: usize,
  /// Whether the last block ended at a line indented less than its own:
  /// then a line indented more than the enclosing block's is a bad dedent.
  dedent: bool,
}

impl Parser {
  pub fn new(buf: &str) -> Result<Parser, ParseSpanError> {
    Ok(Parser {
      tokens: tokenize(buf)?,
      pos: 0,
      end: 0,
      dedent: false,
    })
  }

  pub fn lbp(&self, tok: &Token) -> RawBp {
    match tok {
      Token::LDeduct => 200,
      Token::Comma => 205,
      Token::Equal | Token::SlashEq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => 220,
      Token::Plus | Token::Dash => 300,
      Token::Star | Token::Slash => 320,
      Token::LParen => 400,
      Token::Query => 1200,
      Token::Dot => 1600,
      _ => 0,
    }
  }

  fn peek(&self) -> &SpanToken {
    &self.tokens[self.pos]
  }

  fn next(&mut self) -> SpanToken {
    let cur = self.tokens[self.pos].clone();
    if self.pos + 1 < self.tokens.len() {
      self.pos += 1;
    }
    if !matches!(cur.tok, Token::Indent(_) | Token::NL | Token::_Eof) {
      self.end = cur.span.end;
    }
    cur
  }

  fn at_word(&self, word: &str) -> bool {
    matches!(&self.peek().tok, Token::Ident(w) if &**w == word)
  }

  /// The error for a token where `want` should be. At the end of a line or
  /// the file, it points just after the last token.
  #[track_caller]
  fn unexpected(&self, cur: SpanToken, want: &'static str) -> ParseSpanError {
    match cur.tok {
      Token::Reserved(s) => ParseSpanError::new(cur.span, ParseError::Reserved(s)),
      tok @ (Token::NL | Token::_Eof) => {
        ParseSpanError::new(self.end..self.end, ParseError::Expected(want, tok))
      }
      tok => ParseSpanError::new(cur.span, ParseError::Expected(want, tok)),
    }
  }

  #[track_caller]
  fn expect(&mut self, tok: Token, want: &'static str) -> Result<SpanToken, ParseSpanError> {
    let cur = self.next();
    if cur.tok == tok {
      return Ok(cur);
    }
    Err(self.unexpected(cur, want))
  }

  #[track_caller]
  fn ident(&mut self, want: &'static str) -> Result<SpanIdent, ParseSpanError> {
    let cur = self.next();
    match cur.tok {
      Token::Ident(text) => Ok(SpanIdent {
        span: cur.span,
        text,
      }),
      tok => Err(self.unexpected(
        SpanToken {
          span: cur.span,
          tok,
        },
        want,
      )),
    }
  }

  pub fn mod_(&mut self) -> Result<Mod, ParseSpanError> {
    let ctx = SentCtx {
      depth: 0,
      indent: SentIndent::Eq(0),
    };
    let body = self.block(ctx, &(0..0))?;
    match self.peek().tok {
      Token::_Eof => Ok(Mod {
        span: 0..self.end,
        body,
      }),
      _ => {
        let cur = self.peek().clone();
        Err(self.unexpected(cur, "the end of the file"))
      }
    }
  }

  /// A block's sentences, up to a line indented less, or the end of the
  /// file. `colon` is the `:` that opened it, where an empty block's error
  /// points.
  pub fn block(&mut self, ctx: SentCtx, colon: &Span) -> Result<Vec<SentRef>, ParseSpanError> {
    if ctx.depth > MAX_DEPTH {
      return error(colon.clone(), ParseError::TooDeep);
    }
    let indent = match (ctx.indent, &self.peek().tok) {
      (SentIndent::Eq(indent), _) => indent,
      (SentIndent::Gt(outer), Token::Indent(n)) if *n > outer => *n,
      (SentIndent::Gt(_), _) => return error(colon.clone(), ParseError::EmptyBlock),
    };
    let mut body = Vec::new();
    loop {
      let cur = self.peek().clone();
      match cur.tok {
        Token::Indent(n) if n == indent => {
          self.next();
          self.dedent = false;
          let ctx = SentCtx {
            depth: ctx.depth + 1,
            indent: SentIndent::Eq(indent),
          };
          body.push(Rc::new(self.sent(ctx)?));
        }
        Token::Indent(n) if n > indent => {
          let err = if self.dedent {
            ParseError::Dedent
          } else {
            ParseError::Indent
          };
          return error(self.tokens[self.pos + 1].span.clone(), err);
        }
        Token::Indent(_) => {
          self.dedent = true;
          return Ok(body);
        }
        Token::_Eof => return Ok(body),
        _ => return Err(self.unexpected(cur, "a new line")),
      }
    }
  }

  /// The end of a line, after a `:` that opens a block, and the block.
  fn block_after_colon(
    &mut self,
    ctx: SentCtx,
    colon: &Span,
  ) -> Result<Vec<SentRef>, ParseSpanError> {
    self.expect(Token::NL, "the end of the line after `:`")?;
    let indent = match ctx.indent {
      SentIndent::Eq(indent) | SentIndent::Gt(indent) => indent,
    };
    let ctx = SentCtx {
      depth: ctx.depth,
      indent: SentIndent::Gt(indent),
    };
    self.block(ctx, colon)
  }

  /// One sentence, after its line's `Indent`, through its `NL` and any block.
  pub fn sent(&mut self, ctx: SentCtx) -> Result<Sent, ParseSpanError> {
    let start = self.peek().span.start;
    let word = match &self.peek().tok {
      Token::Ident(w) if matches!(self.tokens[self.pos + 1].tok, Token::Ident(_)) => {
        Some(w.clone())
      }
      _ => None,
    };
    if let Some(word) = word {
      match &*word {
        "type" => {
          self.next();
          let name = self.ident("a type's name")?;
          self.expect(Token::Equal, "`=`")?;
          let ty = self.ident("a type")?;
          self.expect(Token::NL, "the end of the line")?;
          return Ok(Sent::Type(start..ty.span.end, name, ty));
        }
        "enum" => {
          self.next();
          let name = self.ident("a type's name")?;
          let colon = self.expect(Token::Colon, "`:`")?;
          self.expect(Token::NL, "the end of the line after `:`")?;
          let ctors = self.enum_block(ctx, &colon.span)?;
          return Ok(Sent::Enum(start..colon.span.end, name, ctors));
        }
        "on" => {
          self.next();
          let pattern = self.term(ctx.term(0))?;
          let colon = self.expect(Token::Colon, "`:`")?;
          let body = self.block_after_colon(ctx, &colon.span)?;
          return Ok(Sent::On(start..colon.span.end, pattern.into(), body));
        }
        w if is_lower(w) => {
          let word = self.ident("a declaration's word")?;
          return self.decl_sent(word, start);
        }
        _ => {}
      }
    }
    match self.peek().tok {
      Token::For => {
        self.next();
        let body = self.term(ctx.term(200))?;
        let colon = self.expect(Token::Colon, "`,` or `:`")?;
        let inner = self.block_after_colon(ctx, &colon.span)?;
        return Ok(Sent::For(start..colon.span.end, body.into(), inner));
      }
      Token::Del => {
        self.next();
        let head = self.term(ctx.term(0))?;
        let end = self.end;
        self.expect(Token::NL, "the end of the line")?;
        return Ok(Sent::Del(start..end, head.into()));
      }
      _ => {}
    }
    let term = self.term(ctx.term(0))?;
    if self.peek().tok == Token::Colon {
      let colon = self.next();
      let body = self.block_after_colon(ctx, &colon.span)?;
      return Ok(Sent::Block(start..colon.span.end, term.into(), body));
    }
    self.expect(Token::NL, "the end of the line")?;
    Ok(Sent::Just(term.span(), term.into()))
  }

  /// An enum's lines: constructors, separated by commas.
  fn enum_block(&mut self, ctx: SentCtx, colon: &Span) -> Result<Vec<Decl>, ParseSpanError> {
    let indent = match (ctx.indent, &self.peek().tok) {
      (SentIndent::Eq(outer) | SentIndent::Gt(outer), Token::Indent(n)) if *n > outer => *n,
      _ => return error(colon.clone(), ParseError::EmptyBlock),
    };
    let mut ctors = Vec::new();
    loop {
      let cur = self.peek().clone();
      match cur.tok {
        Token::Indent(n) if n == indent => {
          self.next();
          self.dedent = false;
          loop {
            ctors.push(self.decl()?);
            let sep = self.next();
            match sep.tok {
              Token::Comma => {}
              Token::NL => break,
              tok => {
                return Err(self.unexpected(
                  SpanToken {
                    span: sep.span,
                    tok,
                  },
                  "`,`",
                ))
              }
            }
          }
        }
        Token::Indent(n) if n > indent => {
          return error(self.tokens[self.pos + 1].span.clone(), ParseError::Indent);
        }
        Token::Indent(_) => {
          self.dedent = true;
          return Ok(ctors);
        }
        _ => return Ok(ctors),
      }
    }
  }

  /// A declaration sentence, after its word: declarations, then the `key`
  /// columns, then flags.
  fn decl_sent(&mut self, word: SpanIdent, start: usize) -> Result<Sent, ParseSpanError> {
    let mut decls = vec![self.decl()?];
    while self.peek().tok == Token::Comma {
      self.next();
      decls.push(self.decl()?);
    }
    let mut key = Vec::new();
    if self.at_word("key") {
      self.next();
      key.push(self.ident("a field's name")?);
      while self.peek().tok == Token::Comma {
        self.next();
        key.push(self.ident("a field's name")?);
      }
    }
    let mut flags = Vec::new();
    while matches!(&self.peek().tok, Token::Ident(w) if is_lower(w)) {
      if self.at_word("key") {
        return error(self.peek().span.clone(), ParseError::KeyAfterFlags);
      }
      flags.push(self.ident("a flag")?);
    }
    let end = self.end;
    let want = match (key.is_empty(), flags.is_empty()) {
      (true, true) => "`,`, `key`, a flag or the end of the line",
      (false, true) => "`,`, a flag or the end of the line",
      (_, false) => "a flag or the end of the line",
    };
    self.expect(Token::NL, want)?;
    Ok(Sent::Decl(start..end, word, decls, key, flags))
  }

  /// `Name` or `Name(fields)`, where a field is `name: Type` or a type, with
  /// a default after `=`.
  fn decl(&mut self) -> Result<Decl, ParseSpanError> {
    let name = self.ident("a name")?;
    let mut params = Vec::new();
    if self.peek().tok == Token::LParen {
      self.next();
      while self.peek().tok != Token::RParen {
        let first = self.ident("a field's name, or a type")?;
        let start = first.span.start;
        let (name, ty) = if self.peek().tok == Token::Colon {
          self.next();
          (Some(first), self.ident("a type")?)
        } else {
          (None, first)
        };
        let default = if self.peek().tok == Token::Equal {
          self.next();
          let ctx = TermCtx { depth: 1, bp: 205 };
          Some(self.term(ctx)?.into())
        } else {
          None
        };
        params.push(Param {
          span: start..self.end,
          name,
          ty,
          default,
        });
        if self.peek().tok != Token::RParen {
          let want = if params.last().is_some_and(|p| p.name.is_some()) {
            "`,` or `)`"
          } else {
            "`:`, `,` or `)`"
          };
          self.expect(Token::Comma, want)?;
        }
      }
      self.next();
    }
    Ok(Decl {
      span: name.span.start..self.end,
      name,
      params,
    })
  }

  pub fn term(&mut self, ctx: TermCtx) -> Result<Term, ParseSpanError> {
    if ctx.depth > MAX_DEPTH {
      return error(self.peek().span.clone(), ParseError::TooDeep);
    }
    let cur = self.next();
    let mut lterm = self.term_nud(ctx, cur)?;
    while ctx.bp < self.lbp(&self.peek().tok) {
      let cur = self.next();
      lterm = self.term_led(ctx, lterm, cur)?;
    }
    Ok(lterm)
  }

  pub fn term_nud(&mut self, ctx: TermCtx, cur: SpanToken) -> Result<Term, ParseSpanError> {
    let span = cur.span.clone();
    let inner = |bp| TermCtx {
      depth: ctx.depth + 1,
      bp,
    };
    Ok(match cur.tok {
      Token::Ident(s) => Term::Ident(span, s),
      Token::PlaceIdent(_) => Term::PlaceIdent(span),
      Token::AtomLit(s) => Term::AtomLit(span, s),
      Token::IntLit(s) => Term::IntLit(span, s),
      Token::FloatLit(s) => Term::FloatLit(span, s),
      Token::True => Term::LogicLit(span, true),
      Token::False => Term::LogicLit(span, false),
      Token::LParen => {
        let term = self.term(inner(0))?;
        self.expect(Token::RParen, "`)`")?;
        Term::Group(span.start..self.end, term.into())
      }
      Token::Not => {
        let term = self.term(inner(215))?;
        Term::Not(span.start..self.end, term.into())
      }
      Token::Dash => {
        let term = self.term(inner(330))?;
        Term::Minus(span.start..self.end, term.into())
      }
      tok => return Err(self.unexpected(SpanToken { span, tok }, "a term")),
    })
  }

  pub fn term_led(
    &mut self,
    ctx: TermCtx,
    lterm: Term,
    cur: SpanToken,
  ) -> Result<Term, ParseSpanError> {
    let start = lterm.span().start;
    let rctx = TermCtx {
      depth: ctx.depth + 1,
      bp: self.lbp(&cur.tok),
    };
    match cur.tok {
      Token::Comma => {
        let rterm = self.term(rctx)?.into();
        let mut tup = match lterm {
          Term::Bunch(_, tup) => tup,
          lterm => vec![lterm.into()],
        };
        tup.push(rterm);
        Ok(Term::Bunch(start..self.end, tup))
      }
      Token::LParen => {
        let mut tup = vec![lterm.into()];
        let arg = TermCtx {
          depth: ctx.depth + 1,
          bp: 205,
        };
        while self.peek().tok != Token::RParen {
          tup.push(self.term(arg)?.into());
          if self.peek().tok != Token::RParen {
            self.expect(Token::Comma, "`,` or `)`")?;
          }
        }
        self.next();
        Ok(Term::Apply(start..self.end, tup))
      }
      Token::Query => Ok(Term::Query(start..self.end, lterm.into())),
      Token::Dot => {
        let name = self.ident("a name, after `.`")?;
        Ok(Term::Attr(start..self.end, lterm.into(), name))
      }
      tok => {
        if lterm.is_comparison() && rctx.bp == 220 {
          return error(cur.span, ParseError::Chained);
        }
        let (l, r) = (lterm.into(), self.term(rctx)?.into());
        let span = start..self.end;
        Ok(match tok {
          Token::LDeduct => Term::Deduct(span, l, r),
          Token::Equal => Term::Equal(span, l, r),
          Token::SlashEq => Term::NEqual(span, l, r),
          Token::Lt => Term::Compare(span, Cmp::Lt, l, r),
          Token::LtEq => Term::Compare(span, Cmp::LtEq, l, r),
          Token::Gt => Term::Compare(span, Cmp::Gt, l, r),
          Token::GtEq => Term::Compare(span, Cmp::GtEq, l, r),
          Token::Plus => Term::Add(span, l, r),
          Token::Dash => Term::Sub(span, l, r),
          Token::Star => Term::Mul(span, l, r),
          Token::Slash => Term::Div(span, l, r),
          tok => {
            return Err(self.unexpected(
              SpanToken {
                span: cur.span,
                tok,
              },
              "an operator",
            ))
          }
        })
      }
    }
  }
}
