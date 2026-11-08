//! lem's tokens and tokenizer.
//!
//! After zinterp's (`_pythia/src/zinterp/parse.rs` in pythia): the same token
//! names; `Indent(n)` at the start of each line, where a tab moves to the
//! next multiple of 8; `--` and `#` comments; `_` alone as a placeholder;
//! zinterp's string escapes, in `"…"` and `"""…"""`; and keywords that win a
//! tie with a name (`for` is a keyword, `format` a name). zinterp's other
//! operators and keywords lex too, as [`Token::Reserved`], so that the
//! languages can converge; the parser refuses them.
//!
//! What differs: without `regex`, operators are a table of spellings, and
//! numbers and names are scanned by hand. `-` is always a token of its own
//! (zinterp reads `-1` as one number); `0x4772b3` and `0.4` are numbers.
//! Lines join as in Python: inside brackets, or after a `\` that ends a line,
//! a line break is a space, and a bracket still open at the end of the file
//! is an error at the bracket. Blank lines and comments make no tokens, every
//! line ends with [`Token::NL`], and a character that starts no token is an
//! error rather than the end of the file.

use super::parse::{ParseError, ParseSpanError};
use super::value::int_fits;
use std::ops::Range;
use std::rc::Rc;

/// A range of bytes in a file's text.
pub type Span = Range<usize>;
/// An indentation, in columns.
pub type RawIndent = u32;
/// A literal's or a name's text.
pub type Lit = Rc<str>;

/// The smallest span that covers both.
pub trait Hull<Rhs> {
  fn hull(&self, rhs: Rhs) -> Self;
}

impl Hull<Span> for Span {
  fn hull(&self, rhs: Span) -> Span {
    self.start.min(rhs.start)..self.end.max(rhs.end)
  }
}

impl Hull<&Span> for Span {
  fn hull(&self, rhs: &Span) -> Span {
    self.start.min(rhs.start)..self.end.max(rhs.end)
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
  /// The start of a line that has tokens, with its indentation.
  Indent(RawIndent),
  /// The end of such a line.
  NL,
  Comma,
  Dot,
  Query,
  LDeduct,
  Colon,
  Plus,
  Dash,
  Star,
  Slash,
  Equal,
  SlashEq,
  Lt,
  LtEq,
  Gt,
  GtEq,
  LParen,
  RParen,
  True,
  False,
  Del,
  For,
  Not,
  /// A whole number, as written: decimal, or hex after `0x`.
  IntLit(Lit),
  /// A number with a fraction, as written.
  FloatLit(Lit),
  /// Text, with its escapes decoded.
  AtomLit(Lit),
  /// `_`, or more underscores alone.
  PlaceIdent(Lit),
  Ident(Lit),
  /// zinterp's: an operator or keyword that lem doesn't use (yet).
  Reserved(&'static str),
  _Eof,
}

impl Token {
  /// How a token without a payload is written.
  pub fn spelling(&self) -> &'static str {
    match self {
      Token::Comma => ",",
      Token::Dot => ".",
      Token::Query => "?",
      Token::LDeduct => ":-",
      Token::Colon => ":",
      Token::Plus => "+",
      Token::Dash => "-",
      Token::Star => "*",
      Token::Slash => "/",
      Token::Equal => "=",
      Token::SlashEq => "/=",
      Token::Lt => "<",
      Token::LtEq => "<=",
      Token::Gt => ">",
      Token::GtEq => ">=",
      Token::LParen => "(",
      Token::RParen => ")",
      Token::True => "True",
      Token::False => "False",
      Token::Del => "del",
      Token::For => "for",
      Token::Not => "not",
      Token::Reserved(s) => s,
      _ => "",
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpanToken {
  pub span: Span,
  pub tok: Token,
}

impl From<(Span, Token)> for SpanToken {
  fn from(t: (Span, Token)) -> SpanToken {
    SpanToken {
      span: t.0,
      tok: t.1,
    }
  }
}

/// The operators, as zinterp's tokenizer orders them: the first that the
/// text starts with wins, so where one spelling begins another, the longer
/// comes first.
const OPERATORS: &[&str] = &[
  ",", ".-", ".=", ".(", "...", "..", ".", "!", "?-", "?=", "?", ";", ":-", ":=", ":<", ":>", ":(",
  ":", "+?", "+", "-.", "-?", "-:", "->", "-/", "-", "=.", "=?", "=:", "==", "=", "/=", "/", "**",
  "*", "<=", "<", ">=", ">", "(", ")", "[", "]", "{", "}", "`",
];

fn operator(op: &'static str) -> Token {
  match op {
    "," => Token::Comma,
    "." => Token::Dot,
    "?" => Token::Query,
    ":-" => Token::LDeduct,
    ":" => Token::Colon,
    "+" => Token::Plus,
    "-" => Token::Dash,
    "*" => Token::Star,
    "/" => Token::Slash,
    "=" => Token::Equal,
    "/=" => Token::SlashEq,
    "<" => Token::Lt,
    "<=" => Token::LtEq,
    ">" => Token::Gt,
    ">=" => Token::GtEq,
    "(" => Token::LParen,
    ")" => Token::RParen,
    op => Token::Reserved(op),
  }
}

/// zinterp's keywords and literals that lem doesn't use. lem's own words,
/// such as `on` and `key`, are names that the parser knows by their place,
/// since a field may have the same name (`Button(…, on=Run)`).
const RESERVED_WORDS: &[&str] = &[
  "None", "Unk", "Par", "Para", "and", "as", "async", "await", "break", "case", "cases", "class",
  "continue", "def", "defmatch", "defproc", "defrule", "elif", "else", "except", "finally",
  "fresh", "from", "global", "if", "import", "in", "is", "lambda", "let", "match", "nonlocal",
  "of", "or", "pass", "quote", "raise", "return", "rule", "try", "unquote", "while", "with",
  "yield",
];

fn word(word: &str) -> Token {
  match word {
    "True" => Token::True,
    "False" => Token::False,
    "del" => Token::Del,
    "for" => Token::For,
    "not" => Token::Not,
    _ => match RESERVED_WORDS.iter().find(|w| **w == word) {
      Some(w) => Token::Reserved(w),
      None if word.bytes().all(|b| b == b'_') => Token::PlaceIdent(word.into()),
      None => Token::Ident(word.into()),
    },
  }
}

/// All of `buf`'s tokens, ending with [`Token::_Eof`].
pub fn tokenize(buf: &str) -> Result<Vec<SpanToken>, ParseSpanError> {
  let mut tokens = Tokenizer::new(buf);
  let mut out = Vec::new();
  loop {
    let token = tokens.next_token()?;
    let eof = token.tok == Token::_Eof;
    out.push(token);
    if eof {
      return Ok(out);
    }
  }
}

pub struct Tokenizer<'a> {
  buf: &'a str,
  pos: usize,
  /// At the start of a line, before its indentation.
  bol: bool,
  /// Whether this line has had a token, and so gets an `NL`.
  line: bool,
  /// The brackets open, innermost last: while any is, a line break is a
  /// space.
  open: Vec<(Span, char)>,
}

impl<'a> Tokenizer<'a> {
  pub fn new(buf: &'a str) -> Tokenizer<'a> {
    Tokenizer {
      buf,
      pos: 0,
      bol: true,
      line: false,
      open: Vec::new(),
    }
  }

  fn peek_char(&self) -> Option<char> {
    self.buf[self.pos..].chars().next()
  }

  fn rest(&self) -> &'a str {
    &self.buf[self.pos..]
  }

  fn advance(&mut self, o: usize) -> Span {
    let start = self.pos;
    self.pos += o;
    start..self.pos
  }

  /// Skips to the end of the line, before its line break.
  fn skip_comment(&mut self) {
    let o = self.rest().find(['\n', '\r']).unwrap_or(self.rest().len());
    self.pos += o;
  }

  /// Skips one line break, `\n`, `\r\n` or `\r`, if there's one.
  fn skip_line_break(&mut self) {
    if self.rest().starts_with("\r\n") {
      self.pos += 2;
    } else if self.rest().starts_with(['\n', '\r']) {
      self.pos += 1;
    }
  }

  fn is_comment(&self) -> bool {
    self.rest().starts_with('#') || self.rest().starts_with("--")
  }

  pub fn next_token(&mut self) -> Result<SpanToken, ParseSpanError> {
    loop {
      if self.bol {
        let mut indent = 0;
        let mut o = 0;
        for c in self.rest().chars() {
          match c {
            ' ' => indent += 1,
            '\t' => indent = (indent / 8 + 1) * 8,
            _ => break,
          }
          o += 1;
        }
        let span = self.advance(o);
        if self.is_comment() {
          self.skip_comment();
        }
        if self.rest().is_empty() {
          self.bol = false;
          continue;
        }
        if self.rest().starts_with(['\n', '\r']) {
          self.skip_line_break();
          continue;
        }
        self.bol = false;
        self.line = true;
        return Ok((span, Token::Indent(indent)).into());
      }
      let o = self.rest().len() - self.rest().trim_start_matches([' ', '\t']).len();
      self.pos += o;
      let Some(c) = self.peek_char() else {
        if let Some((span, c)) = self.open.pop() {
          return Err(ParseSpanError::new(span, ParseError::Unclosed(c)));
        }
        let end = self.pos;
        if std::mem::take(&mut self.line) {
          return Ok((end..end, Token::NL).into());
        }
        return Ok((end..end, Token::_Eof).into());
      };
      match c {
        '\n' | '\r' => {
          let start = self.pos;
          self.skip_line_break();
          if self.open.is_empty() {
            self.bol = true;
            self.line = false;
            return Ok((start..self.pos, Token::NL).into());
          }
        }
        _ if self.is_comment() => self.skip_comment(),
        '\\' => {
          let start = self.pos;
          self.pos += 1;
          let o = self.rest().len() - self.rest().trim_start_matches([' ', '\t']).len();
          self.pos += o;
          if !self.rest().starts_with(['\n', '\r']) {
            return Err(ParseSpanError::new(
              start..start + 1,
              ParseError::StrayBackslash,
            ));
          }
          self.skip_line_break();
        }
        '"' => return self.string(),
        '0'..='9' => return self.number(),
        'A'..='Z' | 'a'..='z' | '_' => {
          let o = self
            .rest()
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(self.rest().len());
          let span = self.advance(o);
          return Ok((span.clone(), word(&self.buf[span])).into());
        }
        _ => {
          let Some(op) = OPERATORS.iter().find(|op| self.rest().starts_with(**op)) else {
            let span = self.pos..self.pos + c.len_utf8();
            return Err(ParseSpanError::new(span, ParseError::BadChar(c)));
          };
          match *op {
            "(" | "[" | "{" => self.open.push((self.pos..self.pos + 1, c)),
            ")" | "]" | "}" => {
              self.open.pop();
            }
            _ => {}
          }
          let span = self.advance(op.len());
          return Ok((span, operator(op)).into());
        }
      }
    }
  }

  fn number(&mut self) -> Result<SpanToken, ParseSpanError> {
    let start = self.pos;
    let digits = |s: &str, hex: bool| {
      s.find(|c: char| !(c.is_ascii_digit() || (hex && c.is_ascii_hexdigit())))
        .unwrap_or(s.len())
    };
    let rest = self.rest();
    // The token, or what's wrong with the number's digits.
    let (len, tok) = if let Some(hex) = rest.strip_prefix("0x") {
      let hex = &hex[..digits(hex, true)];
      let len = 2 + hex.len();
      let tok = match i64::from_str_radix(hex, 16) {
        _ if hex.is_empty() => Err(ParseError::BadNumber),
        Ok(n) if int_fits(n) => Ok(Token::IntLit(rest[..len].into())),
        _ => Err(ParseError::IntTooLarge),
      };
      (len, tok)
    } else {
      let whole = digits(rest, false);
      let after = &rest[whole..];
      let fraction = match after.strip_prefix('.') {
        Some(more) if more.starts_with(|c: char| c.is_ascii_digit()) => 1 + digits(more, false),
        _ => 0,
      };
      let len = whole + fraction;
      let text = &rest[..len];
      let tok = if fraction > 0 {
        match text.parse::<f64>() {
          Ok(x) if x.is_finite() => Ok(Token::FloatLit(text.into())),
          _ => Err(ParseError::NumTooLarge),
        }
      } else {
        // Digits alone fail to parse only when they're too many.
        match text.parse::<i64>() {
          Ok(n) if int_fits(n) => Ok(Token::IntLit(text.into())),
          _ => Err(ParseError::IntTooLarge),
        }
      };
      (len, tok)
    };
    self.pos += len;
    let runs_on = self
      .peek_char()
      .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    if runs_on {
      let o = self
        .rest()
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(self.rest().len());
      return Err(ParseSpanError::new(
        start..self.pos + o,
        ParseError::BadNumber,
      ));
    }
    match tok {
      Ok(tok) => Ok((start..self.pos, tok).into()),
      Err(err) => Err(ParseSpanError::new(start..self.pos, err)),
    }
  }

  fn string(&mut self) -> Result<SpanToken, ParseSpanError> {
    let start = self.pos;
    let delim = if self.rest().starts_with("\"\"\"") {
      "\"\"\""
    } else {
      "\""
    };
    let lines = delim.len() == 3;
    self.pos += delim.len();
    let mut text = String::new();
    loop {
      if self.rest().starts_with(delim) {
        self.pos += delim.len();
        return Ok((start..self.pos, Token::AtomLit(text.into())).into());
      }
      let at = self.pos;
      match self.peek_char() {
        None => return Err(ParseSpanError::new(start..at, ParseError::UnclosedStr)),
        Some('\\') => {
          self.pos += 1;
          let c = self
            .escape()
            .ok_or_else(|| ParseSpanError::new(at..self.pos.max(at + 1), ParseError::BadEscape))?;
          text.push(c);
        }
        Some('\n' | '\r') if !lines => {
          return Err(ParseSpanError::new(start..at, ParseError::UnclosedStr))
        }
        Some(c @ ('\n' | '\r' | '\t')) if lines => {
          text.push(c);
          self.pos += 1;
        }
        Some(c) if c < ' ' || c == '\x7f' => {
          return Err(ParseSpanError::new(at..at + 1, ParseError::ControlInStr));
        }
        Some(c) => {
          text.push(c);
          self.pos += c.len_utf8();
        }
      }
    }
  }

  /// The character that the escape after a `\` stands for, as zinterp's
  /// strings (and JSON's) have them; None for a bad one.
  fn escape(&mut self) -> Option<char> {
    let c = self.peek_char()?;
    self.pos += c.len_utf8();
    Some(match c {
      '"' => '"',
      '\\' => '\\',
      '/' => '/',
      'b' => '\x08',
      'f' => '\x0c',
      'n' => '\n',
      'r' => '\r',
      't' => '\t',
      'u' => {
        let n1 = self.hex4()?;
        match n1 {
          0xd800..=0xdbff => {
            self.rest().starts_with("\\u").then_some(())?;
            self.pos += 2;
            let n2 = self.hex4()?;
            (0xdc00..=0xdfff).contains(&n2).then_some(())?;
            char::from_u32((((n1 - 0xd800) << 10) | (n2 - 0xdc00)) + 0x1_0000)?
          }
          n => char::from_u32(n)?,
        }
      }
      _ => return None,
    })
  }

  fn hex4(&mut self) -> Option<u32> {
    let hex = self.rest().get(..4)?;
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
      return None;
    }
    self.pos += 4;
    u32::from_str_radix(hex, 16).ok()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The tokens, one word each: the token's name, with its payload.
  fn lex(src: &str) -> String {
    let tokens = tokenize(src).unwrap_or_else(|e| panic!("{}", e.render("t", src)));
    let words: Vec<String> = tokens
      .iter()
      .map(|t| match &t.tok {
        Token::Indent(n) => format!("Indent({n})"),
        Token::IntLit(s) | Token::FloatLit(s) | Token::Ident(s) | Token::PlaceIdent(s) => {
          s.to_string()
        }
        Token::AtomLit(s) => format!("{s:?}"),
        Token::Reserved(s) => format!("Reserved({s})"),
        tok => format!("{tok:?}"),
      })
      .collect();
    words.join(" ")
  }

  fn lex_error(src: &str) -> String {
    let error = tokenize(src).expect_err(src);
    error.render("t", src)
  }

  #[test]
  fn a_rule_lexes_as_in_zinterp() {
    assert_eq!(
      lex("Late(Jobs, by=Minutes) :- Job():\n  for Job(c, _):\n"),
      "Indent(0) Late LParen Jobs Comma by Equal Minutes RParen LDeduct Job LParen RParen Colon \
       NL Indent(2) For Job LParen c Comma _ RParen Colon NL _Eof"
    );
  }

  #[test]
  fn keywords_win_only_whole_words() {
    assert_eq!(
      lex("not nothing for format True Trueish del on key _x __ in"),
      "Indent(0) Not nothing For format True Trueish Del on key _x __ Reserved(in) NL _Eof"
    );
  }

  #[test]
  fn operators_take_the_longest_spelling() {
    assert_eq!(
      lex("a:-b :c <= < >= > /= / = == -: - ! ;"),
      "Indent(0) a LDeduct b Colon c LtEq Lt GtEq Gt SlashEq Slash Equal Reserved(==) \
       Reserved(-:) Dash Reserved(!) Reserved(;) NL _Eof"
    );
  }

  #[test]
  fn numbers_and_minus() {
    assert_eq!(
      lex("-1 ends-t 0.4 0x4772b3ff 1.x"),
      "Indent(0) Dash 1 ends Dash t 0.4 0x4772b3ff 1 Dot x NL _Eof"
    );
  }

  #[test]
  fn whole_numbers_fit_in_53_bits() {
    assert_eq!(
      lex("9007199254740991 0x1fffffffffffff 9007199254740993.0"),
      "Indent(0) 9007199254740991 0x1fffffffffffff 9007199254740993.0 NL _Eof"
    );
    for src in [
      "9007199254740992",
      "0x20000000000000",
      "99999999999999999999",
    ] {
      let error = lex_error(src);
      assert!(
        error.starts_with("t:1:1: a whole number must be at most 9007199254740991 (2^53 − 1)"),
        "{error}"
      );
      assert!(
        error.ends_with(&format!("\n{}", "^".repeat(src.len()))),
        "{error}"
      );
    }
  }

  #[test]
  fn indents_comments_and_blank_lines() {
    let src = "a:\n\n  -- note\n  b # more\n\tc\n   \n";
    assert_eq!(
      lex(src),
      "Indent(0) a Colon NL Indent(2) b NL Indent(8) c NL _Eof"
    );
    // The last line needs no line break, and a file of comments is empty.
    assert_eq!(lex("a"), "Indent(0) a NL _Eof");
    assert_eq!(lex("# x\n-- y"), "_Eof");
    assert_eq!(lex(""), "_Eof");
  }

  #[test]
  fn lines_join_inside_brackets_and_after_a_backslash() {
    assert_eq!(
      lex("A(x,\n    -- why\n  y) \\\n  :- B\nC"),
      "Indent(0) A LParen x Comma y RParen LDeduct B NL Indent(0) C NL _Eof"
    );
    assert_eq!(
      lex("A(\r\n)\r\nB\rC"),
      "Indent(0) A LParen RParen NL Indent(0) B NL Indent(0) C NL _Eof"
    );
  }

  #[test]
  fn text_has_zinterps_escapes() {
    assert_eq!(
      lex(r#"X("a\"b\\c\/\n\u00e9\ud83d\ude00", "é")"#),
      "Indent(0) X LParen \"a\\\"b\\\\c/\\né😀\" Comma \"é\" RParen NL _Eof"
    );
    assert_eq!(
      lex("\"\"\"one\n\ttwo\"\"\""),
      "Indent(0) \"one\\n\\ttwo\" NL _Eof"
    );
  }

  #[test]
  fn lexical_errors_say_where_and_what() {
    assert_eq!(
      lex_error("A(\"Run now)\n"),
      "t:1:3: this text has no closing `\"` on its line (`\"\"\"` spans lines):\nA(\"Run now)\n  \
       ^^^^^^^^^"
    );
    assert!(lex_error("x = \"a\\qb\"").starts_with("t:1:7: the escapes are"));
    assert!(lex_error("x = \"\\ud800\"").starts_with("t:1:6: the escapes are"));
    assert!(lex_error("x = 12px").starts_with("t:1:5: a number must end before any letter"));
    assert!(lex_error("x = 99999999999999999999").contains("\n    ^^^^^^^^^^^^^^^^^^^^"));
    assert!(lex_error("x = 0x").starts_with("t:1:5: a number"));
    assert!(lex_error("a \\ b").starts_with("t:1:3: a `\\` joins the next line"));
    assert!(lex_error("a @ b").starts_with("t:1:3: '@' isn't part of any token"));
    assert!(lex_error("\"a\tb\"").starts_with("t:1:3: a control character"));
    assert_eq!(
      lex_error("A(B(x),\n  C(y\n"),
      "t:2:4: this `(` has no matching `)`:\n  C(y\n   ^"
    );
  }
}
