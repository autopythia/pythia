//! A small JSON reader and writer for the protocol's requests and replies,
//! with no dependencies. A number keeps its text, so that an `id` is echoed
//! exactly. An object keeps its members in order, and [`Value::get`] finds the
//! last member of a name, as Python's json module does.

/// The deepest nesting that [`parse`] reads.
pub const MAX_DEPTH: usize = 64;

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
  Null,
  Bool(bool),
  /// A number as written: valid JSON number text.
  Number(String),
  String(String),
  Array(Vec<Value>),
  /// The members in order; a name may repeat.
  Object(Vec<(String, Value)>),
}

/// An object of (name, value) members, in order.
pub fn object<'a>(members: impl IntoIterator<Item = (&'a str, Value)>) -> Value {
  Value::Object(
    members
      .into_iter()
      .map(|(name, value)| (name.to_string(), value))
      .collect(),
  )
}

impl Value {
  /// The last member named `name`, if this is an object.
  pub fn get(&self, name: &str) -> Option<&Value> {
    match self {
      Value::Object(members) => members
        .iter()
        .rev()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value),
      _ => None,
    }
  }

  pub fn as_str(&self) -> Option<&str> {
    match self {
      Value::String(text) => Some(text),
      _ => None,
    }
  }

  pub fn as_bool(&self) -> Option<bool> {
    match self {
      Value::Bool(value) => Some(*value),
      _ => None,
    }
  }

  /// A number written as a whole, non-negative number (no sign, fraction or
  /// exponent) that fits in a u64.
  pub fn as_u64(&self) -> Option<u64> {
    match self {
      Value::Number(text) if text.bytes().all(|byte| byte.is_ascii_digit()) => text.parse().ok(),
      _ => None,
    }
  }

  /// What the value is, for messages: "a string", "null".
  pub fn kind(&self) -> &'static str {
    match self {
      Value::Null => "null",
      Value::Bool(_) => "a boolean",
      Value::Number(_) => "a number",
      Value::String(_) => "a string",
      Value::Array(_) => "an array",
      Value::Object(_) => "an object",
    }
  }

  /// The value as compact JSON text.
  pub fn to_vec(&self) -> Vec<u8> {
    let mut out = Vec::new();
    self.write(&mut out);
    out
  }

  /// Appends the value as compact JSON text. Strings keep their non-ASCII
  /// characters; quotes, backslashes and control characters are escaped, so
  /// the text never contains a raw newline.
  pub fn write(&self, out: &mut Vec<u8>) {
    match self {
      Value::Null => out.extend_from_slice(b"null"),
      Value::Bool(true) => out.extend_from_slice(b"true"),
      Value::Bool(false) => out.extend_from_slice(b"false"),
      Value::Number(text) => out.extend_from_slice(text.as_bytes()),
      Value::String(text) => write_string(text, out),
      Value::Array(items) => {
        out.push(b'[');
        for (index, item) in items.iter().enumerate() {
          if index > 0 {
            out.push(b',');
          }
          item.write(out);
        }
        out.push(b']');
      }
      Value::Object(members) => {
        out.push(b'{');
        for (index, (name, value)) in members.iter().enumerate() {
          if index > 0 {
            out.push(b',');
          }
          write_string(name, out);
          out.push(b':');
          value.write(out);
        }
        out.push(b'}');
      }
    }
  }
}

impl From<bool> for Value {
  fn from(value: bool) -> Value {
    Value::Bool(value)
  }
}

impl From<&str> for Value {
  fn from(text: &str) -> Value {
    Value::String(text.to_string())
  }
}

impl From<String> for Value {
  fn from(text: String) -> Value {
    Value::String(text)
  }
}

impl From<u32> for Value {
  fn from(number: u32) -> Value {
    Value::Number(number.to_string())
  }
}

impl From<u64> for Value {
  fn from(number: u64) -> Value {
    Value::Number(number.to_string())
  }
}

impl From<usize> for Value {
  fn from(number: usize) -> Value {
    Value::Number(number.to_string())
  }
}

/// A finite number as its shortest decimal text (2.0 is `2`); NaN and the
/// infinities, which JSON lacks, are null.
impl From<f64> for Value {
  fn from(number: f64) -> Value {
    if number.is_finite() {
      Value::Number(number.to_string())
    } else {
      Value::Null
    }
  }
}

impl From<Vec<Value>> for Value {
  fn from(items: Vec<Value>) -> Value {
    Value::Array(items)
  }
}

/// None is null.
impl<T: Into<Value>> From<Option<T>> for Value {
  fn from(value: Option<T>) -> Value {
    value.map_or(Value::Null, Into::into)
  }
}

fn write_string(text: &str, out: &mut Vec<u8>) {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  out.push(b'"');
  let bytes = text.as_bytes();
  let mut start = 0;
  for (index, &byte) in bytes.iter().enumerate() {
    let escape: &[u8] = match byte {
      b'"' => b"\\\"",
      b'\\' => b"\\\\",
      b'\n' => b"\\n",
      b'\r' => b"\\r",
      b'\t' => b"\\t",
      0x00..=0x1F => &[
        b'\\',
        b'u',
        b'0',
        b'0',
        HEX[usize::from(byte >> 4)],
        HEX[usize::from(byte & 0xF)],
      ],
      _ => continue,
    };
    out.extend_from_slice(&bytes[start..index]);
    out.extend_from_slice(escape);
    start = index + 1;
  }
  out.extend_from_slice(&bytes[start..]);
  out.push(b'"');
}

/// Reads one JSON value, with nothing but whitespace around it. An error
/// says what was expected, and where (a byte offset).
pub fn parse(text: &str) -> Result<Value, String> {
  let mut parser = Parser {
    text,
    bytes: text.as_bytes(),
    pos: 0,
  };
  let value = parser.value(0)?;
  parser.skip_whitespace();
  if parser.pos < parser.bytes.len() {
    return Err(parser.expected("the end of the text"));
  }
  Ok(value)
}

struct Parser<'a> {
  text: &'a str,
  bytes: &'a [u8],
  pos: usize,
}

impl Parser<'_> {
  fn expected(&self, what: &str) -> String {
    if self.pos < self.bytes.len() {
      format!("expected {what} at byte {}", self.pos)
    } else {
      format!("expected {what} at the end")
    }
  }

  fn peek(&self) -> Option<u8> {
    self.bytes.get(self.pos).copied()
  }

  fn eat(&mut self, byte: u8) -> bool {
    let found = self.peek() == Some(byte);
    if found {
      self.pos += 1;
    }
    found
  }

  fn skip_whitespace(&mut self) {
    while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
      self.pos += 1;
    }
  }

  /// A value whose containers are `depth` deep.
  fn value(&mut self, depth: usize) -> Result<Value, String> {
    self.skip_whitespace();
    match self.peek() {
      Some(b'{') => self.object(depth + 1),
      Some(b'[') => self.array(depth + 1),
      Some(b'"') => self.string().map(Value::String),
      Some(b't') => self.literal("true", Value::Bool(true)),
      Some(b'f') => self.literal("false", Value::Bool(false)),
      Some(b'n') => self.literal("null", Value::Null),
      Some(b'-' | b'0'..=b'9') => self.number(),
      _ => Err(self.expected("a value")),
    }
  }

  fn nest(&self, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
      return Err(format!(
        "nested deeper than {MAX_DEPTH} at byte {}",
        self.pos
      ));
    }
    Ok(())
  }

  fn object(&mut self, depth: usize) -> Result<Value, String> {
    self.nest(depth)?;
    self.pos += 1;
    let mut members = Vec::new();
    self.skip_whitespace();
    if self.eat(b'}') {
      return Ok(Value::Object(members));
    }
    loop {
      self.skip_whitespace();
      if self.peek() != Some(b'"') {
        return Err(self.expected("a member name in quotes"));
      }
      let name = self.string()?;
      self.skip_whitespace();
      if !self.eat(b':') {
        return Err(self.expected("':'"));
      }
      let value = self.value(depth)?;
      members.push((name, value));
      self.skip_whitespace();
      if self.eat(b'}') {
        return Ok(Value::Object(members));
      }
      if !self.eat(b',') {
        return Err(self.expected("',' or '}'"));
      }
    }
  }

  fn array(&mut self, depth: usize) -> Result<Value, String> {
    self.nest(depth)?;
    self.pos += 1;
    let mut items = Vec::new();
    self.skip_whitespace();
    if self.eat(b']') {
      return Ok(Value::Array(items));
    }
    loop {
      items.push(self.value(depth)?);
      self.skip_whitespace();
      if self.eat(b']') {
        return Ok(Value::Array(items));
      }
      if !self.eat(b',') {
        return Err(self.expected("',' or ']'"));
      }
    }
  }

  fn literal(&mut self, word: &str, value: Value) -> Result<Value, String> {
    if !self.bytes[self.pos..].starts_with(word.as_bytes()) {
      return Err(self.expected("a value"));
    }
    self.pos += word.len();
    Ok(value)
  }

  fn digits(&mut self) -> Result<(), String> {
    if !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
      return Err(self.expected("a digit"));
    }
    while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
      self.pos += 1;
    }
    Ok(())
  }

  fn number(&mut self) -> Result<Value, String> {
    let start = self.pos;
    self.eat(b'-');
    // No leading zeros: 0, or a digit from 1 and more digits.
    if !self.eat(b'0') {
      self.digits()?;
    }
    if self.eat(b'.') {
      self.digits()?;
    }
    if matches!(self.peek(), Some(b'e' | b'E')) {
      self.pos += 1;
      if matches!(self.peek(), Some(b'+' | b'-')) {
        self.pos += 1;
      }
      self.digits()?;
    }
    Ok(Value::Number(self.text[start..self.pos].to_string()))
  }

  /// A string, from its opening quote. Runs of plain characters are copied
  /// whole; they end only at ASCII bytes, which are character boundaries.
  fn string(&mut self) -> Result<String, String> {
    self.pos += 1;
    let mut text = String::new();
    let mut start = self.pos;
    loop {
      match self.peek() {
        None => return Err(self.expected("'\"' to end the string")),
        Some(b'"') => {
          text.push_str(&self.text[start..self.pos]);
          self.pos += 1;
          return Ok(text);
        }
        Some(b'\\') => {
          text.push_str(&self.text[start..self.pos]);
          self.pos += 1;
          text.push(self.escape()?);
          start = self.pos;
        }
        Some(0x00..=0x1F) => {
          return Err(format!(
            "a control character in a string at byte {}",
            self.pos
          ))
        }
        Some(_) => self.pos += 1,
      }
    }
  }

  /// The character of an escape, after its backslash.
  fn escape(&mut self) -> Result<char, String> {
    let at = self.pos - 1;
    let Some(byte) = self.peek() else {
      return Err(self.expected("an escape"));
    };
    self.pos += 1;
    Ok(match byte {
      b'"' => '"',
      b'\\' => '\\',
      b'/' => '/',
      b'b' => '\u{8}',
      b'f' => '\u{c}',
      b'n' => '\n',
      b'r' => '\r',
      b't' => '\t',
      b'u' => {
        let lone = || format!("a lone surrogate in a \\u escape at byte {at}");
        let first = self.hex4()?;
        let code = match first {
          0xD800..=0xDBFF => {
            if !(self.eat(b'\\') && self.eat(b'u')) {
              return Err(lone());
            }
            let second = self.hex4()?;
            if !(0xDC00..=0xDFFF).contains(&second) {
              return Err(lone());
            }
            0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
          }
          0xDC00..=0xDFFF => return Err(lone()),
          _ => first,
        };
        char::from_u32(code).ok_or_else(lone)?
      }
      _ => return Err(format!("an unknown escape at byte {at}")),
    })
  }

  fn hex4(&mut self) -> Result<u32, String> {
    let digits = self
      .bytes
      .get(self.pos..self.pos + 4)
      .filter(|digits| digits.iter().all(u8::is_ascii_hexdigit))
      .ok_or_else(|| self.expected("four hex digits"))?;
    let value = digits.iter().fold(0, |value, &digit| {
      value * 16 + char::from(digit).to_digit(16).unwrap_or(0)
    });
    self.pos += 4;
    Ok(value)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn round(text: &str) -> String {
    String::from_utf8(parse(text).unwrap().to_vec()).unwrap()
  }

  #[test]
  fn values_read_and_write_back() {
    assert_eq!(
      round(r#" {"a": [1, -2.5e3, 0, true, false, null], "b": {}, "c": []} "#),
      r#"{"a":[1,-2.5e3,0,true,false,null],"b":{},"c":[]}"#
    );
    // A number keeps its text, beyond what a float holds.
    assert_eq!(round("12345678901234567890123"), "12345678901234567890123");
    assert_eq!(
      parse("12345678901234567890").unwrap().as_u64(),
      Some(12345678901234567890)
    );
  }

  #[test]
  fn strings_unescape_and_escape() {
    let value = parse(r#""q\" b\\ s\/ \b\f\n\r\t é \u00e9 \ud83d\ude00 \u0001""#).unwrap();
    assert_eq!(
      value,
      Value::String("q\" b\\ s/ \u{8}\u{c}\n\r\t é é 😀 \u{1}".into())
    );
    assert_eq!(
      String::from_utf8(value.to_vec()).unwrap(),
      r#""q\" b\\ s/ \u0008\u000c\n\r\t é é 😀 \u0001""#
    );
    // No raw newline survives writing.
    assert!(!Value::from("a\nb\r").to_vec().contains(&b'\n'));
  }

  #[test]
  fn bad_text_is_refused_with_where() {
    for (text, error) in [
      ("", "expected a value at the end"),
      ("{", "expected a member name in quotes at the end"),
      ("[1,]", "expected a value at byte 3"),
      (r#"{"a":1,}"#, "expected a member name in quotes at byte 7"),
      ("01", "expected the end of the text at byte 1"),
      ("-", "expected a digit at the end"),
      ("1.", "expected a digit at the end"),
      (".5", "expected a value at byte 0"),
      ("1e+", "expected a digit at the end"),
      ("+1", "expected a value at byte 0"),
      ("tru", "expected a value at byte 0"),
      ("\"a", "expected '\"' to end the string at the end"),
      ("\"a\nb\"", "a control character in a string at byte 2"),
      (r#""\x""#, "an unknown escape at byte 1"),
      (r#""\u12""#, "expected four hex digits at byte 3"),
      (r#""\u+123""#, "expected four hex digits at byte 3"),
      (r#""\ud800""#, "a lone surrogate in a \\u escape at byte 1"),
      (r#""\udc00""#, "a lone surrogate in a \\u escape at byte 1"),
      (
        r#""\ud800\u0041""#,
        "a lone surrogate in a \\u escape at byte 1",
      ),
      ("[1] [2]", "expected the end of the text at byte 4"),
    ] {
      assert_eq!(parse(text), Err(error.to_string()), "{text:?}");
    }
  }

  #[test]
  fn nesting_is_limited() {
    let deep = |depth| "[".repeat(depth) + &"]".repeat(depth);
    assert!(parse(&deep(MAX_DEPTH)).is_ok());
    assert_eq!(
      parse(&deep(MAX_DEPTH + 1)),
      Err(format!(
        "nested deeper than {MAX_DEPTH} at byte {MAX_DEPTH}"
      ))
    );
    // Far deeper text fails the same way, without overflowing the stack.
    assert!(parse(&deep(100_000)).is_err());
  }

  #[test]
  fn get_finds_the_last_member_of_a_name() {
    let value = parse(r#"{"op": "a", "x": 1, "op": "b"}"#).unwrap();
    assert_eq!(value.get("op").and_then(Value::as_str), Some("b"));
    assert_eq!(value.get("x").and_then(Value::as_u64), Some(1));
    assert_eq!(value.get("y"), None);
  }

  #[test]
  fn as_u64_takes_whole_numbers_only() {
    for (text, expected) in [
      ("0", Some(0)),
      ("4294967296", Some(4294967296)),
      ("-1", None),
      ("1.0", None),
      ("1e2", None),
      ("18446744073709551616", None),
      ("\"1\"", None),
    ] {
      assert_eq!(parse(text).unwrap().as_u64(), expected, "{text}");
    }
  }

  #[test]
  fn numbers_from_rust_values() {
    assert_eq!(Value::from(2.0).to_vec(), b"2");
    assert_eq!(Value::from(1.5).to_vec(), b"1.5");
    assert_eq!(Value::from(-0.25).to_vec(), b"-0.25");
    assert_eq!(Value::from(f64::NAN), Value::Null);
    assert_eq!(Value::from(f64::INFINITY), Value::Null);
    assert_eq!(Value::from(u64::MAX).to_vec(), b"18446744073709551615");
    assert_eq!(Value::from(None::<String>), Value::Null);
  }

  #[test]
  fn objects_keep_their_order() {
    let value = object([("z", 1u32.into()), ("a", "x".into())]);
    assert_eq!(value.to_vec(), br#"{"z":1,"a":"x"}"#);
  }
}
