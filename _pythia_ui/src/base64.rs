//! Standard base64 (RFC 4648, with padding), for the PNG in a screenshot's
//! reply. Encoding only.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The length of [`encode`]'s text for `len` bytes.
pub fn encoded_len(len: usize) -> usize {
  len.div_ceil(3) * 4
}

/// `data` in base64.
pub fn encode(data: &[u8]) -> String {
  let mut out = String::with_capacity(encoded_len(data.len()));
  let mut chunks = data.chunks_exact(3);
  for chunk in &mut chunks {
    let bits = (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]);
    push(&mut out, bits, 4);
  }
  match *chunks.remainder() {
    [a] => {
      push(&mut out, u32::from(a) << 16, 2);
      out.push_str("==");
    }
    [a, b] => {
      push(&mut out, (u32::from(a) << 16) | (u32::from(b) << 8), 3);
      out.push('=');
    }
    _ => {}
  }
  out
}

/// Pushes the first `count` of the four 6-bit groups in `bits` (24 bits).
fn push(out: &mut String, bits: u32, count: usize) {
  for shift in [18, 12, 6, 0].into_iter().take(count) {
    out.push(char::from(ALPHABET[((bits >> shift) & 63) as usize]));
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn rfc_4648_vectors() {
    for (data, text) in [
      ("", ""),
      ("f", "Zg=="),
      ("fo", "Zm8="),
      ("foo", "Zm9v"),
      ("foob", "Zm9vYg=="),
      ("fooba", "Zm9vYmE="),
      ("foobar", "Zm9vYmFy"),
    ] {
      assert_eq!(encode(data.as_bytes()), text);
      assert_eq!(encoded_len(data.len()), text.len());
    }
  }

  /// A plain decoder, for the test only.
  fn decode(text: &str) -> Vec<u8> {
    let value = |c: u8| ALPHABET.iter().position(|&a| a == c).unwrap() as u32;
    let mut out = Vec::new();
    for quad in text.as_bytes().chunks(4) {
      let digits: Vec<u8> = quad.iter().copied().filter(|&c| c != b'=').collect();
      let bits =
        digits.iter().fold(0, |bits, &c| (bits << 6) | value(c)) << (6 * (4 - digits.len()));
      out.extend_from_slice(&bits.to_be_bytes()[1..digits.len()]);
    }
    out
  }

  #[test]
  fn every_byte_and_length_round_trips() {
    let data: Vec<u8> = (0..=255).cycle().take(1000).collect();
    for len in 0..64 {
      assert_eq!(decode(&encode(&data[..len])), &data[..len]);
    }
    assert_eq!(decode(&encode(&data)), data);
  }
}
