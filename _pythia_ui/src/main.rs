//! pythia-ui: the plugin's dispatcher as a program (README.md).
//!
//! With no arguments it serves the protocol: one JSON request per line on
//! stdin, one reply per line on stdout. `info` and `screenshot` are for trying
//! it by hand.

use pythia_ui::{json, Failure, ScreenshotOptions};
use std::io::{BufRead, Write};
use std::process::ExitCode;

const USAGE: &str = "\
usage: pythia-ui              serve: one JSON request per line on stdin, one reply
                              per line on stdout (see README.md)
       pythia-ui info         print the info reply
       pythia-ui screenshot [--display ID] [--capture PATH] [--no-cursor] FILE.png
                              take a screenshot and write it to FILE.png; PATH is
                              auto, screencapturekit or coregraphics
";

fn main() -> ExitCode {
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.first().map(String::as_str) {
    None => serve(),
    Some("info") if args.len() == 1 => {
      let reply = pythia_ui::dispatch(br#"{"op": "info"}"#);
      println!("{}", String::from_utf8_lossy(&reply));
      ExitCode::SUCCESS
    }
    Some("screenshot") => screenshot(&args[1..]),
    Some("-h" | "--help") => {
      print!("{USAGE}");
      ExitCode::SUCCESS
    }
    Some("--version") => {
      println!("pythia-ui {}", pythia_ui::VERSION);
      ExitCode::SUCCESS
    }
    Some(_) => {
      eprint!("{USAGE}");
      ExitCode::from(2)
    }
  }
}

/// Answers each request line until stdin ends; blank lines are skipped.
fn serve() -> ExitCode {
  let stdin = std::io::stdin();
  let mut stdout = std::io::stdout().lock();
  for line in stdin.lock().split(b'\n') {
    let line = match line {
      Ok(line) => line,
      Err(error) => {
        eprintln!("pythia-ui: reading stdin: {error}");
        return ExitCode::FAILURE;
      }
    };
    if line.iter().all(u8::is_ascii_whitespace) {
      continue;
    }
    let mut reply = pythia_ui::dispatch(&line);
    reply.push(b'\n');
    if stdout
      .write_all(&reply)
      .and_then(|()| stdout.flush())
      .is_err()
    {
      // The reader is gone, so no one is left to answer.
      return ExitCode::SUCCESS;
    }
  }
  ExitCode::SUCCESS
}

fn screenshot(args: &[String]) -> ExitCode {
  let (options, path) = match screenshot_args(args) {
    Ok(parsed) => parsed,
    Err(message) => {
      eprint!("pythia-ui: {message}\n{USAGE}");
      return ExitCode::from(2);
    }
  };
  let shot = match pythia_ui::screenshot(&options) {
    Ok(shot) => shot,
    Err(failure) => return fail(&failure),
  };
  if let Err(error) = std::fs::write(&path, &shot.png) {
    eprintln!("pythia-ui: can't write {path}: {error}");
    return ExitCode::FAILURE;
  }
  let mut fields = pythia_ui::screenshot_fields(&shot, false);
  fields.push(("file", path.as_str().into()));
  println!(
    "{}",
    String::from_utf8_lossy(&json::object(fields).to_vec())
  );
  ExitCode::SUCCESS
}

fn fail(failure: &Failure) -> ExitCode {
  eprintln!("pythia-ui: {}: {}", failure.kind.name(), failure.error);
  ExitCode::FAILURE
}

fn screenshot_args(args: &[String]) -> Result<(ScreenshotOptions, String), String> {
  let mut options = ScreenshotOptions::default();
  let mut path = None;
  let mut args = args.iter();
  while let Some(arg) = args.next() {
    match arg.as_str() {
      "--display" => {
        let id = args.next().ok_or("--display needs a display ID")?;
        let id = id
          .parse()
          .map_err(|_| format!("--display: not a display ID: {id:?}"))?;
        options.display = Some(id);
      }
      "--capture" => {
        let name = args.next().ok_or("--capture needs a path")?;
        options.capture = pythia_ui::parse_capture(name).map_err(|failure| failure.error)?;
      }
      "--no-cursor" => options.cursor = false,
      _ if arg.starts_with('-') && arg.len() > 1 => return Err(format!("unknown option {arg:?}")),
      _ if path.is_none() => path = Some(arg.clone()),
      _ => return Err(format!("one file only, not also {arg:?}")),
    }
  }
  let path = path.ok_or("screenshot needs a file to write, such as shot.png")?;
  Ok((options, path))
}
