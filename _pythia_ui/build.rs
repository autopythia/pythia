//! Sets the plugin's install name (LC_ID_DYLIB) to @rpath/libpythia_ui.dylib,
//! rather than the linker's default, the absolute path it was built at.

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
    // As ld64 options: .cargo/config.toml links with rust-lld itself, which
    // takes them as given (a linker driver such as cc would need -Wl,).
    println!("cargo:rustc-cdylib-link-arg=-install_name");
    println!("cargo:rustc-cdylib-link-arg=@rpath/libpythia_ui.dylib");
  }
}
