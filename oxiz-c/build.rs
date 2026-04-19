fn main() {
    if std::env::var("OXIZ_C_GEN_HEADER").as_deref() == Ok("1") {
        let crate_dir = match std::env::var("CARGO_MANIFEST_DIR") {
            Ok(d) => d,
            Err(e) => {
                eprintln!("oxiz-c build: CARGO_MANIFEST_DIR not set: {e}");
                return;
            }
        };
        let config = cbindgen::Config::from_file(
            std::path::Path::new(&crate_dir).join("cbindgen.toml"),
        )
        .unwrap_or_default();
        match cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .generate()
        {
            Ok(bindings) => {
                bindings.write_to_file(
                    std::path::Path::new(&crate_dir).join("include/oxiz.h"),
                );
            }
            Err(e) => {
                eprintln!("oxiz-c build: cbindgen failed: {e}");
            }
        }
    }
}
