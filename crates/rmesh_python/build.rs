fn main() {
    // Set rpath for libpython so tests can find it at runtime
    // This reads the Python home from .venv/pyvenv.cfg (created by uv)
    let venv_cfg = std::path::Path::new("../../.venv/pyvenv.cfg");
    if venv_cfg.exists() {
        let content = std::fs::read_to_string(venv_cfg).unwrap();
        for line in content.lines() {
            if let Some(home) = line.strip_prefix("home = ") {
                // home is like /path/to/python/bin, we want /path/to/python/lib
                if let Some(bin_dir) = std::path::Path::new(home.trim()).parent() {
                    let lib_dir = bin_dir.join("lib");
                    if lib_dir.exists() {
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
                    }
                }
                break;
            }
        }
        println!("cargo:rerun-if-changed=../../.venv/pyvenv.cfg");
    }
}
