use std::process::Command;

fn main() {
    // Add libpython directory to linker search path
    let output = Command::new("uv")
        .args([
            "run",
            "python",
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
        ])
        .output()
        .expect("Failed to run `uv run python`. Is uv installed?");

    if output.status.success() {
        let libdir = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !libdir.is_empty() && libdir != "None" {
            println!("cargo:rustc-link-search=native={}", libdir);
        }
    }
}
