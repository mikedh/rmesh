use anyhow::Result;
use shaderloom::Shaderloom;

fn main() -> Result<()> {
    println!("Building shaders with shaderloom...");

    let shaderloom = Shaderloom::new();

    let loom_path = "shader_src/loom.lua";
    if std::path::Path::new(loom_path).exists() {
        shaderloom.build_from_file(loom_path)?;

        println!("cargo:rerun-if-changed=shader_src/");
        println!("cargo:rerun-if-changed=shader_src/include/");
    } else {
        eprintln!("Warning: {loom_path} not found, skipping shader build");
    }

    Ok(())
}
