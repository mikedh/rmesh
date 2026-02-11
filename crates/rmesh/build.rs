use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile all shaders (render + compute) when GPU features are enabled
    if std::env::var("CARGO_FEATURE_WGPU").is_ok() {
        compile_shaders()?;
    }

    Ok(())
}

fn compile_shaders() -> Result<(), Box<dyn std::error::Error>> {
    use shaderloom::Shaderloom;

    let loom_path = "shader_src/loom.lua";
    if Path::new(loom_path).exists() {
        Shaderloom::new().build_from_file(loom_path)?;
        println!("cargo:rerun-if-changed=shader_src/");
    } else {
        eprintln!("Warning: {loom_path} not found, skipping shader build");
    }
    Ok(())
}
