//! EXPRESS schema → Rust codegen CLI.
//!
//! Usage:
//!   cargo run -p codegen -- ./reference/APs/10303-214e3-aim-long.exp -o src/boundary/step/ap214.rs

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use codegen::{generate, parse_express, strip_comments_and_lower};

#[derive(Parser)]
#[command(name = "express-gen")]
#[command(about = "Generate Rust code from EXPRESS schemas")]
struct Args {
    /// Path to the EXPRESS (.exp) schema file
    input: PathBuf,

    /// Output Rust file path
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read and preprocess the EXPRESS file
    let content = fs::read(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let processed = strip_comments_and_lower(&content);

    // Parse the schema
    let (remaining, mut syntax) =
        parse_express(&processed).map_err(|e| anyhow::anyhow!("Parse error: {e:?}"))?;

    if !remaining.trim().is_empty() {
        eprintln!(
            "Warning: unparsed content remaining: {}...",
            &remaining[..remaining.len().min(100)]
        );
    }

    // Generate Rust code
    let rust_code = generate(&mut syntax)?;

    // Write output
    match args.output {
        Some(path) => {
            fs::write(&path, &rust_code)
                .with_context(|| format!("Failed to write {}", path.display()))?;
            eprintln!("Generated {} bytes to {}", rust_code.len(), path.display());
        }
        None => {
            print!("{rust_code}");
        }
    }

    Ok(())
}
