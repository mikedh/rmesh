# Schemas

Schema files for mesh formats.

## glTF

The `glTF/` directory is a clone of the [Khronos glTF repo](https://github.com/KhronosGroup/glTF) containing JSON Schema files for glTF 2.0 and extensions.

### Regenerating Rust types

To regenerate `src/schemas/gltf_2/mod.rs` from the JSON Schema files:

```bash
cargo run --manifest-path crates/rmesh/schemas/codegen/Cargo.toml
```

Or from the codegen directory:

```bash
cd crates/rmesh/schemas/codegen && cargo run
```

## Collada

Compressed XSD schemas for Collada 1.4.1 and 1.5.
