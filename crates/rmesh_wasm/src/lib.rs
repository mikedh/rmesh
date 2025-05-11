mod utils;

use wasm_bindgen::prelude::*;

use rmesh::exchange::{MeshFormat, load_mesh};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, rmesh-wasm!");
}

#[wasm_bindgen]
pub fn load_mesh_ex(file_data: &[u8], file_type: &str) -> Result<String, String> {
    let mesh_format = MeshFormat::from_string(file_type).map_err(|e| e.to_string())?;
    let mesh = load_mesh(file_data, mesh_format).map_err(|e| e.to_string())?;
    // just print the debug info
    Ok(format!("{mesh:?}"))
}
