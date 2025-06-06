mod utils;

use wasm_bindgen::prelude::*;
use rmesh_macro::wasm_result;

use rmesh::exchange::{MeshFormat, load_mesh};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, rmesh-wasm!");
}

#[wasm_result]
pub fn load_mesh_ex(file_data: &[u8], file_type: &str) -> String {
    let mesh_format = MeshFormat::from_string(file_type)?;
    let mesh = load_mesh(file_data, mesh_format)?;
    // just print the debug info
    Ok(format!("{mesh:?}"))
}
