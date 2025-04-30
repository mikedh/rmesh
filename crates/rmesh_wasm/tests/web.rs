//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use std::{assert_eq, println};

use wasm_bindgen_test::*;


// wasm_bindgen_test_configure!(run_in_browser);



#[wasm_bindgen_test]
fn load_mesh() {
    let stl_data = include_bytes!("../../../test/data/unit_cube.STL");
    let file_type = "stl";
    let mesh = rmesh_wasm::load_mesh_ex(stl_data, file_type).unwrap();

    assert!(mesh.contains("Trimesh"));
}
