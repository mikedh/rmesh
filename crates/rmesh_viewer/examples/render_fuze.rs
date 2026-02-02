use std::path::Path;

use rmesh::exchange::{FileResolver, FileType, load};
use rmesh_viewer::{RenderOptions, SceneViewer};

fn main() {
    let obj_path = Path::new("test/data/fuze.obj");
    let data = std::fs::read(obj_path).expect("failed to read fuze.obj");
    let resolver = FileResolver::from_file_path(obj_path);
    let scene = load(&data, Some(FileType::OBJ), Some(&resolver)).unwrap();

    let options = RenderOptions {
        width: 1280,
        height: 720,
        background: [0.15, 0.15, 0.18],
    };
    let rgba = scene.render_to_image(&options);

    let img =
        image::RgbaImage::from_raw(options.width, options.height, rgba).expect("render failed");
    img.save("fuze_render.png").expect("failed to save PNG");
    println!(
        "Saved fuze_render.png ({}x{})",
        options.width, options.height
    );
}
