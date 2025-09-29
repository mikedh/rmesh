use crate::mesh::Trimesh;
use crate::path::Path;

pub enum Geometry {
    Mesh(Box<Trimesh>),
    Path(Path),
}
