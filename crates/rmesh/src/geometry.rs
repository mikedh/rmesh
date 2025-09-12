use crate::mesh::Trimesh;
use crate::path::Path;

pub enum Geometry {
    Mesh(Trimesh),
    Path(Path),
}
