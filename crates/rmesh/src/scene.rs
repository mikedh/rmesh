use nalgebra::Matrix4;

use crate::geometry::Geometry;

#[derive(Default)]
pub struct Light {
    // Add light properties as needed
    pub name: String,
}

#[derive(Default)]
pub enum SceneNodeKind {
    #[default]
    GEOMETRY = 1,
    CAMERA = 2,
    LIGHT = 3,
    CUSTOM = 4,
}

#[derive(Default)]
pub struct SceneNode {
    // A human readable name for the node
    pub name: String,

    // the indices of the child nodes in the SceneGraph's nodes vector
    pub children: Vec<usize>,

    // the transform from the parent node to this node
    // or None if they are at the same position (i.e. identity)
    pub transform: Option<Matrix4<f64>>,

    // The type of this node.
    pub kind: SceneNodeKind,

    // Indices into the Scene's geometry, lights, camera, or custom
    // user-tracked property depending on the value of `kind`
    pub index: Vec<usize>,
}

#[derive(Default)]
pub struct SceneGraph {
    // The root node index in the nodes vector
    pub root: usize,

    // A flat list of nodes in the scene.
    pub nodes: Vec<SceneNode>,
}

impl SceneGraph {
    pub fn new() -> Self {
        SceneGraph::default()
    }

    pub fn add_node(&mut self, node: SceneNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }
}

#[derive(Default)]
pub struct Scene {
    // geometry in the scene
    pub geometry: Vec<Geometry>,

    pub lights: Vec<Light>,

    // Instances of the scene graph
    pub graph: SceneGraph,

    // The node index of the camera.
    pub camera: usize,
}

impl Scene {
    pub fn new() -> Self {
        Scene::default()
    }

    pub fn add_geometry(&mut self, geom: Geometry) -> usize {
        let index = self.geometry.len();
        self.geometry.push(geom);
        index
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::creation;

    #[test]
    fn test_scene_basic() {
        let mut scene = Scene::new();

        let mesh = creation::create_box(&[1.0, 1.0, 1.0]);
        let geom_index = scene.add_geometry(Geometry::Mesh(Box::new(mesh)));

        let root_node = SceneNode {
            name: "root".to_string(),
            children: Vec::new(),
            transform: None,
            index: vec![geom_index],
            kind: SceneNodeKind::GEOMETRY,
        };

        let root_index = scene.graph.add_node(root_node);
        scene.graph.root = root_index;

        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 1);
        assert_eq!(scene.graph.root, 0);
        assert_eq!(scene.graph.nodes[0].name, "root");
        assert_eq!(scene.graph.nodes[0].index.len(), 1);
    }
}
