use nalgebra::Matrix4;

use crate::geometry::Geometry;

#[derive(Default)]
pub struct SceneNode {
    // A human readable name for the node
    pub name: String,

    // the indices of the child nodes in the SceneGraph's nodes vector
    pub children: Vec<usize>,

    // the transform from the parent node to this node
    // or None if they are at the same position (i.e. identity)
    pub transform: Option<Matrix4<f64>>,

    // Indices into the Scene's geometry
    pub geometry: Vec<usize>,
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

    // Instances of the scene graph
    pub graph: SceneGraph,
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
            geometry: vec![geom_index],
        };

        let root_index = scene.graph.add_node(root_node);
        scene.graph.root = root_index;

        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 1);
        assert_eq!(scene.graph.root, 0);
        assert_eq!(scene.graph.nodes[0].name, "root");
        assert_eq!(scene.graph.nodes[0].geometry.len(), 1);
    }
}
