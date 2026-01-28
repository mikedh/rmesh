//! Scene graph and related types for organizing 3D content.

mod animation;
mod camera;
mod light;
mod trackball;

pub use animation::*;
pub use camera::*;
pub use light::*;
pub use trackball::*;

use indexmap::IndexMap;
use nalgebra::Matrix4;

use crate::geometry::Geometry;

/// The type of content a scene node references.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SceneNodeKind {
    /// Node references geometry.
    #[default]
    Geometry,
    /// Node references a camera.
    Camera,
    /// Node references a light.
    Light,
    /// Node references custom user data.
    Custom,
}

/// A node in the scene graph.
#[derive(Debug, Clone, Default)]
pub struct SceneNode {
    /// Human-readable name for the node.
    pub name: String,

    /// Indices of child nodes in the SceneGraph's nodes vector.
    pub children: Vec<usize>,

    /// Transform from parent node to this node.
    /// None indicates identity transform.
    pub transform: Option<Matrix4<f64>>,

    /// The type of content this node references.
    pub kind: SceneNodeKind,

    /// Indices into the Scene's geometry, lights, cameras, or custom arrays
    /// depending on the value of `kind`.
    pub index: Vec<usize>,
}

/// A hierarchical scene graph.
#[derive(Debug, Clone, Default)]
pub struct SceneGraph {
    /// The root node index in the nodes vector.
    pub root: usize,

    /// Flat list of all nodes in the scene.
    pub nodes: Vec<SceneNode>,
}

impl SceneGraph {
    /// Create a new empty scene graph.
    pub fn new() -> Self {
        SceneGraph::default()
    }

    /// Add a node to the graph and return its index.
    pub fn add_node(&mut self, node: SceneNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    /// Walk the scene graph depth-first, calling the visitor with each node
    /// and its accumulated world transform.
    pub fn walk<F>(&self, mut visitor: F)
    where
        F: FnMut(usize, &SceneNode, &Matrix4<f64>),
    {
        if self.nodes.is_empty() {
            return;
        }
        self.walk_recursive(self.root, &Matrix4::identity(), &mut visitor);
    }

    fn walk_recursive<F>(&self, node_index: usize, parent_transform: &Matrix4<f64>, visitor: &mut F)
    where
        F: FnMut(usize, &SceneNode, &Matrix4<f64>),
    {
        let node = &self.nodes[node_index];
        let world_transform = match &node.transform {
            Some(t) => parent_transform * t,
            None => *parent_transform,
        };

        visitor(node_index, node, &world_transform);

        for &child_index in &node.children {
            self.walk_recursive(child_index, &world_transform, visitor);
        }
    }

    /// Find a node by name, returning its index if found.
    pub fn find_by_name(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    /// Compute the world transform for a node by traversing from root.
    pub fn world_transform(&self, target: usize) -> Matrix4<f64> {
        // Build parent map
        let mut parent: Vec<Option<usize>> = vec![None; self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            for &child in &node.children {
                parent[child] = Some(i);
            }
        }

        // Walk from target to root, collecting transforms
        let mut transforms = Vec::new();
        let mut current = target;
        loop {
            if let Some(t) = &self.nodes[current].transform {
                transforms.push(*t);
            }
            match parent[current] {
                Some(p) => current = p,
                None => break,
            }
        }

        // Multiply transforms from root to target
        transforms
            .into_iter()
            .rev()
            .fold(Matrix4::identity(), |acc, t| acc * t)
    }
}

/// A complete scene with geometry, lights, cameras, and a scene graph.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    /// Geometry in the scene, keyed by name. Order is preserved.
    pub geometry: IndexMap<String, Geometry>,

    /// Lights in the scene.
    pub lights: Vec<Light>,

    /// Cameras in the scene.
    pub cameras: Vec<Camera>,

    /// The scene graph organizing nodes.
    pub graph: SceneGraph,

    /// Index of the active camera in the cameras array.
    pub active_camera: Option<usize>,

    /// Animations that can be played.
    pub animations: Vec<Animation>,
}

/// Generate a unique name not already in the geometry map.
/// Pattern: "name" -> "name_1" -> "name_2" ...
/// Empty name becomes "geometry_0", "geometry_1", ...
fn unique_name(start: &str, contains: &IndexMap<String, Geometry>) -> String {
    if !start.is_empty() && !contains.contains_key(start) {
        return start.to_string();
    }

    let base = if start.is_empty() { "geometry" } else { start };
    for i in 0.. {
        let candidate = format!("{}_{}", base, i);
        if !contains.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

impl Scene {
    /// Create a new empty scene.
    pub fn new() -> Self {
        Scene::default()
    }

    /// Add geometry with unique name handling.
    /// If name exists, appends "_0", "_1", etc. until unique.
    /// Returns the actual name used and the index.
    pub fn add_geometry(&mut self, name: &str, geom: Geometry) -> (String, usize) {
        let unique = unique_name(name, &self.geometry);
        let index = self.geometry.len();
        self.geometry.insert(unique.clone(), geom);
        (unique, index)
    }

    /// Add a light to the scene and return its index.
    pub fn add_light(&mut self, light: Light) -> usize {
        let index = self.lights.len();
        self.lights.push(light);
        index
    }

    /// Add a camera to the scene and return its index.
    pub fn add_camera(&mut self, camera: Camera) -> usize {
        let index = self.cameras.len();
        self.cameras.push(camera);
        index
    }

    /// Add an animation to the scene and return its index.
    pub fn add_animation(&mut self, animation: Animation) -> usize {
        let index = self.animations.len();
        self.animations.push(animation);
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
        let (name, geom_index) = scene.add_geometry("cube", Geometry::Mesh(Box::new(mesh)));

        let root_node = SceneNode {
            name: "root".to_string(),
            children: Vec::new(),
            transform: None,
            index: vec![geom_index],
            kind: SceneNodeKind::Geometry,
        };

        let root_index = scene.graph.add_node(root_node);
        scene.graph.root = root_index;

        assert_eq!(name, "cube");
        assert_eq!(scene.geometry.len(), 1);
        assert!(scene.geometry.contains_key("cube"));
        assert_eq!(scene.graph.nodes.len(), 1);
        assert_eq!(scene.graph.root, 0);
        assert_eq!(scene.graph.nodes[0].name, "root");
        assert_eq!(scene.graph.nodes[0].index.len(), 1);
    }

    #[test]
    fn test_unique_name() {
        let mut scene = Scene::new();

        let mesh1 = creation::create_box(&[1.0, 1.0, 1.0]);
        let (name1, _) = scene.add_geometry("cube", Geometry::Mesh(Box::new(mesh1)));
        assert_eq!(name1, "cube");

        // Adding with same name should generate unique
        let mesh2 = creation::create_box(&[2.0, 2.0, 2.0]);
        let (name2, _) = scene.add_geometry("cube", Geometry::Mesh(Box::new(mesh2)));
        assert_eq!(name2, "cube_0");

        // Adding with empty name should generate "geometry_0"
        let mesh3 = creation::create_box(&[3.0, 3.0, 3.0]);
        let (name3, _) = scene.add_geometry("", Geometry::Mesh(Box::new(mesh3)));
        assert_eq!(name3, "geometry_0");

        // Adding another empty should generate "geometry_1"
        let mesh4 = creation::create_box(&[4.0, 4.0, 4.0]);
        let (name4, _) = scene.add_geometry("", Geometry::Mesh(Box::new(mesh4)));
        assert_eq!(name4, "geometry_1");
    }

    #[test]
    fn test_scene_graph_walk() {
        let mut graph = SceneGraph::new();

        let child = SceneNode {
            name: "child".to_string(),
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 0.0, 0.0))),
            ..Default::default()
        };
        let child_idx = graph.add_node(child);

        let root = SceneNode {
            name: "root".to_string(),
            children: vec![child_idx],
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(0.0, 1.0, 0.0))),
            ..Default::default()
        };
        graph.add_node(root);
        graph.root = 1;

        let mut visited = Vec::new();
        graph.walk(|idx, node, transform| {
            visited.push((idx, node.name.clone(), transform[(0, 3)], transform[(1, 3)]));
        });

        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0].1, "root");
        assert!((visited[0].2 - 0.0).abs() < 1e-10); // root x = 0
        assert!((visited[0].3 - 1.0).abs() < 1e-10); // root y = 1
        assert_eq!(visited[1].1, "child");
        assert!((visited[1].2 - 1.0).abs() < 1e-10); // child x = 0 + 1
        assert!((visited[1].3 - 1.0).abs() < 1e-10); // child y = 1 + 0
    }

    #[test]
    fn test_find_by_name() {
        let mut graph = SceneGraph::new();
        graph.add_node(SceneNode {
            name: "first".to_string(),
            ..Default::default()
        });
        graph.add_node(SceneNode {
            name: "second".to_string(),
            ..Default::default()
        });

        assert_eq!(graph.find_by_name("first"), Some(0));
        assert_eq!(graph.find_by_name("second"), Some(1));
        assert_eq!(graph.find_by_name("third"), None);
    }

    #[test]
    fn test_world_transform() {
        let mut graph = SceneGraph::new();

        let grandchild = SceneNode {
            name: "grandchild".to_string(),
            transform: Some(Matrix4::new_scaling(2.0)),
            ..Default::default()
        };
        let grandchild_idx = graph.add_node(grandchild);

        let child = SceneNode {
            name: "child".to_string(),
            children: vec![grandchild_idx],
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(5.0, 0.0, 0.0))),
            ..Default::default()
        };
        let child_idx = graph.add_node(child);

        let root = SceneNode {
            name: "root".to_string(),
            children: vec![child_idx],
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(0.0, 10.0, 0.0))),
            ..Default::default()
        };
        graph.add_node(root);
        graph.root = 2;

        let world = graph.world_transform(grandchild_idx);
        // Translation should be (5, 10, 0) and scale 2
        assert!((world[(0, 3)] - 5.0).abs() < 1e-10);
        assert!((world[(1, 3)] - 10.0).abs() < 1e-10);
        assert!((world[(0, 0)] - 2.0).abs() < 1e-10);
    }
}
