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
use nalgebra::{Matrix4, Point3};

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

    /// Add geometry with optional placement transforms.
    ///
    /// If `transforms` is `Some`, each matrix creates a SceneGraph node
    /// referencing the shared geometry. If `None`, one node with identity
    /// transform is created.
    ///
    /// Returns the unique name assigned to the geometry.
    pub fn add(
        &mut self,
        name: &str,
        geom: Geometry,
        transforms: Option<&[Matrix4<f64>]>,
    ) -> String {
        // Ensure a root node exists (Custom kind so it doesn't inflate geometry counts)
        if self.graph.nodes.is_empty() {
            self.graph.nodes.push(SceneNode {
                kind: SceneNodeKind::Custom,
                ..Default::default()
            });
        }

        let (actual_name, geom_index) = self.add_geometry(name, geom);
        let root = self.graph.root;

        match transforms {
            Some(xforms) if !xforms.is_empty() => {
                for t in xforms {
                    let node = SceneNode {
                        name: actual_name.clone(),
                        transform: Some(*t),
                        kind: SceneNodeKind::Geometry,
                        index: vec![geom_index],
                        ..Default::default()
                    };
                    let idx = self.graph.add_node(node);
                    self.graph.nodes[root].children.push(idx);
                }
            }
            _ => {
                let node = SceneNode {
                    name: actual_name.clone(),
                    kind: SceneNodeKind::Geometry,
                    index: vec![geom_index],
                    ..Default::default()
                };
                let idx = self.graph.add_node(node);
                self.graph.nodes[root].children.push(idx);
            }
        }

        actual_name
    }

    /// Fill holes in all mesh geometry.
    ///
    /// For each `Geometry::Mesh` entry:
    /// - If already watertight, keep as-is.
    /// - Otherwise, replace with the result of `fill_holes()`.
    /// - If still not watertight after filling and `drop` is true, remove it.
    ///
    /// Non-mesh geometry is always kept. When geometry is removed, scene graph
    /// node indices are remapped accordingly.
    pub fn fill_holes(&mut self, drop: bool) {
        let keys: Vec<String> = self.geometry.keys().cloned().collect();
        let mut to_remove = std::collections::HashSet::new();

        for key in &keys {
            let Some(geom) = self.geometry.get_mut(key) else {
                continue;
            };
            if let Geometry::Mesh(mesh) = geom {
                if mesh.is_watertight() {
                    continue;
                }
                let filled = mesh.fill_holes();
                if !filled.is_watertight() && drop {
                    to_remove.insert(key.clone());
                } else {
                    **mesh = filled;
                }
            }
        }

        if to_remove.is_empty() {
            return;
        }

        // Build old_index -> new_index mapping
        let mut old_to_new: Vec<Option<usize>> = Vec::with_capacity(keys.len());
        let mut new_idx = 0usize;
        for key in &keys {
            if to_remove.contains(key) {
                old_to_new.push(None);
            } else {
                old_to_new.push(Some(new_idx));
                new_idx += 1;
            }
        }

        // Remove geometry entries
        for key in &to_remove {
            self.geometry.shift_remove(key);
        }

        // Fix scene graph: remap or remove geometry indices
        for node in &mut self.graph.nodes {
            if node.kind != SceneNodeKind::Geometry {
                continue;
            }
            node.index = node
                .index
                .iter()
                .filter_map(|&old| {
                    if old < old_to_new.len() {
                        old_to_new[old]
                    } else {
                        None
                    }
                })
                .collect();
        }
    }

    /// Compute the world-space axis-aligned bounding box of all geometry.
    ///
    /// Walks the scene graph to apply world transforms. If the graph is empty
    /// but geometry exists, uses identity transforms. Returns `None` if the
    /// scene has no geometry with valid bounds.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        let mut global_min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut global_max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        let mut found = false;

        let geometry_names: Vec<String> = self.geometry.keys().cloned().collect();

        // Try scene graph first
        let mut used_graph = false;
        self.graph.walk(|_idx, node, world_transform| {
            if node.kind != SceneNodeKind::Geometry {
                return;
            }
            for &geom_idx in &node.index {
                if geom_idx >= geometry_names.len() {
                    continue;
                }
                let name = &geometry_names[geom_idx];
                if let Some(geom) = self.geometry.get(name) {
                    used_graph = true;
                    if let Some((local_min, local_max)) = geom.bounds() {
                        transform_bounds(
                            &local_min,
                            &local_max,
                            world_transform,
                            &mut global_min,
                            &mut global_max,
                        );
                        found = true;
                    }
                }
            }
        });

        // Fallback: if graph didn't reference geometry, use identity
        if !used_graph {
            let identity = Matrix4::identity();
            for geom in self.geometry.values() {
                if let Some((local_min, local_max)) = geom.bounds() {
                    transform_bounds(
                        &local_min,
                        &local_max,
                        &identity,
                        &mut global_min,
                        &mut global_max,
                    );
                    found = true;
                }
            }
        }

        if found {
            Some((global_min, global_max))
        } else {
            None
        }
    }

    /// Get the extents of the scene bounding box (max - min per axis).
    pub fn extents(&self) -> Option<[f64; 3]> {
        self.bounds()
            .map(|(min, max)| [max.x - min.x, max.y - min.y, max.z - min.z])
    }
}

/// Transform the 8 corners of an AABB and update global min/max.
fn transform_bounds(
    local_min: &Point3<f64>,
    local_max: &Point3<f64>,
    transform: &Matrix4<f64>,
    global_min: &mut Point3<f64>,
    global_max: &mut Point3<f64>,
) {
    let corners = [
        Point3::new(local_min.x, local_min.y, local_min.z),
        Point3::new(local_max.x, local_min.y, local_min.z),
        Point3::new(local_min.x, local_max.y, local_min.z),
        Point3::new(local_max.x, local_max.y, local_min.z),
        Point3::new(local_min.x, local_min.y, local_max.z),
        Point3::new(local_max.x, local_min.y, local_max.z),
        Point3::new(local_min.x, local_max.y, local_max.z),
        Point3::new(local_max.x, local_max.y, local_max.z),
    ];
    for corner in &corners {
        let v = transform * nalgebra::Vector4::new(corner.x, corner.y, corner.z, 1.0);
        let p = Point3::new(v.x, v.y, v.z);
        global_min.x = global_min.x.min(p.x);
        global_min.y = global_min.y.min(p.y);
        global_min.z = global_min.z.min(p.z);
        global_max.x = global_max.x.max(p.x);
        global_max.y = global_max.y.max(p.y);
        global_max.z = global_max.z.max(p.z);
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
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(
                1.0, 0.0, 0.0,
            ))),
            ..Default::default()
        };
        let child_idx = graph.add_node(child);

        let root = SceneNode {
            name: "root".to_string(),
            children: vec![child_idx],
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(
                0.0, 1.0, 0.0,
            ))),
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
    fn test_scene_add() {
        let mut scene = Scene::new();
        let mesh = creation::create_box(&[1.0, 1.0, 1.0]);
        let name = scene.add("cube", Geometry::Mesh(Box::new(mesh)), None);
        assert_eq!(name, "cube");
        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 2); // root + 1 geometry node
        assert!(scene.bounds().is_some());
    }

    #[test]
    fn test_scene_add_transforms() {
        let mut scene = Scene::new();
        let mesh = creation::create_box(&[1.0, 1.0, 1.0]);
        let t1 = Matrix4::new_translation(&nalgebra::Vector3::new(5.0, 0.0, 0.0));
        let t2 = Matrix4::new_translation(&nalgebra::Vector3::new(0.0, 5.0, 0.0));
        let name = scene.add("cube", Geometry::Mesh(Box::new(mesh)), Some(&[t1, t2]));
        assert_eq!(name, "cube");
        assert_eq!(scene.geometry.len(), 1);
        assert_eq!(scene.graph.nodes.len(), 3); // root + 2 transform nodes
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
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(
                5.0, 0.0, 0.0,
            ))),
            ..Default::default()
        };
        let child_idx = graph.add_node(child);

        let root = SceneNode {
            name: "root".to_string(),
            children: vec![child_idx],
            transform: Some(Matrix4::new_translation(&nalgebra::Vector3::new(
                0.0, 10.0, 0.0,
            ))),
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

    #[test]
    fn test_scene_fill_holes_fixable() {
        let mut scene = Scene::new();
        // Add a watertight cube
        let cube = creation::create_box(&[1.0, 1.0, 1.0]);
        scene.add("cube", Geometry::Mesh(Box::new(cube)), None);

        // Add an open cube (missing one face)
        let open_cube = creation::create_box(&[2.0, 2.0, 2.0]);
        let open = open_cube.submesh(&(0..10).collect::<Vec<_>>());
        assert!(!open.is_watertight());
        scene.add("open", Geometry::Mesh(Box::new(open)), None);

        assert_eq!(scene.geometry.len(), 2);
        scene.fill_holes(false);
        assert_eq!(
            scene.geometry.len(),
            2,
            "no geometry should be removed with drop=false"
        );

        // Both should now be watertight
        for (name, geom) in &scene.geometry {
            if let Geometry::Mesh(m) = geom {
                assert!(
                    m.is_watertight(),
                    "mesh '{name}' should be watertight after fill_holes"
                );
            }
        }
    }

    #[test]
    fn test_scene_fill_holes_drop() {
        use crate::mesh::Trimesh;

        let mut scene = Scene::new();
        // Add a watertight cube
        let cube = creation::create_box(&[1.0, 1.0, 1.0]);
        scene.add("cube", Geometry::Mesh(Box::new(cube)), None);

        // Create a non-manifold mesh (3 triangles sharing one edge).
        // This can never become watertight by filling boundary loops.
        let tri = Trimesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, -1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
            None,
            None,
        )
        .unwrap();
        scene.add("triangle", Geometry::Mesh(Box::new(tri)), None);

        assert_eq!(scene.geometry.len(), 2);
        scene.fill_holes(true);

        // Triangle can't be made watertight → should be dropped
        assert_eq!(scene.geometry.len(), 1);
        assert!(scene.geometry.contains_key("cube"));

        // Remaining graph nodes referencing geometry should have valid indices
        for node in &scene.graph.nodes {
            if node.kind == SceneNodeKind::Geometry {
                for &idx in &node.index {
                    assert!(idx < scene.geometry.len(), "stale geometry index {idx}");
                }
            }
        }
    }
}
