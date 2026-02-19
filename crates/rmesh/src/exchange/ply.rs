use anyhow::{Result, anyhow};
use nalgebra::{Point3, Vector2, Vector3, Vector4};

use crate::attributes::{Attributes, LoadSource, Material, SimpleMaterial};
use crate::creation::Triangulator;
use crate::image::LazyImage;
use crate::mesh::Trimesh;
use crate::resolvers::Resolver;

/// PLY encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

/// PLY scalar data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyType {
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "char" | "int8" => Ok(PlyType::Char),
            "uchar" | "uint8" => Ok(PlyType::UChar),
            "short" | "int16" => Ok(PlyType::Short),
            "ushort" | "uint16" => Ok(PlyType::UShort),
            "int" | "int32" => Ok(PlyType::Int),
            "uint" | "uint32" => Ok(PlyType::UInt),
            "float" | "float32" => Ok(PlyType::Float),
            "double" | "float64" => Ok(PlyType::Double),
            _ => Err(anyhow!("Unknown PLY type: `{}`", s)),
        }
    }

    /// Size in bytes for binary reading.
    fn size(self) -> usize {
        match self {
            PlyType::Char | PlyType::UChar => 1,
            PlyType::Short | PlyType::UShort => 2,
            PlyType::Int | PlyType::UInt | PlyType::Float => 4,
            PlyType::Double => 8,
        }
    }

    /// Read a single value from bytes at the given offset with the given endianness.
    /// Returns the value as f64 and the number of bytes consumed.
    fn read_binary(self, data: &[u8], offset: usize, big_endian: bool) -> Result<(f64, usize)> {
        let size = self.size();
        if offset + size > data.len() {
            return Err(anyhow!("Unexpected end of binary PLY data"));
        }
        let b = &data[offset..offset + size];

        /// Read a multi-byte primitive from a byte slice with endian dispatch.
        macro_rules! read_endian {
            ($ty:ty, $bytes:expr, $big:expr) => {{
                let arr = <[u8; std::mem::size_of::<$ty>()]>::try_from($bytes).unwrap();
                f64::from(if $big {
                    <$ty>::from_be_bytes(arr)
                } else {
                    <$ty>::from_le_bytes(arr)
                })
            }};
        }

        let val = match self {
            PlyType::Char => {
                #[allow(clippy::cast_possible_wrap)]
                let v = b[0] as i8;
                f64::from(v)
            }
            PlyType::UChar => f64::from(b[0]),
            PlyType::Short => read_endian!(i16, b, big_endian),
            PlyType::UShort => read_endian!(u16, b, big_endian),
            PlyType::Int => read_endian!(i32, b, big_endian),
            PlyType::UInt => read_endian!(u32, b, big_endian),
            PlyType::Float => read_endian!(f32, b, big_endian),
            PlyType::Double => read_endian!(f64, b, big_endian),
        };
        Ok((val, size))
    }

    /// Write a value for ASCII output, using integer format for integer types.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn write_ascii(self, val: f64, out: &mut String) {
        use std::fmt::Write;
        match self {
            PlyType::Char | PlyType::Short | PlyType::Int => write!(out, "{}", val as i64).unwrap(),
            PlyType::UChar | PlyType::UShort | PlyType::UInt => {
                write!(out, "{}", val as u64).unwrap();
            }
            PlyType::Float | PlyType::Double => write!(out, "{}", val).unwrap(),
        }
    }

    /// PLY type name string.
    fn name(self) -> &'static str {
        match self {
            PlyType::Char => "char",
            PlyType::UChar => "uchar",
            PlyType::Short => "short",
            PlyType::UShort => "ushort",
            PlyType::Int => "int",
            PlyType::UInt => "uint",
            PlyType::Float => "float",
            PlyType::Double => "double",
        }
    }
}

/// A single property definition (scalar or list).
#[derive(Debug, Clone)]
enum PropertyDef {
    Scalar {
        name: String,
        dtype: PlyType,
    },
    List {
        name: String,
        count_type: PlyType,
        value_type: PlyType,
    },
}

impl PropertyDef {
    fn name(&self) -> &str {
        match self {
            PropertyDef::Scalar { name, .. } | PropertyDef::List { name, .. } => name,
        }
    }
}

/// An element definition from the header.
#[derive(Debug, Clone)]
struct ElementDef {
    name: String,
    count: usize,
    properties: Vec<PropertyDef>,
}

/// A parsed PLY property value.
#[derive(Debug, Clone)]
enum PropertyData {
    Scalar(f64),
    List(Vec<f64>),
}

/// One row of an element.
type ElementRow = Vec<PropertyData>;

/// A parsed element with all its data.
#[derive(Debug, Clone)]
struct Element {
    def: ElementDef,
    data: Vec<ElementRow>,
}

impl Element {
    /// Find the index of a property by canonical name.
    fn property_index(&self, name: &str) -> Option<usize> {
        self.def.properties.iter().position(|p| p.name() == name)
    }

    /// Extract a scalar f64 column by canonical property name.
    fn scalar_column(&self, name: &str) -> Option<Vec<f64>> {
        let idx = self.property_index(name)?;
        let mut col = Vec::with_capacity(self.data.len());
        for row in &self.data {
            match &row[idx] {
                PropertyData::Scalar(v) => col.push(*v),
                PropertyData::List(_) => return None,
            }
        }
        Some(col)
    }

    /// Extract a list column by canonical property name.
    fn list_column(&self, name: &str) -> Option<Vec<Vec<f64>>> {
        let idx = self.property_index(name)?;
        let mut col = Vec::with_capacity(self.data.len());
        for row in &self.data {
            match &row[idx] {
                PropertyData::List(v) => col.push(v.clone()),
                PropertyData::Scalar(_) => return None,
            }
        }
        Some(col)
    }
}

/// The full PLY model in native representation.
pub struct PlyModel {
    _format: PlyFormat,
    comments: Vec<String>,
    elements: Vec<Element>,
}

/// Canonicalize a property name: lowercase, trim, then map known aliases.
fn canonicalize(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "vertex_index" | "vertex_indices" => "vertex_indices".to_string(),
        "texture_u" | "s" => "u".to_string(),
        "texture_v" | "t" => "v".to_string(),
        _ => lower,
    }
}

/// Maximum element count to prevent OOM from malformed headers.
const MAX_ELEMENT_COUNT: usize = 500_000_000;

impl PlyModel {
    /// Parse a PLY file from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // Find end of header — scan for "end_header\n" or "end_header\r\n"
        let header_end = find_header_end(data)
            .ok_or_else(|| anyhow!("Could not find end_header in PLY file"))?;

        let header_bytes = &data[..header_end];
        let header_text = std::str::from_utf8(header_bytes)
            .map_err(|_| anyhow!("PLY header is not valid UTF-8"))?;

        // Validate magic — first non-empty line must be "ply"
        let first_line = header_text
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        if !first_line.trim().eq_ignore_ascii_case("ply") {
            return Err(anyhow!("PLY file does not start with 'ply' magic"));
        }

        let mut format = None;
        let mut comments = Vec::new();
        let mut element_defs: Vec<ElementDef> = Vec::new();

        for line in header_text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.first().map(|s| s.to_ascii_lowercase()).as_deref() {
                Some("format") => {
                    if parts.len() < 3 {
                        return Err(anyhow!("Invalid format line: `{}`", line));
                    }
                    format = Some(match parts[1].to_ascii_lowercase().as_str() {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        "binary_big_endian" => PlyFormat::BinaryBigEndian,
                        _ => return Err(anyhow!("Unknown PLY format: `{}`", parts[1])),
                    });
                }
                Some("comment") => {
                    let text = parts[1..].join(" ");
                    comments.push(text);
                }
                Some("element") => {
                    if parts.len() < 3 {
                        return Err(anyhow!("Invalid element line: `{}`", line));
                    }
                    let name = parts[1].trim().to_ascii_lowercase();
                    let count: usize = parts[2]
                        .parse()
                        .map_err(|_| anyhow!("Invalid element count: `{}`", parts[2]))?;
                    if count > MAX_ELEMENT_COUNT {
                        return Err(anyhow!(
                            "Element `{}` count {} exceeds maximum {}",
                            name,
                            count,
                            MAX_ELEMENT_COUNT
                        ));
                    }
                    element_defs.push(ElementDef {
                        name,
                        count,
                        properties: Vec::new(),
                    });
                }
                Some("property") => {
                    let current = element_defs
                        .last_mut()
                        .ok_or_else(|| anyhow!("Property before any element definition"))?;

                    if parts.len() >= 5 && parts[1].eq_ignore_ascii_case("list") {
                        // property list <count_type> <value_type> <name>
                        let count_type = PlyType::from_str(parts[2])?;
                        let value_type = PlyType::from_str(parts[3])?;
                        let name = canonicalize(parts[4]);
                        current.properties.push(PropertyDef::List {
                            name,
                            count_type,
                            value_type,
                        });
                    } else if parts.len() >= 3 {
                        // property <type> <name>
                        let dtype = PlyType::from_str(parts[1])?;
                        let name = canonicalize(parts[2]);
                        current.properties.push(PropertyDef::Scalar { name, dtype });
                    } else {
                        return Err(anyhow!("Invalid property line: `{}`", line));
                    }
                }
                Some("end_header") => break,
                _ => {
                    // ignore unknown header lines
                }
            }
        }

        let format = format.ok_or_else(|| anyhow!("PLY header missing format line"))?;

        // Parse body
        let body = &data[header_end..];
        let elements = match format {
            PlyFormat::Ascii => parse_ascii_body(body, &element_defs)?,
            PlyFormat::BinaryLittleEndian => parse_binary_body(body, &element_defs, false)?,
            PlyFormat::BinaryBigEndian => parse_binary_body(body, &element_defs, true)?,
        };

        Ok(PlyModel {
            _format: format,
            comments,
            elements,
        })
    }

    /// Extract `TextureFile` path from PLY comments.
    fn texture_file(&self) -> Option<&str> {
        self.comments
            .iter()
            .find_map(|c| c.strip_prefix("TextureFile "))
    }

    /// Convert this PLY model to a Trimesh.
    pub fn to_mesh(&self, resolver: Option<&dyn Resolver>) -> Result<Trimesh> {
        // Find vertex element
        let vertex_elem = self
            .elements
            .iter()
            .find(|e| e.def.name == "vertex")
            .ok_or_else(|| anyhow!("PLY file has no vertex element"))?;

        if vertex_elem.data.is_empty() {
            // Empty mesh
            return Trimesh::new(vec![], vec![], None, None);
        }

        // Extract vertex positions (required)
        let xs = vertex_elem
            .scalar_column("x")
            .ok_or_else(|| anyhow!("PLY vertex element missing 'x' property"))?;
        let ys = vertex_elem
            .scalar_column("y")
            .ok_or_else(|| anyhow!("PLY vertex element missing 'y' property"))?;
        let zs = vertex_elem
            .scalar_column("z")
            .ok_or_else(|| anyhow!("PLY vertex element missing 'z' property"))?;

        let vertices: Vec<Point3<f64>> = xs
            .iter()
            .zip(ys.iter())
            .zip(zs.iter())
            .map(|((&x, &y), &z)| Point3::new(x, y, z))
            .collect();

        // Extract optional vertex normals
        let normals = if let (Some(nx), Some(ny), Some(nz)) = (
            vertex_elem.scalar_column("nx"),
            vertex_elem.scalar_column("ny"),
            vertex_elem.scalar_column("nz"),
        ) {
            Some(
                nx.iter()
                    .zip(ny.iter())
                    .zip(nz.iter())
                    .map(|((&x, &y), &z)| Vector3::new(x, y, z))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Extract optional vertex UVs (canonicalized from s/t/texture_u/texture_v)
        let uvs = if let (Some(u), Some(v)) = (
            vertex_elem.scalar_column("u"),
            vertex_elem.scalar_column("v"),
        ) {
            Some(
                u.iter()
                    .zip(v.iter())
                    .map(|(&u, &v)| Vector2::new(u, v))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Extract optional vertex colors
        let colors = if let (Some(r), Some(g), Some(b)) = (
            vertex_elem.scalar_column("red"),
            vertex_elem.scalar_column("green"),
            vertex_elem.scalar_column("blue"),
        ) {
            let a = vertex_elem.scalar_column("alpha");
            Some(
                r.iter()
                    .zip(g.iter())
                    .zip(b.iter())
                    .enumerate()
                    .map(|(i, ((&r, &g), &b))| {
                        let alpha = a.as_ref().map_or(255.0, |a| a[i]);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        Vector4::new(r as u8, g as u8, b as u8, alpha as u8)
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Extract faces (with optional per-face texcoords)
        let face_elem = self.elements.iter().find(|e| e.def.name == "face");
        let (faces, face_uvs) = if let Some(face_elem) = face_elem {
            if let Some(index_lists) = face_elem.list_column("vertex_indices") {
                let texcoord_lists = face_elem.list_column("texcoord");
                triangulate_faces(&index_lists, texcoord_lists.as_deref(), &vertices)?
            } else {
                (vec![], None)
            }
        } else {
            (vec![], None)
        };

        // Split vertices when per-face UVs are present
        let (vertices, faces, uvs, normals, colors) = if let Some(ref face_uvs) = face_uvs {
            let (v, f, uv, n, c) = split_vertices_by_face_uvs(
                &vertices,
                &faces,
                face_uvs,
                normals.as_deref(),
                colors.as_deref(),
            );
            (v, f, Some(uv), n, c)
        } else {
            (vertices, faces, uvs, normals, colors)
        };

        // Build attributes
        let mut attributes = Attributes::default();
        if let Some(n) = normals {
            attributes.normals.push(n);
        }
        if let Some(uv) = uvs {
            attributes.uv.push(uv);
        }
        if let Some(c) = colors {
            attributes.colors.push(c);
        }

        let has_attrs = !attributes.normals.is_empty()
            || !attributes.uv.is_empty()
            || !attributes.colors.is_empty();

        let mut mesh = Trimesh::new(
            vertices,
            faces,
            if has_attrs { Some(attributes) } else { None },
            None,
        )?;

        // Load texture from comment if resolver available
        if let (Some(tex_file), Some(res)) = (self.texture_file(), resolver)
            && let Ok(bytes) = res.resolve(tex_file)
        {
            mesh.materials.push(Material::Simple(SimpleMaterial {
                diffuse_texture: Some(LazyImage::new(bytes)),
                ..Default::default()
            }));
        }

        // Set load source
        let header = if self.comments.is_empty() {
            None
        } else {
            Some(self.comments.join("\n"))
        };
        mesh.source = LoadSource {
            format: Some(super::FileType::PLY),
            header,
        };

        Ok(mesh)
    }

    /// Build a PlyModel from a Trimesh for export.
    pub fn from_mesh(mesh: &Trimesh) -> Self {
        let num_vertices = mesh.vertices.len();
        let num_faces = mesh.faces.len();

        let has_normals = !mesh.attributes_vertex.normals.is_empty()
            && mesh.attributes_vertex.normals[0].len() == num_vertices;
        let has_uv = !mesh.attributes_vertex.uv.is_empty()
            && mesh.attributes_vertex.uv[0].len() == num_vertices;
        let has_colors = !mesh.attributes_vertex.colors.is_empty()
            && mesh.attributes_vertex.colors[0].len() == num_vertices;

        // Build vertex element definition
        let mut vertex_props = vec![
            PropertyDef::Scalar {
                name: "x".to_string(),
                dtype: PlyType::Float,
            },
            PropertyDef::Scalar {
                name: "y".to_string(),
                dtype: PlyType::Float,
            },
            PropertyDef::Scalar {
                name: "z".to_string(),
                dtype: PlyType::Float,
            },
        ];
        if has_normals {
            vertex_props.push(PropertyDef::Scalar {
                name: "nx".to_string(),
                dtype: PlyType::Float,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "ny".to_string(),
                dtype: PlyType::Float,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "nz".to_string(),
                dtype: PlyType::Float,
            });
        }
        if has_uv {
            vertex_props.push(PropertyDef::Scalar {
                name: "s".to_string(),
                dtype: PlyType::Float,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "t".to_string(),
                dtype: PlyType::Float,
            });
        }
        if has_colors {
            vertex_props.push(PropertyDef::Scalar {
                name: "red".to_string(),
                dtype: PlyType::UChar,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "green".to_string(),
                dtype: PlyType::UChar,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "blue".to_string(),
                dtype: PlyType::UChar,
            });
            vertex_props.push(PropertyDef::Scalar {
                name: "alpha".to_string(),
                dtype: PlyType::UChar,
            });
        }

        let vertex_def = ElementDef {
            name: "vertex".to_string(),
            count: num_vertices,
            properties: vertex_props,
        };

        // Build vertex data
        let mut vertex_data = Vec::with_capacity(num_vertices);
        for i in 0..num_vertices {
            let v = &mesh.vertices[i];
            let mut row: ElementRow = vec![
                PropertyData::Scalar(v.x),
                PropertyData::Scalar(v.y),
                PropertyData::Scalar(v.z),
            ];
            if has_normals {
                let n = &mesh.attributes_vertex.normals[0][i];
                row.push(PropertyData::Scalar(n.x));
                row.push(PropertyData::Scalar(n.y));
                row.push(PropertyData::Scalar(n.z));
            }
            if has_uv {
                let uv = &mesh.attributes_vertex.uv[0][i];
                row.push(PropertyData::Scalar(uv.x));
                row.push(PropertyData::Scalar(uv.y));
            }
            if has_colors {
                let c = &mesh.attributes_vertex.colors[0][i];
                row.push(PropertyData::Scalar(f64::from(c.x)));
                row.push(PropertyData::Scalar(f64::from(c.y)));
                row.push(PropertyData::Scalar(f64::from(c.z)));
                row.push(PropertyData::Scalar(f64::from(c.w)));
            }
            vertex_data.push(row);
        }

        // Build face element
        let face_def = ElementDef {
            name: "face".to_string(),
            count: num_faces,
            properties: vec![PropertyDef::List {
                name: "vertex_indices".to_string(),
                count_type: PlyType::UChar,
                value_type: PlyType::Int,
            }],
        };

        let face_data: Vec<ElementRow> = mesh
            .faces
            .iter()
            .map(|f| {
                vec![PropertyData::List(vec![
                    f[0] as f64,
                    f[1] as f64,
                    f[2] as f64,
                ])]
            })
            .collect();

        PlyModel {
            _format: PlyFormat::Ascii,
            comments: Vec::new(),
            elements: vec![
                Element {
                    def: vertex_def,
                    data: vertex_data,
                },
                Element {
                    def: face_def,
                    data: face_data,
                },
            ],
        }
    }

    /// Export the PLY model as an ASCII string.
    pub fn to_ply_string(&self) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        out.push_str("ply\n");
        out.push_str("format ascii 1.0\n");

        for comment in &self.comments {
            writeln!(out, "comment {}", comment).unwrap();
        }

        for elem in &self.elements {
            writeln!(out, "element {} {}", elem.def.name, elem.def.count).unwrap();
            for prop in &elem.def.properties {
                match prop {
                    PropertyDef::Scalar { name, dtype } => {
                        writeln!(out, "property {} {}", dtype.name(), name).unwrap();
                    }
                    PropertyDef::List {
                        name,
                        count_type,
                        value_type,
                    } => {
                        writeln!(
                            out,
                            "property list {} {} {}",
                            count_type.name(),
                            value_type.name(),
                            name
                        )
                        .unwrap();
                    }
                }
            }
        }

        out.push_str("end_header\n");

        for elem in &self.elements {
            for row in &elem.data {
                let mut first = true;
                for (prop_def, prop_data) in elem.def.properties.iter().zip(row.iter()) {
                    match (prop_def, prop_data) {
                        (PropertyDef::Scalar { dtype, .. }, PropertyData::Scalar(v)) => {
                            if !first {
                                out.push(' ');
                            }
                            dtype.write_ascii(*v, &mut out);
                            first = false;
                        }
                        (
                            PropertyDef::List {
                                count_type,
                                value_type,
                                ..
                            },
                            PropertyData::List(vals),
                        ) => {
                            if !first {
                                out.push(' ');
                            }
                            count_type.write_ascii(vals.len() as f64, &mut out);
                            for v in vals {
                                out.push(' ');
                                value_type.write_ascii(*v, &mut out);
                            }
                            first = false;
                        }
                        _ => {}
                    }
                }
                out.push('\n');
            }
        }

        out
    }
}

/// Export a Trimesh as a PLY ASCII string.
pub fn export_ply(mesh: &Trimesh) -> String {
    PlyModel::from_mesh(mesh).to_ply_string()
}

/// Find the byte offset immediately after the "end_header\n" line.
fn find_header_end(data: &[u8]) -> Option<usize> {
    let needle = b"end_header";
    let i = data.windows(needle.len()).position(|w| w == needle)?;
    let after = i + needle.len();
    // Skip \r\n or \n
    if after < data.len()
        && data[after] == b'\r'
        && after + 1 < data.len()
        && data[after + 1] == b'\n'
    {
        return Some(after + 2);
    }
    if after < data.len() && data[after] == b'\n' {
        return Some(after + 1);
    }
    // end_header at very end of file (no trailing newline)
    Some(after)
}

/// Parse a row of tokens according to the property definitions.
/// Returns the parsed row, consuming only the tokens needed.
fn parse_ascii_row(tokens: &[&str], properties: &[PropertyDef]) -> Result<ElementRow> {
    let mut row = Vec::with_capacity(properties.len());
    let mut pos = 0;
    for prop in properties {
        match prop {
            PropertyDef::Scalar { .. } => {
                if pos >= tokens.len() {
                    return Err(anyhow!("Unexpected end of ASCII PLY row data"));
                }
                let val: f64 = tokens[pos]
                    .parse()
                    .map_err(|_| anyhow!("Failed to parse PLY value: `{}`", tokens[pos]))?;
                row.push(PropertyData::Scalar(val));
                pos += 1;
            }
            PropertyDef::List { .. } => {
                if pos >= tokens.len() {
                    return Err(anyhow!("Unexpected end of ASCII PLY row data"));
                }
                let count: usize = tokens[pos]
                    .parse()
                    .map_err(|_| anyhow!("Failed to parse list count: `{}`", tokens[pos]))?;
                pos += 1;
                let mut vals = Vec::with_capacity(count);
                for _ in 0..count {
                    if pos >= tokens.len() {
                        return Err(anyhow!("Unexpected end of ASCII PLY list data"));
                    }
                    let val: f64 = tokens[pos].parse().map_err(|_| {
                        anyhow!("Failed to parse PLY list value: `{}`", tokens[pos])
                    })?;
                    vals.push(val);
                    pos += 1;
                }
                row.push(PropertyData::List(vals));
            }
        }
    }
    Ok(row)
}

/// Parse ASCII body data line-by-line.
/// Each element row occupies one line; extra trailing tokens on a line are ignored.
fn parse_ascii_body(body: &[u8], element_defs: &[ElementDef]) -> Result<Vec<Element>> {
    let text =
        std::str::from_utf8(body).map_err(|_| anyhow!("PLY ASCII body is not valid UTF-8"))?;

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let mut elements = Vec::with_capacity(element_defs.len());

    for def in element_defs {
        let mut data = Vec::with_capacity(def.count);
        for _ in 0..def.count {
            let line = lines
                .next()
                .ok_or_else(|| anyhow!("Unexpected end of ASCII PLY data"))?;
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let row = parse_ascii_row(&tokens, &def.properties)?;
            data.push(row);
        }
        elements.push(Element {
            def: def.clone(),
            data,
        });
    }

    Ok(elements)
}

/// Parse binary body data (little-endian or big-endian).
fn parse_binary_body(
    body: &[u8],
    element_defs: &[ElementDef],
    big_endian: bool,
) -> Result<Vec<Element>> {
    let mut offset = 0;
    let mut elements = Vec::with_capacity(element_defs.len());

    for def in element_defs {
        let mut data = Vec::with_capacity(def.count);
        for _ in 0..def.count {
            let mut row = Vec::with_capacity(def.properties.len());
            for prop in &def.properties {
                match prop {
                    PropertyDef::Scalar { dtype, .. } => {
                        let (val, size) = dtype.read_binary(body, offset, big_endian)?;
                        row.push(PropertyData::Scalar(val));
                        offset += size;
                    }
                    PropertyDef::List {
                        count_type,
                        value_type,
                        ..
                    } => {
                        let (count_f, size) = count_type.read_binary(body, offset, big_endian)?;
                        offset += size;
                        if count_f < 0.0 || count_f.is_nan() {
                            return Err(anyhow!("Invalid binary list count: {}", count_f));
                        }
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let count = count_f as usize;
                        let max_count = (body.len() - offset) / value_type.size();
                        if count > max_count {
                            return Err(anyhow!(
                                "Binary list count {} exceeds available data ({} items of {} bytes)",
                                count,
                                max_count,
                                value_type.size()
                            ));
                        }
                        let mut vals = Vec::with_capacity(count);
                        for _ in 0..count {
                            let (val, size) = value_type.read_binary(body, offset, big_endian)?;
                            vals.push(val);
                            offset += size;
                        }
                        row.push(PropertyData::List(vals));
                    }
                }
            }
            data.push(row);
        }
        elements.push(Element {
            def: def.clone(),
            data,
        });
    }

    Ok(elements)
}

type TriFaces = (Vec<[usize; 3]>, Option<Vec<[Vector2<f64>; 3]>>);

/// Triangulate face index lists into triangle faces, optionally carrying
/// per-face texcoord lists through the triangulation.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn triangulate_faces(
    index_lists: &[Vec<f64>],
    texcoord_lists: Option<&[Vec<f64>]>,
    vertices: &[Point3<f64>],
) -> Result<TriFaces> {
    let mut triangulator = Triangulator::new();
    let mut faces = Vec::new();
    let has_texcoords = texcoord_lists.is_some();
    let mut face_uvs: Vec<[Vector2<f64>; 3]> = Vec::new();

    for (fi, indices_f64) in index_lists.iter().enumerate() {
        // Validate and convert indices
        let mut indices = Vec::with_capacity(indices_f64.len());
        for (j, &v) in indices_f64.iter().enumerate() {
            if v < 0.0 || v.is_nan() {
                return Err(anyhow!("Face {} vertex {} has invalid index: {}", fi, j, v));
            }
            let vi = v as usize;
            if vi >= vertices.len() {
                return Err(anyhow!(
                    "Face {} vertex {} index {} out of range (vertices.len() = {})",
                    fi,
                    j,
                    vi,
                    vertices.len()
                ));
            }
            indices.push(vi);
        }
        let n = indices.len();

        // Parse per-face texcoord pairs if available (zero-pad if too short)
        let tc: Vec<Vector2<f64>> = if let Some(tcs) = texcoord_lists {
            let tc_list = &tcs[fi];
            (0..n)
                .map(|j| {
                    let u = tc_list.get(j * 2).copied().unwrap_or(0.0);
                    let v = tc_list.get(j * 2 + 1).copied().unwrap_or(0.0);
                    Vector2::new(u, v)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Compute local-index triangles for the polygon
        let local_tris: Vec<[usize; 3]> = match n {
            0..=2 => Vec::new(),
            3 => vec![[0, 1, 2]],
            4 => vec![[0, 1, 2], [0, 2, 3]],
            _ => triangulator.triangulate_3d(&indices, &[], vertices, true, true)?,
        };

        for tri in &local_tris {
            faces.push([indices[tri[0]], indices[tri[1]], indices[tri[2]]]);
            if has_texcoords {
                face_uvs.push([tc[tri[0]], tc[tri[1]], tc[tri[2]]]);
            }
        }
    }

    let result_uvs = if has_texcoords && !face_uvs.is_empty() {
        Some(face_uvs)
    } else {
        None
    };

    Ok((faces, result_uvs))
}

type SplitResult = (
    Vec<Point3<f64>>,
    Vec<[usize; 3]>,
    Vec<Vector2<f64>>,
    Option<Vec<Vector3<f64>>>,
    Option<Vec<Vector4<u8>>>,
);

/// Split vertices so that each unique (vertex, UV) pair gets its own index.
/// This converts per-face UVs into per-vertex UVs suitable for indexed rendering.
fn split_vertices_by_face_uvs(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    face_uvs: &[[Vector2<f64>; 3]],
    normals: Option<&[Vector3<f64>]>,
    colors: Option<&[Vector4<u8>]>,
) -> SplitResult {
    use std::collections::HashMap;

    /// Canonicalize f64 bits so ±0.0 and NaN variants hash identically.
    fn canonical_bits(v: f64) -> u64 {
        if v == 0.0 {
            0
        } else if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        }
    }

    let mut map: HashMap<(usize, u64, u64), usize> = HashMap::new();
    let mut new_verts = Vec::new();
    let mut new_uvs = Vec::new();
    let mut new_normals: Option<Vec<Vector3<f64>>> = normals.map(|_| Vec::new());
    let mut new_colors: Option<Vec<Vector4<u8>>> = colors.map(|_| Vec::new());
    let mut new_faces = Vec::with_capacity(faces.len());

    for (fi, face) in faces.iter().enumerate() {
        let mut new_face = [0usize; 3];
        for corner in 0..3 {
            let vi = face[corner];
            let uv = face_uvs[fi][corner];
            let key = (vi, canonical_bits(uv.x), canonical_bits(uv.y));
            let new_vi = if let Some(&idx) = map.get(&key) {
                idx
            } else {
                let idx = new_verts.len();
                new_verts.push(vertices[vi]);
                new_uvs.push(uv);
                if let Some(ref mut nn) = new_normals {
                    nn.push(normals.unwrap()[vi]);
                }
                if let Some(ref mut nc) = new_colors {
                    nc.push(colors.unwrap()[vi]);
                }
                map.insert(key, idx);
                idx
            };
            new_face[corner] = new_vi;
        }
        new_faces.push(new_face);
    }

    (new_verts, new_faces, new_uvs, new_normals, new_colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exchange::{FileType, load};
    use crate::geometry::Geometry;
    use crate::resolvers::FileResolver;

    /// Path to trimesh test models.
    fn trimesh_model(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("HOME"))
            .join("trimesh/models")
            .join(name)
    }

    /// Load a PLY file from trimesh models directory.
    fn load_ply(name: &str) -> PlyModel {
        let path = trimesh_model(name);
        let data = std::fs::read(&path)
            .unwrap_or_else(|_| panic!("Could not read test file: {}", path.display()));
        PlyModel::from_bytes(&data).unwrap()
    }

    #[test]
    fn test_ply_ascii_with_colors() {
        let path = trimesh_model("fuze_ascii.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("fuze_ascii.ply");
        assert_eq!(ply._format, PlyFormat::Ascii);

        let vertex = ply
            .elements
            .iter()
            .find(|e| e.def.name == "vertex")
            .unwrap();
        assert_eq!(vertex.data.len(), 502);
        assert!(vertex.property_index("red").is_some());

        let face = ply.elements.iter().find(|e| e.def.name == "face").unwrap();
        assert_eq!(face.data.len(), 1000);

        let mesh = ply.to_mesh(None).unwrap();
        // Vertices are split by per-face UVs, so count exceeds original 502
        assert!(mesh.vertices.len() > 502);
        assert_eq!(mesh.faces.len(), 1000);
        // Colors and UVs are carried through the split
        assert_eq!(mesh.attributes_vertex.colors.len(), 1);
        assert_eq!(mesh.attributes_vertex.colors[0].len(), mesh.vertices.len());
        assert_eq!(mesh.attributes_vertex.uv.len(), 1);
        assert_eq!(mesh.attributes_vertex.uv[0].len(), mesh.vertices.len());
        assert_eq!(mesh.source.format, Some(FileType::PLY));
    }

    #[test]
    fn test_ply_ascii_with_normals_uvs() {
        let path = trimesh_model("plane.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("plane.ply");
        assert_eq!(ply._format, PlyFormat::Ascii);

        let vertex = ply
            .elements
            .iter()
            .find(|e| e.def.name == "vertex")
            .unwrap();
        assert_eq!(vertex.data.len(), 4);
        // "s" and "t" should be canonicalized to "u" and "v"
        assert!(vertex.property_index("u").is_some());
        assert!(vertex.property_index("v").is_some());
        assert!(vertex.property_index("nx").is_some());

        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        // Quad face should be triangulated into 2 triangles
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.attributes_vertex.normals.len(), 1);
        assert_eq!(mesh.attributes_vertex.uv.len(), 1);
    }

    #[test]
    fn test_ply_binary_le() {
        let path = trimesh_model("tet.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("tet.ply");
        assert_eq!(ply._format, PlyFormat::BinaryLittleEndian);

        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 4);
    }

    #[test]
    fn test_ply_point_cloud() {
        let path = trimesh_model("points_ascii.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("points_ascii.ply");
        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 5);
        assert_eq!(mesh.faces.len(), 0);
    }

    #[test]
    fn test_ply_empty() {
        let path = trimesh_model("empty.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("empty.ply");
        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 0);
        assert_eq!(mesh.faces.len(), 0);
    }

    #[test]
    fn test_ply_metadata_extra_property() {
        // metadata.ply has an extra scalar "face_type" on the face element
        let path = trimesh_model("metadata.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("metadata.ply");
        let face = ply.elements.iter().find(|e| e.def.name == "face").unwrap();
        // Should have vertex_indices list AND face_type scalar
        assert!(face.property_index("vertex_indices").is_some());
        assert!(face.property_index("face_type").is_some());

        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);
    }

    #[test]
    fn test_ply_quad_triangulation() {
        let path = trimesh_model("cube_blender_uv.ply");
        if !path.exists() {
            return;
        }

        let ply = load_ply("cube_blender_uv.ply");
        let mesh = ply.to_mesh(None).unwrap();
        assert_eq!(mesh.vertices.len(), 24);
        // 6 quad faces → 12 triangles
        assert_eq!(mesh.faces.len(), 12);
        assert_eq!(mesh.attributes_vertex.normals.len(), 1);
        assert_eq!(mesh.attributes_vertex.uv.len(), 1);
    }

    #[test]
    fn test_ply_round_trip() {
        let path = trimesh_model("tet.ply");
        if !path.exists() {
            return;
        }

        // Load binary PLY → mesh → PlyModel → ASCII string → parse again
        let ply1 = load_ply("tet.ply");
        let mesh1 = ply1.to_mesh(None).unwrap();

        let ply2 = PlyModel::from_mesh(&mesh1);
        let ascii = ply2.to_ply_string();
        let ply3 = PlyModel::from_bytes(ascii.as_bytes()).unwrap();
        let mesh2 = ply3.to_mesh(None).unwrap();

        assert_eq!(mesh1.vertices.len(), mesh2.vertices.len());
        assert_eq!(mesh1.faces.len(), mesh2.faces.len());

        // Verify vertex positions match
        for (a, b) in mesh1.vertices.iter().zip(mesh2.vertices.iter()) {
            assert!((a.x - b.x).abs() < 1e-6);
            assert!((a.y - b.y).abs() < 1e-6);
            assert!((a.z - b.z).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ply_load_integration() {
        // Test via the exchange::load() entry point
        let path = trimesh_model("tet.ply");
        if !path.exists() {
            return;
        }

        let data = std::fs::read(&path).unwrap();
        let scene = load(&data, Some(FileType::PLY), None).unwrap();
        assert_eq!(scene.geometry.len(), 1);
        if let Geometry::Mesh(mesh) = scene.geometry.values().next().unwrap() {
            assert_eq!(mesh.vertices.len(), 4);
            assert_eq!(mesh.faces.len(), 4);
        } else {
            panic!("Expected Mesh geometry");
        }
    }

    #[test]
    fn test_ply_magic_detection() {
        // PLY magic bytes detection
        assert_eq!(
            FileType::from_bytes(b"ply\nformat ascii 1.0\n"),
            Some(FileType::PLY)
        );
        assert_eq!(
            FileType::from_bytes(b"ply\r\nformat binary_little_endian 1.0\r\n"),
            Some(FileType::PLY)
        );
        assert_eq!(
            FileType::from_bytes(b"PLY\nformat ascii 1.0\n"),
            Some(FileType::PLY)
        );
    }

    #[test]
    fn test_ply_fuze_round_trip() {
        let path = trimesh_model("fuze_ascii.ply");
        if !path.exists() {
            return;
        }

        let resolver =
            FileResolver::new(std::path::PathBuf::from(env!("HOME")).join("trimesh/models"));
        let data = std::fs::read(&path).unwrap();
        let ply = PlyModel::from_bytes(&data).unwrap();
        let mesh = ply.to_mesh(Some(&resolver)).unwrap();

        // Vertices should be split (more than original 502)
        assert!(mesh.vertices.len() > 502);
        assert_eq!(mesh.faces.len(), 1000);

        // UVs present and nonzero
        assert_eq!(mesh.attributes_vertex.uv.len(), 1);
        assert_eq!(mesh.attributes_vertex.uv[0].len(), mesh.vertices.len());
        assert!(
            mesh.attributes_vertex.uv[0]
                .iter()
                .any(|uv| uv.x != 0.0 || uv.y != 0.0)
        );

        // Texture loaded
        assert!(!mesh.materials.is_empty());
        if let Material::Simple(ref mat) = mesh.materials[0] {
            assert!(mat.diffuse_texture.is_some());
            assert!(mat.diffuse_texture.as_ref().unwrap().bytes_len() > 0);
        } else {
            panic!("Expected Simple material");
        }

        // Round-trip: export → reload
        let ply2 = PlyModel::from_mesh(&mesh);
        let ascii = ply2.to_ply_string();
        let ply3 = PlyModel::from_bytes(ascii.as_bytes()).unwrap();
        let mesh2 = ply3.to_mesh(None).unwrap();

        assert_eq!(mesh.vertices.len(), mesh2.vertices.len());
        assert_eq!(mesh.faces.len(), mesh2.faces.len());
        assert_eq!(
            mesh.attributes_vertex.uv.len(),
            mesh2.attributes_vertex.uv.len()
        );
        assert_eq!(
            mesh.attributes_vertex.uv[0].len(),
            mesh2.attributes_vertex.uv[0].len()
        );
    }
}
