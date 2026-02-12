pub(crate) mod cdt;
pub mod faces;
pub(crate) mod hex_grid;
pub mod step;
pub mod tesselate;
pub mod topology;

pub use faces::{Surface, SurfaceCurvature, SurfaceDict};
pub use step::{StepError, from_step};
pub use tesselate::TesselationParams;
pub use topology::{
    BrepEdge, BrepError, BrepFace, BrepLoop, BrepModel, BrepShell, BrepSolid, BrepVertex, Curve,
    CurveBSpline, CurveCircle, CurveEllipse, CurveLine, EdgeUse, OrientedEdge,
};
