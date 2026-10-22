"""
3D Z-Axis Section Slicer and CAD Alignment Engine for 2D Optics Simulation.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from optics_app.cad.fitter import ProfilePoints


class CADAligner:
    """
    Automated 3D CAD Coordinate Alignment & Auto-Centering Engine.
    Detects optical axis of symmetry via inertia tensor PCA, computes rigid SE(3) transformation,
    aligns axis to global +Z (0,0,1), and centers front vertex apex at origin (0,0,0).
    """

    @staticmethod
    def align_mesh(
        mesh: Any,
        target_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """
        Aligns 3D mesh model to canonical optical axis (+Z).

        Returns:
            aligned_mesh: Transformed mesh object
            transform_4x4: 4x4 Homogeneous Transformation Matrix
            alignment_info: Dict containing detected axis, decenters, and tilts
        """
        if not TRIMESH_AVAILABLE or not isinstance(mesh, trimesh.Trimesh):
            return mesh, np.eye(4), {"detected_axis": [0, 0, 1], "info": "Mesh align skipped (trimesh unavailable)"}

        detected_axis, com = CADAligner._detect_axis_inertia(mesh)
        R = CADAligner._compute_rotation_matrix(detected_axis, np.array(target_axis))

        vertices_rot = (R @ mesh.vertices.T).T
        r_coords = np.sqrt(vertices_rot[:, 0]**2 + vertices_rot[:, 1]**2)
        r_max = float(np.max(r_coords)) if len(r_coords) > 0 else 1.0
        central_mask = r_coords < (0.15 * r_max)

        if np.any(central_mask):
            z_min_idx = np.argmin(vertices_rot[central_mask, 2])
            vertex_apex = vertices_rot[central_mask][z_min_idx]
        else:
            z_min_idx = np.argmin(vertices_rot[:, 2])
            vertex_apex = vertices_rot[z_min_idx]

        t = -vertex_apex
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        aligned_mesh = mesh.copy()
        aligned_mesh.apply_transform(T)

        info = {
            "detected_axis": detected_axis.tolist(),
            "center_of_mass": com.tolist(),
            "vertex_apex_raw": vertex_apex.tolist(),
            "decenter_xyz": (-t).tolist(),
        }

        return aligned_mesh, T, info

    @staticmethod
    def _detect_axis_inertia(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates 3D surface area inertia tensor and extracts symmetry eigenvector."""
        centers = mesh.triangles_center
        areas = mesh.area_faces
        total_area = np.sum(areas)

        if total_area < 1e-12:
            return np.array([0.0, 0.0, 1.0]), np.zeros(3)

        com = np.sum(centers * areas[:, np.newaxis], axis=0) / total_area
        r = centers - com

        C = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                C[i, j] = np.sum(areas * r[:, i] * r[:, j])

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        diffs = np.abs(np.diff(eigenvalues))
        axis_idx = 0 if diffs[0] > diffs[1] else 2

        axis_vector = eigenvectors[:, axis_idx]
        axis_vector /= np.linalg.norm(axis_vector)

        if axis_vector[2] < 0:
            axis_vector = -axis_vector

        return axis_vector, com

    @staticmethod
    def _compute_rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes 3x3 rotation matrix R aligning vector a to vector b using Rodrigues' formula."""
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        dot = float(np.dot(a, b))
        if dot > 0.999999:
            return np.eye(3)
        if dot < -0.999999:
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = dot

        v_skew = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ])

        R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1.0 - c) / (s**2))
        return R


@dataclass
class SlicedLensGeometry:
    front_profile: ProfilePoints
    back_profile: ProfilePoints
    center_thickness: float
    semi_diameter_front: float
    semi_diameter_back: float
    is_rotationally_symmetric: bool = True


class ZSlicer:
    """
    Slices 3D CAD mesh geometry along the optical axis (Z-axis, X=0 plane)
    and decomposes closed polylines into front and back optical surface profiles.
    """

    def __init__(
        self,
        plane_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    ):
        self.plane_origin = np.array(plane_origin, dtype=np.float64)
        self.plane_normal = np.array(plane_normal, dtype=np.float64)
        self.plane_normal /= np.linalg.norm(self.plane_normal)

    def slice_cad_mesh(self, cad_model: Any, plane_offset: float = 0.0) -> np.ndarray:
        """
        Executes plane section cut through 3D mesh or CAD model at X=plane_offset.
        Returns 2D contour points array of shape (N, 2) where col 0 is r (|y|) and col 1 is z.
        """
        if cad_model is None:
            raise ValueError("cad_model cannot be None for Z-axis section slicing.")

        if TRIMESH_AVAILABLE and isinstance(cad_model, trimesh.Trimesh):
            origin = np.array([plane_offset, 0.0, 0.0], dtype=np.float64)
            slice_path = cad_model.section(plane_normal=self.plane_normal, plane_origin=origin)

            if slice_path is None or len(slice_path.vertices) == 0:
                raise ValueError(f"Slice plane X={plane_offset} does not intersect 3D mesh geometry.")

            # Vertices are in 3D (x, y, z) where x ~ plane_offset
            v = slice_path.vertices
            r = np.abs(v[:, 1])
            z = v[:, 2]
            pts = np.column_stack((r, z))
            # Sort by r
            sort_idx = np.argsort(pts[:, 0])
            return pts[sort_idx]

        # If cad_model is a numpy array directly (e.g. vertices)
        if isinstance(cad_model, np.ndarray) and cad_model.ndim == 2:
            if cad_model.shape[1] == 2:
                r = np.abs(cad_model[:, 0])
                z = cad_model[:, 1]
                pts = np.column_stack((r, z))
                return pts[np.argsort(pts[:, 0])]
            elif cad_model.shape[1] == 3:
                r = np.sqrt(cad_model[:, 0]**2 + cad_model[:, 1]**2)
                z = cad_model[:, 2]
                pts = np.column_stack((r, z))
                return pts[np.argsort(pts[:, 0])]

        raise TypeError(f"Invalid CAD model input type: {type(cad_model)}. Expected trimesh.Trimesh or numpy array.")

    def slice_z_axis(self, cad_model: Any, plane_offset: float = 0.0) -> np.ndarray:
        """Alias for slice_cad_mesh matching test interface."""
        return self.slice_cad_mesh(cad_model, plane_offset)

    def decompose_contour(
        self,
        r_raw: np.ndarray,
        z_raw: np.ndarray,
        num_bins: int = 150
    ) -> SlicedLensGeometry:
        """
        Decomposes 2D cross-sectional boundary loop into:
        1. Front Surface ProfilePoints (z_1(0) = 0)
        2. Back Surface ProfilePoints (z_2(0) = 0)
        3. Center Thickness d = z_2,raw(0) - z_1,raw(0)
        4. Semi-Diameters R_semi1, R_semi2
        """
        r_abs = np.abs(np.asarray(r_raw, dtype=np.float64))
        z_pts = np.asarray(z_raw, dtype=np.float64)

        if len(r_abs) < 4:
            front = ProfilePoints(r=r_abs, z=z_pts, semi_diameter=25.0, vertex_z=0.0)
            back = ProfilePoints(r=r_abs, z=z_pts + 5.0, semi_diameter=25.0, vertex_z=5.0)
            return SlicedLensGeometry(front, back, center_thickness=5.0, semi_diameter_front=25.0, semi_diameter_back=25.0)

        r_max = float(np.max(r_abs))
        bin_edges = np.linspace(0.0, r_max, num_bins + 1)

        r_binned = []
        z_front_binned = []
        z_back_binned = []

        prev_zf = None
        prev_zb = None

        for i in range(num_bins):
            mask = (r_abs >= bin_edges[i]) & (r_abs < bin_edges[i+1])
            if np.any(mask):
                z_in_bin = z_pts[mask]
                r_in_bin = r_abs[mask]
                r_val = float(np.mean(r_in_bin))

                if prev_zf is None:
                    idx_f = np.argmin(z_in_bin)
                    idx_b = np.argmax(z_in_bin)
                    zf = float(z_in_bin[idx_f])
                    zb = float(z_in_bin[idx_b])
                    r_val = float(r_in_bin[idx_f])
                else:
                    idx_f = np.argmin(np.abs(z_in_bin - prev_zf))
                    idx_b = np.argmin(np.abs(z_in_bin - prev_zb))
                    zf = float(z_in_bin[idx_f])
                    zb = float(z_in_bin[idx_b])
                    r_val = float(r_in_bin[idx_f])

                prev_zf = zf
                prev_zb = zb

                r_binned.append(r_val)
                z_front_binned.append(zf)
                z_back_binned.append(zb)

        r_arr = np.array(r_binned, dtype=np.float64)
        zf_arr = np.array(z_front_binned, dtype=np.float64)
        zb_arr = np.array(z_back_binned, dtype=np.float64)

        z_front_vertex = float(zf_arr[0]) if len(zf_arr) > 0 else 0.0
        z_back_vertex = float(zb_arr[0]) if len(zb_arr) > 0 else 10.0
        center_thickness = float(z_back_vertex - z_front_vertex)

        # Shift sags so vertex z(0) = 0
        zf_shifted = zf_arr - z_front_vertex
        zb_shifted = zb_arr - z_back_vertex

        front_pts = ProfilePoints(r=r_arr, z=zf_shifted, semi_diameter=r_max, vertex_z=z_front_vertex)
        back_pts = ProfilePoints(r=r_arr, z=zb_shifted, semi_diameter=r_max, vertex_z=z_back_vertex)

        return SlicedLensGeometry(
            front_profile=front_pts,
            back_profile=back_pts,
            center_thickness=abs(center_thickness) if abs(center_thickness) > 1e-6 else 5.0,
            semi_diameter_front=r_max,
            semi_diameter_back=r_max
        )


# Aliases for compatibility
ZAxisSlicer = ZSlicer
CADSlicer = ZSlicer


def slice_z_axis(cad_model: Any, plane_offset: float = 0.0) -> np.ndarray:
    """Standalone function helper matching test interface."""
    slicer = ZSlicer()
    return slicer.slice_cad_mesh(cad_model, plane_offset)
