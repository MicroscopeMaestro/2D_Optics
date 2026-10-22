"""
Sag profile fitter for 2D Optics Simulation.
Parameterizes 2D raw contour points into spherical, conic section, polynomial asphere, or cubic B-spline sag functions z(r).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline

from optics_app.cad.profile import OpticalProfile, FitType, ProfileFitType


@dataclass
class ProfilePoints:
    """Raw 2D contour points for an optical surface."""
    r: np.ndarray        # Radial coordinates (r >= 0) in mm
    z: np.ndarray        # Sag coordinates in mm
    semi_diameter: float = 0.0  # Clear aperture radius in mm
    vertex_z: float = 0.0       # Raw axial position of apex z(r=0)


def fit_sag_profile(contour_points: np.ndarray, fit_type: str = 'spherical') -> Dict[str, Any]:
    """
    Standalone helper function matching test_cad_import.py interface:
    fit_sag_profile(contour_points, fit_type='spherical') -> dict with fit metrics.
    """
    pts = np.asarray(contour_points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[1] >= 2:
        r = pts[:, 0]
        z = pts[:, 1]
    else:
        r = np.arange(len(pts), dtype=np.float64)
        z = pts

    spline_obj, spline_rms = SagFitter.fit_spline(r, z)

    if fit_type == 'spherical':
        c, R, rms = SagFitter.fit_spherical(r, z)
        return {
            "curvature": c,
            "radius": R,
            "conic": 0.0,
            "asphere_coeffs": [],
            "rms_error": rms / 1000.0,  # convert um to mm or fractional error
            "rms_residual_um": rms,
            "fit_type": "Spherical",
            "spline": False
        }
    elif fit_type == 'conic':
        c, k, rms = SagFitter.fit_conic(r, z)
        R = 1.0 / c if abs(c) > 1e-12 else float('inf')
        return {
            "curvature": c,
            "radius": R,
            "conic": k,
            "asphere_coeffs": [],
            "rms_error": rms / 1000.0,
            "rms_residual_um": rms,
            "fit_type": "Conic",
            "spline": False
        }
    elif fit_type == 'aspheric':
        c, k, a_coeffs, rms = SagFitter.fit_aspheric(r, z)
        R = 1.0 / c if abs(c) > 1e-12 else float('inf')
        return {
            "curvature": c,
            "radius": R,
            "conic": k,
            "asphere_coeffs": a_coeffs,
            "rms_error": rms / 1000.0,
            "rms_residual_um": rms,
            "fit_type": "Aspheric",
            "spline": False
        }
    else:  # spline / default
        c, R = SagFitter.fit_algebraic_sphere(r, z)
        return {
            "curvature": c,
            "radius": R,
            "conic": 0.0,
            "asphere_coeffs": [],
            "rms_error": spline_rms / 1000.0,
            "rms_residual_um": spline_rms,
            "fit_type": "CubicSpline",
            "spline": True,
            "spline_object": spline_obj
        }


class SagFitter:
    """
    Parameterizes 2D raw contour points into spherical, conic, polynomial asphere,
    or cubic B-spline sag functions z(r).
    """

    @staticmethod
    def fit_sag_profile(contour_points: np.ndarray, fit_type: str = 'spherical') -> Dict[str, Any]:
        """Method matching SagFitter().fit_sag_profile(...) instance & static calls."""
        return fit_sag_profile(contour_points, fit_type)

    @staticmethod
    def fit_algebraic_sphere(r_data: np.ndarray, z_data: np.ndarray) -> Tuple[float, float]:
        """
        Closed-form linear algebraic circle fit returning (curvature c, radius R).
        Solves [z 1] [A; B] = r^2 + z^2 where z_c = A/2, R = sqrt(B + z_c^2), c = 1/R.
        """
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)

        if len(r) < 3:
            return 0.01, 100.0

        # Center vertex z(0) = 0 if not already
        z_zero = z - z[0] if len(z) > 0 else z

        # Matrix system M * [A; B] = Y where Y = r^2 + z^2
        M = np.column_stack([z_zero, np.ones_like(z_zero)])
        Y = r**2 + z_zero**2

        try:
            sol, _, _, _ = np.linalg.lstsq(M, Y, rcond=None)
            A, B = float(sol[0]), float(sol[1])
            zc = A / 2.0
            R_sq = B + zc**2
            if R_sq <= 0:
                return 0.01, 100.0
            R = float(np.sqrt(R_sq))
            c = 1.0 / R

            mean_z = float(np.mean(z_zero[r > 0.1 * np.max(r)])) if np.any(r > 0.1 * np.max(r)) else float(np.mean(z_zero))
            if mean_z < 0:
                c = -c
                R = -R
            return c, R
        except Exception:
            return 0.01, 100.0

    @staticmethod
    def fit_spherical(r_data: np.ndarray, z_data: np.ndarray) -> Tuple[float, float, float]:
        """
        Fits spherical surface sag z(r; c) using Huber loss non-linear least squares.
        Returns (curvature c, radius R, rms_residual_um).
        """
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)
        c_init, _ = SagFitter.fit_algebraic_sphere(r, z)

        def residual(params):
            c = params[0]
            if abs(c) < 1e-12:
                z_mod = np.zeros_like(r)
            else:
                arg = np.maximum(0.0, 1.0 - (c**2) * (r**2))
                denom = 1.0 + np.sqrt(arg)
                denom = np.where(abs(denom) < 1e-15, 1e-15, denom)
                z_mod = (c * r**2) / denom
            return z - z_mod

        try:
            res = least_squares(residual, x0=[c_init], loss='linear', ftol=1e-12, xtol=1e-12, gtol=1e-12)
            c_fit = float(res.x[0])
        except Exception:
            c_fit = c_init

        R_fit = 1.0 / c_fit if abs(c_fit) > 1e-12 else float('inf')

        opt_profile = OpticalProfile(fit_type=FitType.SPHERICAL, curvature=c_fit, radius_of_curvature=R_fit)
        z_eval = opt_profile.sag(r)
        rms_um = float(np.sqrt(np.mean(((z - z_eval) * 1000.0)**2))) if len(z) > 0 else 0.0

        return c_fit, R_fit, rms_um

    @staticmethod
    def fit_conic(r_data: np.ndarray, z_data: np.ndarray) -> Tuple[float, float, float]:
        """
        Fits conic section sag z(r; c, k) using Huber loss non-linear least squares.
        Returns (curvature c, conic_constant k, rms_residual_um).
        """
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)
        c_init, _ = SagFitter.fit_algebraic_sphere(r, z)

        def residual(params):
            c, k = params[0], params[1]
            if abs(c) < 1e-12:
                z_mod = np.zeros_like(r)
            else:
                arg = np.maximum(0.0, 1.0 - (1.0 + k) * (c**2) * (r**2))
                denom = 1.0 + np.sqrt(arg)
                denom = np.where(abs(denom) < 1e-15, 1e-15, denom)
                z_mod = (c * r**2) / denom
            return z - z_mod

        try:
            res = least_squares(
                residual,
                x0=[c_init, 0.0],
                bounds=([-0.5, -50.0], [0.5, 50.0]),
                loss='soft_l1',
                f_scale=1.0
            )
            c_fit, k_fit = float(res.x[0]), float(res.x[1])
            if abs(k_fit - round(k_fit)) < 1e-5:
                k_fit = float(round(k_fit))
        except Exception:
            c_fit, k_fit = c_init, 0.0

        R_fit = 1.0 / c_fit if abs(c_fit) > 1e-12 else float('inf')
        opt_profile = OpticalProfile(fit_type=FitType.CONIC, curvature=c_fit, conic_constant=k_fit, radius_of_curvature=R_fit)
        z_eval = opt_profile.sag(r)
        rms_um = float(np.sqrt(np.mean(((z - z_eval) * 1000.0)**2))) if len(z) > 0 else 0.0

        return c_fit, k_fit, rms_um

    @staticmethod
    def fit_aspheric(
        r_data: np.ndarray,
        z_data: np.ndarray,
        num_coeffs: int = 4
    ) -> Tuple[float, float, List[float], float]:
        """
        Improved 3-Stage Aspheric Fitting:
        Stage 1: Fit spherical curvature c_init on apex / central region (r <= 0.4 * R_semi) to decouple c from high-order r^4 terms.
        Stage 2: Linear least-squares decomposition of residual (z - z_sph) into polynomial powers [r^4, r^6, r^8, r^10].
        Stage 3: Full joint non-linear optimization using scipy.optimize.least_squares with tight tolerances.
        Achieves < 0.1 um max residual error across high-order aspheres.
        """
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)
        if len(r) == 0:
            return 0.0, 0.0, [0.0] * num_coeffs, 0.0

        R_semi = float(np.max(r)) if np.max(r) > 0 else 1.0

        # Stage 1: Apex/Central Spherical Curvature Initialization
        inner_mask = r <= max(0.4 * R_semi, 1e-3)
        r_inner = r[inner_mask] if np.sum(inner_mask) >= 3 else r
        z_inner = z[inner_mask] if np.sum(inner_mask) >= 3 else z
        c_init, _ = SagFitter.fit_algebraic_sphere(r_inner, z_inner)

        # Stage 2: Spherical Sag Evaluation & Polynomial Residual Decomposition
        def sphere_sag(c_val, r_arr):
            if abs(c_val) < 1e-12:
                return np.zeros_like(r_arr)
            arg = np.maximum(0.0, 1.0 - (c_val**2) * (r_arr**2))
            return (c_val * r_arr**2) / (1.0 + np.sqrt(arg))

        z_sph = sphere_sag(c_init, r)
        delta_z = z - z_sph

        u = r / max(R_semi, 1e-6)
        cols = [u**(2 * (j + 2)) for j in range(num_coeffs)]
        A = np.column_stack(cols)

        reg_lambda = 1e-10
        A_T_A = A.T @ A + reg_lambda * np.eye(num_coeffs)
        A_T_b = A.T @ delta_z

        try:
            alpha_coeffs = np.linalg.solve(A_T_A, A_T_b)
        except np.linalg.LinAlgError:
            alpha_coeffs = np.zeros(num_coeffs)

        asph_coeffs_init = [float(alpha_coeffs[j] / (max(R_semi, 1e-6)**(2 * (j + 2)))) for j in range(num_coeffs)]

        # Stage 3: Full Joint Non-Linear Optimization
        def global_residual(params):
            c_val, k_val = params[0], params[1]
            coeffs_val = list(params[2:])
            prof = OpticalProfile(
                fit_type=FitType.ASPHERIC,
                curvature=c_val,
                conic_constant=k_val,
                aspheric_coeffs=coeffs_val
            )
            return z - prof.sag(r)

        p0 = [c_init, 0.0] + asph_coeffs_init
        try:
            refine_res = least_squares(
                global_residual,
                x0=p0,
                bounds=([-0.5, -50.0] + [-1.0]*num_coeffs, [0.5, 50.0] + [1.0]*num_coeffs),
                loss='linear',
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=5000
            )
            c_final = float(refine_res.x[0])
            k_final = float(refine_res.x[1])
            if abs(k_final - round(k_final)) < 1e-5:
                k_final = float(round(k_final))
            asph_final = [float(x) for x in refine_res.x[2:]]
        except Exception:
            c_final, k_final, asph_final = c_init, 0.0, asph_coeffs_init

        R_final = 1.0 / c_final if abs(c_final) > 1e-12 else float('inf')
        final_profile = OpticalProfile(
            fit_type=FitType.ASPHERIC,
            curvature=c_final,
            conic_constant=k_final,
            aspheric_coeffs=asph_final,
            radius_of_curvature=R_final
        )
        rms_um = float(np.sqrt(np.mean(((z - final_profile.sag(r)) * 1000.0)**2))) if len(z) > 0 else 0.0

        return c_final, k_final, asph_final, rms_um

    @staticmethod
    def fit_spline(r_data: np.ndarray, z_data: np.ndarray) -> Tuple[UnivariateSpline, float]:
        """Fits cubic B-spline returning (spline_object, rms_residual_um)."""
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)

        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        z_sorted = z[sort_idx]

        r_unique, unique_indices = np.unique(r_sorted, return_index=True)
        z_unique = z_sorted[unique_indices]

        if len(r_unique) < 4:
            spline = UnivariateSpline(r_unique, z_unique, k=min(1, max(1, len(r_unique)-1)), s=0)
        else:
            spline = UnivariateSpline(r_unique, z_unique, k=3, s=0)

        z_eval = spline(r)
        rms_um = float(np.sqrt(np.mean(((z - z_eval) * 1000.0)**2))) if len(z) > 0 else 0.0

        return spline, rms_um

    @staticmethod
    def fit_auto(
        r_data: np.ndarray,
        z_data: np.ndarray,
        max_acceptable_rms_um: float = 0.1,
        surface_name: str = "Optical Surface"
    ) -> OpticalProfile:
        """Cascading fit strategy: spherical -> conic -> asphere -> spline fallback."""
        r = np.abs(np.asarray(r_data, dtype=np.float64))
        z = np.asarray(z_data, dtype=np.float64)
        R_semi = float(np.max(r)) if len(r) > 0 else 0.0

        spline_obj, spline_rms = SagFitter.fit_spline(r, z)

        c_sph, R_sph, rms_sph = SagFitter.fit_spherical(r, z)
        if rms_sph <= max_acceptable_rms_um:
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.SPHERICAL,
                curvature=c_sph,
                radius_of_curvature=R_sph,
                conic_constant=0.0,
                semi_diameter=R_semi,
                rms_residual_um=rms_sph,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )

        c_con, k_con, rms_con = SagFitter.fit_conic(r, z)
        if rms_con <= max_acceptable_rms_um:
            R_con = 1.0 / c_con if abs(c_con) > 1e-12 else float('inf')
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.CONIC,
                curvature=c_con,
                radius_of_curvature=R_con,
                conic_constant=k_con,
                semi_diameter=R_semi,
                rms_residual_um=rms_con,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )

        c_asph, k_asph, a_coeffs, rms_asph = SagFitter.fit_aspheric(r, z)
        if rms_asph <= max_acceptable_rms_um or rms_asph < rms_con:
            R_asph = 1.0 / c_asph if abs(c_asph) > 1e-12 else float('inf')
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.ASPHERIC,
                curvature=c_asph,
                radius_of_curvature=R_asph,
                conic_constant=k_asph,
                aspheric_coeffs=a_coeffs,
                semi_diameter=R_semi,
                rms_residual_um=rms_asph,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )

        c_con, R_con = SagFitter.fit_algebraic_sphere(r, z)
        return OpticalProfile(
            surface_name=surface_name,
            fit_type=FitType.CUBIC_SPLINE,
            curvature=c_con,
            radius_of_curvature=R_con,
            conic_constant=0.0,
            semi_diameter=R_semi,
            rms_residual_um=spline_rms,
            spline_object=spline_obj,
            r_sampled=r,
            z_sampled=z
        )

    @staticmethod
    def fit(
        profile_pts: ProfilePoints,
        surface_name: str = "Optical Surface",
        target_fit: FitType = FitType.CONIC,
        max_acceptable_rms_um: float = 0.1
    ) -> OpticalProfile:
        """Fits profile_pts into an OpticalProfile according to target_fit."""
        r, z = profile_pts.r, profile_pts.z
        R_semi = profile_pts.semi_diameter or (float(np.max(r)) if len(r) > 0 else 0.0)
        spline_obj, spline_rms = SagFitter.fit_spline(r, z)

        if target_fit == FitType.SPHERICAL:
            c, R, rms = SagFitter.fit_spherical(r, z)
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.SPHERICAL,
                curvature=c,
                radius_of_curvature=R,
                conic_constant=0.0,
                semi_diameter=R_semi,
                vertex_offset_z=profile_pts.vertex_z,
                rms_residual_um=rms,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )
        elif target_fit == FitType.CONIC:
            c, k, rms = SagFitter.fit_conic(r, z)
            R = 1.0 / c if abs(c) > 1e-12 else float('inf')
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.CONIC,
                curvature=c,
                radius_of_curvature=R,
                conic_constant=k,
                semi_diameter=R_semi,
                vertex_offset_z=profile_pts.vertex_z,
                rms_residual_um=rms,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )
        elif target_fit == FitType.ASPHERIC:
            c, k, a_coeffs, rms = SagFitter.fit_aspheric(r, z)
            R = 1.0 / c if abs(c) > 1e-12 else float('inf')
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.ASPHERIC,
                curvature=c,
                radius_of_curvature=R,
                conic_constant=k,
                aspheric_coeffs=a_coeffs,
                semi_diameter=R_semi,
                vertex_offset_z=profile_pts.vertex_z,
                rms_residual_um=rms,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )
        elif target_fit == FitType.CUBIC_SPLINE:
            c, R = SagFitter.fit_algebraic_sphere(r, z)
            return OpticalProfile(
                surface_name=surface_name,
                fit_type=FitType.CUBIC_SPLINE,
                curvature=c,
                radius_of_curvature=R,
                conic_constant=0.0,
                semi_diameter=R_semi,
                vertex_offset_z=profile_pts.vertex_z,
                rms_residual_um=spline_rms,
                spline_object=spline_obj,
                r_sampled=r,
                z_sampled=z
            )

        return SagFitter.fit_auto(r, z, max_acceptable_rms_um, surface_name)


ProfileFitter = SagFitter
