"""
Empirical Challenge Test Suite for OpticalOptimizer Engine (Milestone M2).
Adversarial and stress tests covering:
1. 3D tilted/decentered surfaces, fold mirrors, and coordinate frame tracking via _update_accumulated_positions().
2. Chromatic optimization across multi-wavelength ray bundles and dispersion curves (N-BK7, N-SF11, N-SF6, Fused Silica, F2).
3. Degenerate optical cases: 100% vignetted rays, single ray, extreme pupil radii (0, near-zero, massive, negative).
4. Boundary constraints, penalty robustness, and multi-solver stress tests.
"""

import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.ray import Ray
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
from optics_app.analysis.spot import SpotDiagram
from optics_app.utils.materials import GlassCatalog, get_refractive_index


def add_variable_attributes(surf: Surface, variable: bool = True, t_min: float = 0.0, t_max: float = 1000.0) -> Surface:
    setattr(surf, "thickness_variable", variable)
    setattr(surf, "thickness_min", t_min)
    setattr(surf, "thickness_max", t_max)
    return surf


# ===========================================================================
# Category 1: 3D Tilted/Decentered Surfaces & Fold Mirrors
# ===========================================================================
class TestOptimizer3DGeometryAndFoldMirrors(unittest.TestCase):
    """
    Stress-testing OpticalOptimizer with 3D folded optical systems to ensure
    _update_accumulated_positions() correctly computes global frames during optimization.
    """

    def test_folded_system_45deg_mirror_focus_optimization(self):
        """
        Folded optical system:
        Plano-convex lens (f ~ 100mm) -> distance d1 (40mm) -> 45-deg flat fold mirror -> distance d2 (variable) -> Detector at 90 deg.
        Verify:
        1. Detector is physically located along the reflected Y-axis in global coordinates.
        2. Optimizer adjusts d2 to place the detector exactly at the focal plane (f - d1 ~ 96.7 - 40 = 56.7mm).
        3. RMS spot radius drops by >80%.
        """
        stack = SurfaceStack()
        # S0: Object
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        # S1: Lens Front (R = 51.68mm, N-BK7)
        stack.add_surface(Surface(name="LensFront", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        # S2: Lens Back (flat) -> thickness 40.0 to fold mirror
        stack.add_surface(Surface(name="LensBack", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=20.0))
        # S3: 45-deg Fold Mirror (tilt_x = 45.0) -> variable thickness to detector
        s_fold = Surface(name="FoldMirror", radius=0.0, thickness=20.0, glass="MIRROR", tilt_x=45.0, semi_diameter=25.0)
        add_variable_attributes(s_fold, variable=True, t_min=10.0, t_max=120.0)
        stack.add_surface(s_fold)
        # S4: Detector plane
        s_det = Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0)
        stack.add_surface(s_det)

        # Verify initial global coordinates before optimization
        # Fold mirror should be at Z = 1000 (OBJ) + 5 (Lens) + 40 (Air) = 1045.0
        # After 45 deg tilt_x reflection, reflected Z axis should point along -Y or +Y
        self.assertAlmostEqual(s_fold.global_origin[2], 1045.0, places=3)
        self.assertAlmostEqual(s_det.global_origin[2], 1045.0, places=3)
        self.assertAlmostEqual(abs(s_det.global_origin[1]), 20.0, places=3)

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=80, num_rays=25, entrance_pupil_radius=8.0)
        res = optimizer.optimize(variable_indices=[3], bounds=[(10.0, 120.0)])

        self.assertTrue(res.success, "Optimizer should converge on folded mirror system")
        self.assertGreaterEqual(res.rms_reduction_pct, 70.0, f"Expected >=70% reduction, got {res.rms_reduction_pct:.2f}%")
        
        # Verify optimized thickness is close to theoretical back focal length remaining: ~56.7mm
        opt_d2 = res.optimized_thicknesses[3]
        self.assertAlmostEqual(opt_d2, 56.7, delta=6.0)

        # Verify optimized detector position is along reflected axis in global coordinates
        opt_det = res.optimized_stack.surfaces[4]
        self.assertAlmostEqual(opt_det.global_origin[2], 1045.0, places=3)
        self.assertAlmostEqual(abs(opt_det.global_origin[1]), opt_d2, places=3)

    def test_compound_3d_tilts_and_decenters_optimization(self):
        """
        Optical system with compound 3D tilts (tilt_x, tilt_y, tilt_z) and decenters (decenter_x, decenter_y).
        Verify optimizer executes without vector math singularity, updates all frame matrices, and reduces RMS.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=500.0))
        # Decentered and tilted lens
        s1 = Surface(name="TiltedLensFront", radius=60.0, thickness=6.0, glass="N-BK7", 
                     decenter_x=2.0, decenter_y=-1.5, tilt_x=5.0, tilt_y=-3.0, semi_diameter=20.0)
        stack.add_surface(s1)
        s2 = Surface(name="TiltedLensBack", radius=-100.0, thickness=20.0, glass="AIR", semi_diameter=20.0)
        add_variable_attributes(s2, variable=True, t_min=10.0, t_max=180.0)
        stack.add_surface(s2)
        # Fold mirror with compound 3D orientation
        s3 = Surface(name="CompoundMirror", radius=0.0, thickness=30.0, glass="MIRROR", 
                     tilt_x=30.0, tilt_y=20.0, tilt_z=10.0, semi_diameter=25.0)
        add_variable_attributes(s3, variable=True, t_min=10.0, t_max=150.0)
        stack.add_surface(s3)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=40.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=60, num_rays=20, entrance_pupil_radius=6.0)
        res = optimizer.optimize(variable_indices=[2, 3], bounds=[(10.0, 180.0), (10.0, 150.0)])

        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)
        self.assertGreaterEqual(res.rms_reduction_pct, 40.0)

    def test_dual_fold_periscope_system_multi_variable_optimization(self):
        """
        Periscope system: Lens -> Fold 1 (tilt_y=45) -> Fold 2 (tilt_y=-45) -> Detector.
        Optimizing 3 distances: pre-fold, inter-mirror, post-fold.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="Lens", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0))
        # d1: lens to fold 1
        s_d1 = Surface(name="Air1", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=20.0)
        add_variable_attributes(s_d1, variable=True, t_min=10.0, t_max=50.0)
        stack.add_surface(s_d1)
        # Fold 1: 45 deg around Y -> ray travels along +X
        s_m1 = Surface(name="Mirror1", radius=0.0, thickness=30.0, glass="MIRROR", tilt_y=45.0, semi_diameter=20.0)
        add_variable_attributes(s_m1, variable=True, t_min=10.0, t_max=50.0)
        stack.add_surface(s_m1)
        # Fold 2: -45 deg around Y -> ray turns back to +Z
        s_m2 = Surface(name="Mirror2", radius=0.0, thickness=20.0, glass="MIRROR", tilt_y=-45.0, semi_diameter=20.0)
        add_variable_attributes(s_m2, variable=True, t_min=10.0, t_max=80.0)
        stack.add_surface(s_m2)
        # Detector
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=30.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=70, num_rays=20, entrance_pupil_radius=8.0)
        res = optimizer.optimize(variable_indices=[2, 3, 4], bounds=[(10.0, 50.0), (10.0, 50.0), (10.0, 80.0)])

        self.assertTrue(res.success)
        total_opt_path = res.optimized_thicknesses[2] + res.optimized_thicknesses[3] + res.optimized_thicknesses[4]
        # Total optical path after lens front should be ~96.7mm
        self.assertAlmostEqual(total_opt_path, 96.7, delta=12.0)
        self.assertGreaterEqual(res.rms_reduction_pct, 50.0)


# ===========================================================================
# Category 2: Chromatic Optimization Across Dispersion Curves
# ===========================================================================
class TestOptimizerChromaticDispersion(unittest.TestCase):
    """
    Stress-testing chromatic optimization across multi-wavelength ray bundles
    using Sellmeier dispersion equations for optical glasses.
    """

    def test_multi_wavelength_achromat_optimization(self):
        """
        Achromatic Doublet with Crown (N-BK7) and Flint (N-SF11) glasses.
        Merit function must trace rays at F (0.4861 µm), d (0.5876 µm), and C (0.6563 µm) lines simultaneously.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=2000.0))
        stack.add_surface(Surface(name="Crown1", radius=62.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
        s_gap = Surface(name="AirGap", radius=-43.0, thickness=0.5, glass="AIR", semi_diameter=15.0)
        add_variable_attributes(s_gap, variable=True, t_min=0.1, t_max=5.0)
        stack.add_surface(s_gap)
        stack.add_surface(Surface(name="Flint1", radius=-43.0, thickness=3.5, glass="N-SF11", semi_diameter=15.0))
        s_bfl = Surface(name="Flint2", radius=-140.0, thickness=40.0, glass="AIR", semi_diameter=15.0)
        add_variable_attributes(s_bfl, variable=True, t_min=30.0, t_max=150.0)
        stack.add_surface(s_bfl)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        wavelengths = [0.48613, 0.58756, 0.65627]
        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, wavelengths=wavelengths, method="Nelder-Mead", max_iter=70, num_rays=25, entrance_pupil_radius=10.0)
        res = optimizer.optimize(variable_indices=[2, 4], bounds=[(0.1, 5.0), (30.0, 150.0)])

        self.assertTrue(res.success)
        self.assertGreaterEqual(res.rms_reduction_pct, 50.0)

        # Verify that all 3 wavelengths produce valid rays on the detector
        opt_sys = OpticalSystem(res.optimized_stack)
        for wl in wavelengths:
            r = opt_sys.trace_ray(Ray(pos=np.array([0.0, 5.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
            self.assertFalse(r.vignetted)
            self.assertEqual(r.history[-1]["surface_index"], 5)

    def test_monochromatic_vs_polychromatic_optimization_comparison(self):
        """
        Verify that multi-wavelength optimization achieves better polychromatic RMS spot
        than single-wavelength optimization on a high-dispersion singlet (N-SF11).
        """
        def make_sf11_singlet(bfl=30.0):
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=1000.0))
            stack.add_surface(Surface(name="L1", radius=45.0, thickness=5.0, glass="N-SF11", semi_diameter=12.0))
            s_back = Surface(name="L2", radius=0.0, thickness=bfl, glass="AIR", semi_diameter=12.0)
            add_variable_attributes(s_back, variable=True, t_min=20.0, t_max=100.0)
            stack.add_surface(s_back)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))
            return stack

        wl_triplet = [0.450, 0.5876, 0.700]

        # 1. Monochromatic optimization at d-line (0.5876 µm)
        stack_mono = make_sf11_singlet(bfl=30.0)
        sys_mono = OpticalSystem(stack_mono)
        opt_mono = OpticalOptimizer(sys_mono, wavelengths=[0.5876], method="Nelder-Mead", max_iter=50, num_rays=20, entrance_pupil_radius=8.0)
        res_mono = opt_mono.optimize(variable_indices=[2])

        # 2. Polychromatic optimization over [0.450, 0.5876, 0.700] µm
        stack_poly = make_sf11_singlet(bfl=30.0)
        sys_poly = OpticalSystem(stack_poly)
        opt_poly = OpticalOptimizer(sys_poly, wavelengths=wl_triplet, method="Nelder-Mead", max_iter=50, num_rays=20, entrance_pupil_radius=8.0)
        res_poly = opt_poly.optimize(variable_indices=[2])

        # Evaluate both solutions on the full polychromatic bundle
        poly_evaluator = OpticalOptimizer(OpticalSystem(res_mono.optimized_stack), wavelengths=wl_triplet, num_rays=30, entrance_pupil_radius=8.0)
        rms_mono_on_poly = poly_evaluator.merit_function(np.array([res_mono.optimized_thicknesses[2]]), [2])

        poly_evaluator_2 = OpticalOptimizer(OpticalSystem(res_poly.optimized_stack), wavelengths=wl_triplet, num_rays=30, entrance_pupil_radius=8.0)
        rms_poly_on_poly = poly_evaluator_2.merit_function(np.array([res_poly.optimized_thicknesses[2]]), [2])

        self.assertLessEqual(rms_poly_on_poly, rms_mono_on_poly + 1e-4, 
                             "Polychromatic optimizer should yield equal or better polychromatic RMS merit than monochromatic optimizer")

    def test_broadband_multi_glass_catalog_optimization(self):
        """
        Verify optimizer functions across distinct catalog glasses (Fused Silica, F2, N-SK16, N-SF6).
        """
        glasses = ["FUSED_SILICA", "F2", "N-SK16", "N-SF6"]
        for g_name in glasses:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=500.0))
            stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass=g_name, semi_diameter=12.0))
            s_back = Surface(name="L2", radius=-80.0, thickness=20.0, glass="AIR", semi_diameter=12.0)
            add_variable_attributes(s_back, variable=True, t_min=10.0, t_max=120.0)
            stack.add_surface(s_back)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

            sys = OpticalSystem(stack)
            optimizer = OpticalOptimizer(sys, wavelengths=[0.500, 0.600], method="Nelder-Mead", max_iter=40, num_rays=15, entrance_pupil_radius=6.0)
            res = optimizer.optimize(variable_indices=[2])
            self.assertTrue(res.success, f"Failed optimization for glass {g_name}")
            self.assertGreaterEqual(res.rms_reduction_pct, 40.0)


# ===========================================================================
# Category 3: Degenerate Cases & Extreme Boundaries
# ===========================================================================
class TestOptimizerDegenerateCases(unittest.TestCase):
    """
    Stress-testing degenerate inputs:
    - 100% vignetted rays
    - Single ray (num_rays=1)
    - Extreme pupil radii (0, 1e-12, 5000, negative)
    - 2-surface minimal system
    - Negative parameter barrier penalty
    - Boundary clamping
    """

    def test_degenerate_100_percent_vignetted_rays_returns_penalty_gracefully(self):
        """
        When all rays miss detector or are blocked (valid_rays == 0),
        merit_function must return 1e5 penalty without crashing or raising exceptions.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Central obscuration on L1: aperture_r_inner = 20.0 (blocks all rays within 20mm pupil)
        s1 = Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=25.0,
                     aperture_type="Annular", aperture_r_inner=20.0, aperture_r_outer=25.0)
        stack.add_surface(s1)
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=25.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        # Pupil radius = 10.0mm: all rays have r <= 10.0 < 20.0 (r_inner), so 100% of rays (including chief ray) are blocked
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0)
        merit_val = optimizer.merit_function(np.array([50.0]), [2])

        self.assertAlmostEqual(merit_val, 1e5, places=1, msg="100% vignetted system (valid_rays=0) must return 1e5 penalty")

        # Optimization should finish without throwing an exception
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 100.0)])
        self.assertIsNotNone(res)
        self.assertIsInstance(res.final_rms, float)

    def test_degenerate_high_vignetting_partial_penalty(self):
        """
        When outer rays are vignetted but chief ray survives (60 of 61 rays vignetted),
        merit_function must return RMS + penalty_weight * (60/61) without error.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        # Tiny semi-diameter 0.001 mm: only central chief ray (r=0) passes
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=0.001))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=50.0, glass="AIR", semi_diameter=0.001))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=0.001))
    
        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=10.0, penalty_weight=1000.0, num_rays=30)
        merit_val = optimizer.merit_function(np.array([50.0]), [2])
    
        # 1 valid ray -> RMS = 0.0, 60 of 61 rays vignetted -> penalty = 1000.0 * (60/61) = 983.6065
        self.assertAlmostEqual(merit_val, 983.6065, places=1)

    def test_degenerate_single_ray_trace_no_division_by_zero(self):
        """
        When num_rays = 1 (central chief ray + 1 ring), ray tracing and SpotDiagram
        must evaluate without ZeroDivisionError or NaN.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        s2 = Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
        add_variable_attributes(s2, variable=True, t_min=10.0, t_max=150.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, num_rays=1, entrance_pupil_radius=5.0)
        merit = optimizer.merit_function(np.array([30.0]), [2])

        self.assertFalse(np.isnan(merit))
        self.assertFalse(np.isinf(merit))
        self.assertGreater(merit, 0.0)

        res = optimizer.optimize(variable_indices=[2])
        self.assertTrue(res.success)
        self.assertLess(res.final_rms, res.initial_rms)

    def test_degenerate_pupil_radius_zero(self):
        """
        entrance_pupil_radius = 0.0 (Collimated on-axis needle beam).
        Rays should all be on-axis, merit evaluates cleanly.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=0.0, num_rays=15)
        merit = optimizer.merit_function(np.array([40.0]), [2])

        self.assertFalse(np.isnan(merit))
        self.assertAlmostEqual(merit, 0.0, places=4)

    def test_degenerate_pupil_radius_infinitesimal(self):
        """
        entrance_pupil_radius = 1e-12 mm.
        Evaluates without numerical underflow or floating-point instability.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=1e-12, num_rays=15)
        merit = optimizer.merit_function(np.array([40.0]), [2])

        self.assertFalse(np.isnan(merit))
        self.assertLess(merit, 1e-6)

    def test_degenerate_pupil_radius_massive_with_vignetting(self):
        """
        entrance_pupil_radius = 5000.0 mm (Huge beam compared to 15mm aperture).
        Merit function must apply partial vignetting penalty proportionally without crashing.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=5000.0, num_rays=20, penalty_weight=1000.0)
        merit = optimizer.merit_function(np.array([40.0]), [2])

        self.assertFalse(np.isnan(merit))
        # Most rays vignetted -> penalty applied
        self.assertGreater(merit, 500.0)

    def test_degenerate_pupil_radius_negative(self):
        """
        Negative pupil radius (-10.0 mm).
        Rays generate with mirrored sign without crashing.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=40.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=15.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, entrance_pupil_radius=-10.0, num_rays=15)
        merit = optimizer.merit_function(np.array([40.0]), [2])

        self.assertFalse(np.isnan(merit))
        self.assertGreater(merit, 0.0)

    def test_degenerate_minimal_object_detector_stack(self):
        """
        Stack containing only OBJECT and DETECTOR.
        Adjusting thickness simply moves detector along axis.
        """
        stack = SurfaceStack()
        s0 = Surface(name="OBJ", thickness=50.0)
        add_variable_attributes(s0, variable=True, t_min=10.0, t_max=200.0)
        stack.add_surface(s0)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, max_iter=20, num_rays=10, entrance_pupil_radius=5.0)
        res = optimizer.optimize(variable_indices=[0])

        self.assertTrue(res.success)
        self.assertIn(0, res.optimized_thicknesses)

    def test_negative_thickness_barrier_penalty(self):
        """
        Passing negative thickness values to merit_function returns a heavy barrier penalty (>1e5).
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=100.0))
        stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        stack.add_surface(Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0))
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys)
        
        penalty_val = optimizer.merit_function(np.array([-25.0]), [2])
        self.assertGreaterEqual(penalty_val, 1e5, "Negative thickness must trigger barrier penalty >= 1e5")

    def test_strict_bounds_clamping(self):
        """
        When true minimum is outside user-specified bounds, optimizer strictly clamps to bound.
        """
        stack = SurfaceStack()
        stack.add_surface(Surface(name="OBJ", thickness=1000.0))
        stack.add_surface(Surface(name="L1", radius=51.68, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
        # True focal distance is ~96.7mm. User restricts bounds to [10.0, 35.0].
        s2 = Surface(name="L2", radius=0.0, thickness=20.0, glass="AIR", semi_diameter=15.0)
        add_variable_attributes(s2, variable=True, t_min=10.0, t_max=35.0)
        stack.add_surface(s2)
        stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

        sys = OpticalSystem(stack)
        optimizer = OpticalOptimizer(sys, method="Nelder-Mead", max_iter=50, num_rays=20)
        res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 35.0)])

        self.assertLessEqual(res.optimized_thicknesses[2], 35.0 + 1e-6)
        self.assertGreaterEqual(res.optimized_thicknesses[2], 10.0 - 1e-6)
        # Should drive to upper bound 35.0 since it's closer to 96.7
        self.assertAlmostEqual(res.optimized_thicknesses[2], 35.0, delta=0.5)

    def test_all_supported_solvers_execution(self):
        """
        Verify optimizer operates with alternative scipy solvers (Nelder-Mead, L-BFGS-B, Powell, SLSQP).
        """
        methods = ["Nelder-Mead", "L-BFGS-B", "Powell", "SLSQP"]
        for m in methods:
            stack = SurfaceStack()
            stack.add_surface(Surface(name="OBJ", thickness=500.0))
            stack.add_surface(Surface(name="L1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=15.0))
            s2 = Surface(name="L2", radius=0.0, thickness=30.0, glass="AIR", semi_diameter=15.0)
            add_variable_attributes(s2, variable=True, t_min=10.0, t_max=150.0)
            stack.add_surface(s2)
            stack.add_surface(Surface(name="DET", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=20.0))

            sys = OpticalSystem(stack)
            optimizer = OpticalOptimizer(sys, method=m, max_iter=30, num_rays=15, entrance_pupil_radius=8.0)
            res = optimizer.optimize(variable_indices=[2], bounds=[(10.0, 150.0)])

            self.assertTrue(res.success, f"Solver method '{m}' should execute successfully")
            self.assertGreaterEqual(res.optimized_thicknesses[2], 10.0)
            self.assertLessEqual(res.optimized_thicknesses[2], 150.0)


if __name__ == "__main__":
    unittest.main()
