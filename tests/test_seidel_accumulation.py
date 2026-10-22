"""
Unit and Integration Tests for Zemax-Style Seidel Lens Accumulation & Final Aberrations.
Verifies:
1. Surface-by-surface Seidel accumulation (S_I .. S_V, C_L, C_T).
2. Lens grouping into physical singlets, doublets, triplets, filters, and reflective elements.
3. Lens-by-lens Seidel accumulation progression through the optical train.
4. Final aberrations accumulation matching total system aberrations to machine precision.
5. Zemax OpticStudio text report generation.
6. GUI Seidel Aberrations tab integration and Ray Fan Seidel accumulation view.
"""

import os
import sys
import unittest
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.analysis.aberration import (
    calculate_seidel_coefficients,
    group_surfaces_into_lenses,
    generate_zemax_seidel_report,
    SeidelAberrations,
    LensAberrationResult,
    ParaxialSurfaceResult,
)


class TestSeidelAccumulation(unittest.TestCase):
    """Test suite for Seidel accumulation for every lens and final system accumulation."""

    def test_single_lens_accumulation(self):
        """Single singlet lens: lens accumulation equals final accumulation."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface(name="Singlet_S1", radius=50.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="Singlet_S2", radius=-50.0, thickness=95.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(sys)

        self.assertIsInstance(seidel, SeidelAberrations)
        self.assertEqual(len(seidel.lens_details), 1)

        lens1 = seidel.lens_details[0]
        self.assertEqual(lens1.lens_index, 1)
        self.assertEqual(lens1.surface_indices, [1, 2])
        self.assertEqual(lens1.lens_type, "Singlet")
        self.assertEqual(lens1.glass_types, ["N-BK7"])

        # Lens contribution must equal lens accumulated value
        self.assertAlmostEqual(lens1.S_I, lens1.acc_S_I, places=7)
        self.assertAlmostEqual(lens1.S_II, lens1.acc_S_II, places=7)
        self.assertAlmostEqual(lens1.S_III, lens1.acc_S_III, places=7)
        self.assertAlmostEqual(lens1.S_IV, lens1.acc_S_IV, places=7)
        self.assertAlmostEqual(lens1.S_V, lens1.acc_S_V, places=7)

        # Final accumulation must equal total system aberrations
        final = seidel.final_accumulation
        self.assertAlmostEqual(final["S_I"], seidel.S_I, places=7)
        self.assertAlmostEqual(final["S_II"], seidel.S_II, places=7)
        self.assertAlmostEqual(final["S_III"], seidel.S_III, places=7)
        self.assertAlmostEqual(final["S_IV"], seidel.S_IV, places=7)
        self.assertAlmostEqual(final["S_V"], seidel.S_V, places=7)
        self.assertAlmostEqual(lens1.acc_S_I, seidel.S_I, places=7)

    def test_multi_lens_train_accumulation_progression(self):
        """Multi-lens system: each lens accumulates progressively to the final total."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        # Lens 1: Collimating Singlet
        stack.add_surface(Surface(name="L1_Front", radius=60.0, thickness=6.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="L1_Rear", radius=-120.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
        # Lens 2: Achromatic Doublet (Crown + Flint)
        stack.add_surface(Surface(name="Doublet_S1", radius=45.0, thickness=7.0, glass="N-BAF10", semi_diameter=12.5))
        stack.add_surface(Surface(name="Doublet_S2", radius=-35.0, thickness=3.0, glass="SF11", semi_diameter=12.5))
        stack.add_surface(Surface(name="Doublet_S3", radius=-180.0, thickness=50.0, glass="AIR", semi_diameter=12.5))
        # Lens 3: Focusing Singlet
        stack.add_surface(Surface(name="L3_Front", radius=80.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="L3_Rear", radius=-80.0, thickness=80.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))

        sys = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(sys)

        self.assertEqual(len(seidel.lens_details), 3)

        l1, l2, l3 = seidel.lens_details
        self.assertEqual(l1.lens_type, "Singlet")
        self.assertEqual(l2.lens_type, "Doublet")
        self.assertEqual(l3.lens_type, "Singlet")

        # Accumulation step checks
        self.assertAlmostEqual(l1.acc_S_I, l1.S_I, places=7)
        self.assertAlmostEqual(l2.acc_S_I, l1.S_I + l2.S_I, places=7)
        self.assertAlmostEqual(l3.acc_S_I, l1.S_I + l2.S_I + l3.S_I, places=7)

        # Same for all Seidel and chromatic terms
        self.assertAlmostEqual(l3.acc_S_II, l1.S_II + l2.S_II + l3.S_II, places=7)
        self.assertAlmostEqual(l3.acc_S_III, l1.S_III + l2.S_III + l3.S_III, places=7)
        self.assertAlmostEqual(l3.acc_S_IV, l1.S_IV + l2.S_IV + l3.S_IV, places=7)
        self.assertAlmostEqual(l3.acc_S_V, l1.S_V + l2.S_V + l3.S_V, places=7)
        self.assertAlmostEqual(l3.acc_C_L, l1.C_L + l2.C_L + l3.C_L, places=7)
        self.assertAlmostEqual(l3.acc_C_T, l1.C_T + l2.C_T + l3.C_T, places=7)

        # Final accumulated value matches total system aberrations
        self.assertAlmostEqual(l3.acc_S_I, seidel.S_I, places=7)
        self.assertAlmostEqual(l3.acc_S_II, seidel.S_II, places=7)
        self.assertAlmostEqual(l3.acc_S_III, seidel.S_III, places=7)
        self.assertAlmostEqual(l3.acc_S_IV, seidel.S_IV, places=7)
        self.assertAlmostEqual(l3.acc_S_V, seidel.S_V, places=7)
        self.assertAlmostEqual(l3.acc_C_L, seidel.C_L, places=7)
        self.assertAlmostEqual(l3.acc_C_T, seidel.C_T, places=7)

    def test_surface_running_accumulation(self):
        """Surface-level running accumulation terminates at total system aberrations."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface(name="S1", radius=50.0, thickness=6.0, glass="N-BK7"))
        stack.add_surface(Surface(name="S2", radius=-70.0, thickness=20.0, glass="AIR"))
        stack.add_surface(Surface(name="S3", radius=40.0, thickness=5.0, glass="F2"))
        stack.add_surface(Surface(name="S4", radius=-40.0, thickness=60.0, glass="AIR"))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(sys)

        # Last optical surface must have acc_S_I equal to total S_I
        last_surf = seidel.surface_details[-1]
        self.assertAlmostEqual(last_surf.acc_S_I, seidel.S_I, places=7)
        self.assertAlmostEqual(last_surf.acc_S_II, seidel.S_II, places=7)
        self.assertAlmostEqual(last_surf.acc_S_III, seidel.S_III, places=7)
        self.assertAlmostEqual(last_surf.acc_S_IV, seidel.S_IV, places=7)
        self.assertAlmostEqual(last_surf.acc_S_V, seidel.S_V, places=7)
        self.assertAlmostEqual(last_surf.acc_C_L, seidel.C_L, places=7)
        self.assertAlmostEqual(last_surf.acc_C_T, seidel.C_T, places=7)

    def test_zemax_report_text_generation(self):
        """Verify that generate_zemax_seidel_report produces formatted Zemax output."""
        stack = SurfaceStack()
        stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=100.0))
        stack.add_surface(Surface(name="Lens1_S1", radius=50.0, thickness=5.0, glass="N-BK7"))
        stack.add_surface(Surface(name="Lens1_S2", radius=-50.0, thickness=95.0, glass="AIR"))
        stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        sys = OpticalSystem(stack)
        seidel = calculate_seidel_coefficients(sys)
        report = generate_zemax_seidel_report(seidel)

        self.assertIn("ZEMAX OPTICSTUDIO SEIDEL ABERRATION ANALYSIS", report)
        self.assertIn("LENS-BY-LENS SEIDEL ABERRATIONS", report)
        self.assertIn("FINAL SYSTEM ABERRATIONS ACCUMULATION", report)
        self.assertIn("Spherical Aberration (S_I):", report)
        self.assertIn("Coma (S_II):", report)
        self.assertIn("Astigmatism (S_III):", report)
        self.assertIn("Petzval Field Curvature (S_IV):", report)
        self.assertIn("Distortion (S_V):", report)

    def test_full_spectrometer_model_seidel_accumulation(self):
        """Trace a real spectrometer JSON model and verify lens accumulation against total."""
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spectrometer_yz_40mm_100mm_final.json")
        if not os.path.exists(json_path):
            self.skipTest(f"Sample JSON file {json_path} not found")

        sys = OpticalSystem.from_json(json_path)
        seidel = calculate_seidel_coefficients(sys)

        self.assertGreater(len(seidel.lens_details), 0)
        final_lens = seidel.lens_details[-1]

        # The final lens's accumulated aberrations must exactly equal the total system aberrations
        self.assertAlmostEqual(final_lens.acc_S_I, seidel.S_I, places=4)
        self.assertAlmostEqual(final_lens.acc_S_II, seidel.S_II, places=4)
        self.assertAlmostEqual(final_lens.acc_S_III, seidel.S_III, places=4)
        self.assertAlmostEqual(final_lens.acc_S_IV, seidel.S_IV, places=4)
        self.assertAlmostEqual(final_lens.acc_S_V, seidel.S_V, places=4)
        self.assertAlmostEqual(final_lens.acc_C_L, seidel.C_L, places=4)
        self.assertAlmostEqual(final_lens.acc_C_T, seidel.C_T, places=4)

    def test_gui_seidel_tab_exposure(self):
        """Verify AnalysisTabsWidget exposes Seidel Aberrations tab and handles update_plot."""
        from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
        from optics_app.gui.analysis_tabs import AnalysisTabsWidget, SeidelAberrationTab, HAS_MPL_QT

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([]) if QT_AVAILABLE else None
        tabs = AnalysisTabsWidget()

        self.assertIn("Seidel Aberrations", tabs.tabs)
        if QT_AVAILABLE:
            self.assertTrue(hasattr(tabs, "tab_seidel"))
            self.assertIsInstance(tabs.tab_seidel, SeidelAberrationTab)

            if HAS_MPL_QT:
                stack = SurfaceStack()
                stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=100.0))
                stack.add_surface(Surface(name="L1_Front", radius=50.0, thickness=5.0, glass="N-BK7"))
                stack.add_surface(Surface(name="L1_Rear", radius=-50.0, thickness=50.0, glass="AIR"))
                stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0))
                sys = OpticalSystem(stack)

                tabs.tab_seidel.update_plot(system=sys)
                self.assertIsNotNone(tabs.tab_seidel.last_seidel_data)
                self.assertEqual(len(tabs.tab_seidel.last_seidel_data.lens_details), 1)

            tabs.close()

    def test_finite_vs_infinite_conjugate_object_discrimination(self):
        """Verify finite conjugate object (d0 > 0) traces with y0=0, while infinite (d0=0) traces with u0=0."""
        from optics_app.analysis.aberration import trace_paraxial_marginal_ray

        # 1. Finite conjugate object (microscope / finite sample)
        stack_finite = SurfaceStack()
        stack_finite.add_surface(Surface(name="Sample", surface_type=SurfaceType.OBJECT, thickness=25.0))
        stack_finite.add_surface(Surface(name="S1", radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack_finite.add_surface(Surface(name="S2", radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
        stack_finite.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=0.0, semi_diameter=12.5))
        stack_finite.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        trace_finite = trace_paraxial_marginal_ray(stack_finite)
        # For finite object, ray must originate at the center of the object (y0 = 0.0)
        self.assertAlmostEqual(trace_finite[0][0], 0.0, places=6)
        self.assertGreater(abs(trace_finite[0][1]), 0.0)  # non-zero divergence angle u0

        # 2. Infinite conjugate object (distant object / collimated input)
        stack_infinite = SurfaceStack()
        stack_infinite.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=0.0))
        stack_infinite.add_surface(Surface(name="S1", radius=30.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack_infinite.add_surface(Surface(name="S2", radius=-50.0, thickness=40.0, glass="AIR", semi_diameter=12.5))
        stack_infinite.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=0.0, semi_diameter=12.5))
        stack_infinite.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0))

        trace_infinite = trace_paraxial_marginal_ray(stack_infinite)
        # For infinite object, ray must enter parallel to the optical axis (u0 = 0.0) with non-zero pupil height y0
        self.assertAlmostEqual(trace_infinite[0][1], 0.0, places=6)
        self.assertGreater(abs(trace_infinite[0][0]), 0.0)

    def test_ray_fan_tab_seidel_unit_selector(self):
        """Verify RayFanTab exposes combo_seidel_units supporting um and mm scaling."""
        from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
        from optics_app.gui.analysis_tabs import RayFanTab, HAS_MPL_QT

        if not QT_AVAILABLE or not HAS_MPL_QT:
            return

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tab = RayFanTab()
        self.assertTrue(hasattr(tab, "combo_seidel_units"))
        self.assertEqual(tab.combo_seidel_units.count(), 2)
        self.assertEqual(tab.combo_seidel_units.itemData(0), "um")
        self.assertEqual(tab.combo_seidel_units.itemData(1), "mm")

        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="L2", radius=-60.0, thickness=50.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        sys = OpticalSystem(stack)

        tab.update_plot(system=sys)
        tab.close()

    def test_wavefront_opd_and_transverse_blur_conversions(self):
        """Verify W040..W311 wavefront OPD and transverse ray blur conversions from classical Seidel sums."""
        seidel = SeidelAberrations(
            S_I=-0.040,      # mm -> -40 µm sum
            S_II=-0.010,     # mm -> -10 µm sum
            S_III=0.016,     # mm -> +16 µm sum
            S_IV=0.008,      # mm -> +8 µm sum
            S_V=0.006,       # mm -> +6 µm sum
            C_L=-0.005,      # mm -> -5 µm
            C_T=0.002,       # mm -> +2 µm
            image_marginal_u=0.2,
            image_marginal_n=1.0,
        )

        # W040 = S_I / 8 = -0.040 / 8 * 1000 = -5.0 µm
        self.assertAlmostEqual(seidel.W_040_um, -5.0, places=6)
        # W131 = S_II / 2 = -0.010 / 2 * 1000 = -5.0 µm
        self.assertAlmostEqual(seidel.W_131_um, -5.0, places=6)
        # W222 = S_III / 2 = 0.016 / 2 * 1000 = 8.0 µm
        self.assertAlmostEqual(seidel.W_222_um, 8.0, places=6)
        # W220 = (S_III + S_IV) / 4 = (0.016 + 0.008) / 4 * 1000 = 6.0 µm
        self.assertAlmostEqual(seidel.W_220_um, 6.0, places=6)
        # W311 = S_V / 2 = 0.006 / 2 * 1000 = 3.0 µm
        self.assertAlmostEqual(seidel.W_311_um, 3.0, places=6)
        # Chromatic
        self.assertAlmostEqual(seidel.W_CL_um, -5.0, places=6)
        self.assertAlmostEqual(seidel.W_CT_um, 2.0, places=6)
        # TA_sph = S_I / (2 * n * u) = -0.040 / (2 * 1.0 * 0.2) * 1000 = -100.0 µm
        self.assertAlmostEqual(seidel.TA_sph_um, -100.0, places=6)

    def test_seidel_tab_metric_selector(self):
        """Verify SeidelAberrationTab combo_units supports all 5 standard metric representations."""
        from optics_app.gui.qt_compat import QT_AVAILABLE, QtWidgets
        from optics_app.gui.analysis_tabs import SeidelAberrationTab, HAS_MPL_QT

        if not QT_AVAILABLE or not HAS_MPL_QT:
            return

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        tab = SeidelAberrationTab()
        self.assertTrue(hasattr(tab, "combo_units"))
        self.assertEqual(tab.combo_units.count(), 5)
        self.assertEqual(tab.combo_units.itemData(0), "wavefront_um")
        self.assertEqual(tab.combo_units.itemData(1), "waves")
        self.assertEqual(tab.combo_units.itemData(2), "transverse_um")
        self.assertEqual(tab.combo_units.itemData(3), "um")
        self.assertEqual(tab.combo_units.itemData(4), "mm")

        stack = SurfaceStack()
        stack.add_surface(Surface(name="Obj", surface_type=SurfaceType.OBJECT, thickness=50.0))
        stack.add_surface(Surface(name="L1", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=12.5))
        stack.add_surface(Surface(name="Stop", surface_type=SurfaceType.STOP, thickness=5.0, semi_diameter=5.0))
        stack.add_surface(Surface(name="L2", radius=-60.0, thickness=50.0, glass="AIR", semi_diameter=12.5))
        stack.add_surface(Surface(name="Det", surface_type=SurfaceType.DETECTOR, thickness=0.0))
        sys = OpticalSystem(stack)

        for mode_idx in range(tab.combo_units.count()):
            tab.combo_units.setCurrentIndex(mode_idx)
            tab.update_plot(system=sys)
            self.assertIsNotNone(tab.last_seidel_data)

        tab.close()

    def test_raman_spectrometer_micrometer_scale_aberrations(self):
        """Verify the 4-bit PLM Raman spectrometer produces realistic micrometer-scale aberrations."""
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spectrometer_raman_plm_4bit.json")
        if not os.path.exists(json_path):
            self.skipTest(f"File {json_path} not found")

        sys = OpticalSystem.from_json(json_path)
        seidel = calculate_seidel_coefficients(sys)

        # Aberrations must NOT blow up into millimeters; wavefront OPD must be micrometer order
        self.assertLess(abs(seidel.W_040_um), 25.0)  # spherical OPD < 25 µm
        self.assertLess(abs(seidel.W_131_um), 15.0)  # coma OPD < 15 µm
        self.assertLess(abs(seidel.W_222_um), 25.0)  # astigmatism OPD < 25 µm
        self.assertLess(abs(seidel.TA_sph_um), 250.0) # transverse ray blur < 250 µm


if __name__ == "__main__":
    unittest.main()


