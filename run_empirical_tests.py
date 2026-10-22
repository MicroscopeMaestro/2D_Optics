import os
import sys
import shutil
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.cad.importer import (
    CADImporter,
    CADSurfaceImporter,
    STEPParser,
    CADParseError,
    CADGeometryError,
    CADImportError,
    CADFileFormat
)
from optics_app.cad.slicer import ZSlicer, SlicedLensGeometry
from optics_app.cad.fitter import SagFitter, ProfilePoints, fit_sag_profile
from optics_app.cad.profile import OpticalProfile, FitType

def run_empirical_suite():
    print("======================================================================")
    print(" RUNNING EMPIRICAL CHALLENGE SUITE ON REFACTORED CAD ENGINE ")
    print("======================================================================")
    
    passed_count = 0
    failed_count = 0
    failures = []

    def check(test_name: str, condition: bool, detail: str = ""):
        nonlocal passed_count, failed_count
        if condition:
            passed_count += 1
            print(f"[PASS] {test_name} {detail}")
        else:
            failed_count += 1
            msg = f"[FAIL] {test_name} {detail}"
            print(msg)
            failures.append(msg)

    # ------------------------------------------------------------------
    # CATEGORY 1: Arbitrary Non-Standard Filenames & Formats
    # ------------------------------------------------------------------
    print("\n--- Category 1: Arbitrary Non-Standard Filenames ---")
    sample_dir = os.path.join(os.path.dirname(__file__), "tests", "sample_cad")
    spherical_src = os.path.join(sample_dir, "spherical_lens.step")
    aspheric_src = os.path.join(sample_dir, "aspheric_lens.step")
    sldprt_src = os.path.join(sample_dir, "sample_solidworks.sldprt")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1.1: STEP file renamed to arbitrary non-standard name
        custom_step_1 = os.path.join(tmpdir, "test_element_xyz.step")
        shutil.copy(spherical_src, custom_step_1)
        prof_1 = CADSurfaceImporter.extract_optical_profile(custom_step_1)
        check("1.1 Arbitrary STEP filename (test_element_xyz.step)",
              prof_1 is not None and abs(prof_1.curvature - 0.01) < 1e-4,
              f"Curvature={prof_1.curvature:.4f}")

        # Test 1.2: STEP file renamed to .stp with numbers and spaces
        custom_step_2 = os.path.join(tmpdir, "my_custom optics 123 99.stp")
        shutil.copy(aspheric_src, custom_step_2)
        prof_2 = CADSurfaceImporter.extract_optical_profile(custom_step_2)
        check("1.2 Arbitrary STEP filename with spaces (my_custom optics 123 99.stp)",
              prof_2 is not None and prof_2.conic_constant == -1.0,
              f"Conic={prof_2.conic_constant}")

        # Test 1.3: Uppercase extension
        custom_step_3 = os.path.join(tmpdir, "UPPERCASE_LENS.STEP")
        shutil.copy(spherical_src, custom_step_3)
        prof_3 = CADSurfaceImporter.extract_optical_profile(custom_step_3)
        check("1.3 Uppercase extension (.STEP)",
              prof_3 is not None and abs(prof_3.curvature - 0.01) < 1e-4,
              f"Curvature={prof_3.curvature:.4f}")

        # Test 1.4: Arbitrary SolidWorks filename with parameters inside text
        custom_sw_1 = os.path.join(tmpdir, "weird_optics_component.sldprt")
        with open(custom_sw_1, 'w', encoding='utf-8') as f:
            f.write("SolidWorks Part Header\nPARAM R1=120.0 R2=-120.0 Th=12.5\nEND")
        front_sw, back_sw, th_sw = STEPParser.extract_solidworks_profiles(custom_sw_1)
        check("1.4 Arbitrary SolidWorks filename & custom params",
              abs(front_sw.radius_of_curvature - 120.0) < 1e-4 and abs(th_sw - 12.5) < 1e-4,
              f"R1={front_sw.radius_of_curvature}, Th={th_sw}")

    # ------------------------------------------------------------------
    # CATEGORY 2: Invalid Inputs & Strict Exception Propagation
    # ------------------------------------------------------------------
    print("\n--- Category 2: Invalid Inputs & Error Propagation ---")

    # Test 2.1: Non-existent file path
    try:
        CADSurfaceImporter.extract_optical_profile("non_existent_file_path_12345.step")
        check("2.1 Non-existent file raises FileNotFoundError", False, "No exception raised")
    except FileNotFoundError:
        check("2.1 Non-existent file raises FileNotFoundError", True)
    except Exception as e:
        check("2.1 Non-existent file raises FileNotFoundError", False, f"Raised {type(e).__name__}")

    # Test 2.2: Empty file
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = os.path.join(tmpdir, "empty.step")
        open(empty_file, 'w').close()
        try:
            CADSurfaceImporter.extract_optical_profile(empty_file)
            check("2.2 Zero-byte STEP file raises CADParseError", False, "No exception raised")
        except CADParseError:
            check("2.2 Zero-byte STEP file raises CADParseError", True)
        except Exception as e:
            check("2.2 Zero-byte STEP file raises CADParseError", False, f"Raised {type(e).__name__}")

    # Test 2.3: Corrupt STEP text file (no valid entities)
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_file = os.path.join(tmpdir, "corrupt.step")
        with open(corrupt_file, 'w') as f:
            f.write("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nNOT_AN_ENTITY;\nENDSEC;\nEND-ISO-10303-21;\n")
        try:
            CADSurfaceImporter.extract_optical_profile(corrupt_file)
            check("2.3 Corrupt STEP file with no optical surfaces raises CADGeometryError/CADParseError", False, "No exception raised")
        except (CADParseError, CADGeometryError):
            check("2.3 Corrupt STEP file with no optical surfaces raises CADGeometryError/CADParseError", True)
        except Exception as e:
            check("2.3 Corrupt STEP file with no optical surfaces raises CADGeometryError/CADParseError", False, f"Raised {type(e).__name__}")

    # Test 2.4: ZSlicer invalid inputs
    slicer = ZSlicer()
    try:
        slicer.slice_cad_mesh(None)
        check("2.4 ZSlicer(None) raises ValueError", False, "No exception raised")
    except ValueError:
        check("2.4 ZSlicer(None) raises ValueError", True)
    except Exception as e:
        check("2.4 ZSlicer(None) raises ValueError", False, f"Raised {type(e).__name__}")

    try:
        slicer.slice_cad_mesh("invalid_input_type")
        check("2.5 ZSlicer('string') raises TypeError", False, "No exception raised")
    except TypeError:
        check("2.5 ZSlicer('string') raises TypeError", True)
    except Exception as e:
        check("2.5 ZSlicer('string') raises TypeError", False, f"Raised {type(e).__name__}")

    # ------------------------------------------------------------------
    # CATEGORY 3: Edge Case Geometries & Physical Ray Tracing Bounds
    # ------------------------------------------------------------------
    print("\n--- Category 3: Edge Case Geometries & Ray Tracing Bounds ---")

    # Test 3.1: Flat surface (curvature = 0)
    flat_prof = OpticalProfile(surface_name="Flat Surface", curvature=0.0, conic_constant=0.0, semi_diameter=25.0)
    sag_flat = flat_prof.sag(15.0)
    normal_flat = flat_prof.normal_vector(10.0, 10.0, 0.0)
    t_flat, P_flat, n_flat, valid_flat = flat_prof.intersect_ray((5.0, 5.0, -10.0), (0.0, 0.0, 1.0))
    check("3.1 Flat surface sag & normal",
          sag_flat == 0.0 and normal_flat == (0.0, 0.0, 1.0) and valid_flat and abs(P_flat[2]) < 1e-6,
          f"sag={sag_flat}, P_z={P_flat[2]}")

    # Test 3.2: Concave surface (negative curvature)
    concave_prof = OpticalProfile(surface_name="Concave Surface", curvature=-0.02, conic_constant=0.0, semi_diameter=20.0)
    sag_concave = concave_prof.sag(10.0)
    t_conc, P_conc, n_conc, valid_conc = concave_prof.intersect_ray((10.0, 0.0, -10.0), (0.0, 0.0, 1.0))
    check("3.2 Concave surface negative sag & ray hit",
          sag_concave < 0.0 and valid_conc and abs(P_conc[2] - sag_concave) < 1e-5,
          f"sag={sag_concave:.4f}, P_z={P_conc[2]:.4f}")

    # Test 3.3: Hyperboloid (k = -2.0)
    hyp_prof = OpticalProfile(surface_name="Hyperboloid", curvature=0.02, conic_constant=-2.0, semi_diameter=20.0)
    sag_hyp = hyp_prof.sag(10.0)
    t_hyp, P_hyp, n_hyp, valid_hyp = hyp_prof.intersect_ray((10.0, 0.0, -10.0), (0.0, 0.0, 1.0))
    check("3.3 Hyperboloid (k=-2.0) sag & ray hit",
          not np.isnan(sag_hyp) and valid_hyp and abs(P_hyp[2] - sag_hyp) < 1e-5,
          f"sag={sag_hyp:.4f}, P_z={P_hyp[2]:.4f}")

    # Test 3.4: Oblate Spheroid (k = +2.0) with safety max(0, 1 - (1+k)c^2 r^2)
    oblate_prof = OpticalProfile(surface_name="Oblate Spheroid", curvature=0.02, conic_constant=2.0, semi_diameter=20.0)
    sag_oblate = oblate_prof.sag(15.0)
    check("3.4 Oblate Spheroid (k=+2.0) does not produce NaN",
          not np.isnan(sag_oblate),
          f"sag={sag_oblate}")

    # Test 3.5: Ray Intersection Forward Guard (ray going backward/away from surface)
    # Ray starting at z=+10 moving in +Z direction (0,0,1), surface is at z=0
    # Ray distance t would be -10.
    t_back, P_back, n_back, valid_back = flat_prof.intersect_ray((0.0, 0.0, 10.0), (0.0, 0.0, 1.0))
    check("3.5 Ray forward guard (t <= 0 invalid)",
          not valid_back,
          f"t={t_back}, is_valid={valid_back}")

    # Test 3.6: Ray Intersection Aperture Guard (ray hitting outside clear aperture)
    # Surface semi_diameter is 20.0 mm. Ray hits at r = 25.0 mm.
    bounded_prof = OpticalProfile(curvature=0.01, semi_diameter=20.0)
    t_out, P_out, n_out, valid_out = bounded_prof.intersect_ray((25.0, 0.0, -10.0), (0.0, 0.0, 1.0))
    check("3.6 Ray aperture guard (r > semi_diameter invalid)",
          not valid_out,
          f"r_hit=25.0, semi_diameter=20.0, is_valid={valid_out}")

    # Test 3.7: High-order Aspheric Coefficients
    asph_prof = OpticalProfile(
        curvature=0.01,
        conic_constant=-1.0,
        aspheric_coeffs=[1e-5, -1e-8, 1e-11],
        semi_diameter=20.0
    )
    sag_asph = asph_prof.sag(10.0)
    t_asph, P_asph, n_asph, valid_asph = asph_prof.intersect_ray((5.0, 5.0, -10.0), (0.0, 0.0, 1.0))
    check("3.7 High-order aspheric surface sag & ray hit",
          valid_asph and abs(P_asph[2] - sag_asph_at_r(asph_prof, np.sqrt(50))) < 1e-4,
          f"P_z={P_asph[2]:.4f}")

    # Summary
    print("\n======================================================================")
    print(f" SUITE SUMMARY: {passed_count} PASSED, {failed_count} FAILED")
    print("======================================================================")
    if failed_count > 0:
        print("FAILURES DETECTED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL EMPIRICAL TESTS PASSED SUCCESSFULLY!")

def sag_asph_at_r(prof, r):
    return prof.sag(r)

if __name__ == "__main__":
    run_empirical_suite()
