"""
Adversarial Harness for Milestone 2 Optics Engine and STEP Exporter.
Runs comprehensive empirical tests across Scenarios 1-4.
"""
import os
import sys
import math
import tempfile
import traceback
import numpy as np

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import snell_refract_3d, build_rotation_matrix, OpticalSystem
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.engine.dispersion import get_refractive_index, SellmeierDispersion, CauchyDispersion
from optics_app.utils.materials import GlassCatalog
from optics_app.engine.ccd import CCDSensor, CCDMatrix
from optics_app.io.zmx_parser import parse_zmx_string, parse_zmx_file
from optics_app.cad.exporter import STEPExporter, CADExporter, export_to_step, STEPExportEngine, STEPExportOptions


def run_scenario_1_newton_raphson():
    print("=== SCENARIO 1: Newton-Raphson Convergence & Extreme Radii ===")
    results = []

    # 1.1 Extreme Radii Test (R = 5mm to 10000mm, convex and concave)
    radii_to_test = [5.0, 10.0, 50.0, 200.0, 1000.0, 5000.0, 10000.0, -5.0, -10.0, -50.0, -1000.0, -10000.0]
    conics_to_test = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0]

    radii_pass = 0
    radii_total = 0

    for R in radii_to_test:
        for k in conics_to_test:
            radii_total += 1
            asph = AsphericSurface(radius=R, conic=k, asphere_coeffs=[1e-6, -1e-9])
            # Ray along Z at height y=2.0 mm
            ray = Ray(pos=np.array([0.0, 2.0, -20.0]), dir=np.array([0.0, 0.0, 1.0]))
            res = asph.intersect(ray)
            if res is None:
                results.append(f"[FAIL] NR intersection returned None for R={R}, k={k}")
                continue

            hit_pos, normal = res
            r_hit = float(np.sqrt(hit_pos[0]**2 + hit_pos[1]**2))
            z_sag = asph.sag(r_hit)
            residual = abs(hit_pos[2] - z_sag)
            norm_magnitude = float(np.linalg.norm(normal))

            if residual < 1e-6 and abs(norm_magnitude - 1.0) < 1e-6:
                radii_pass += 1
            else:
                results.append(f"[FAIL] NR residual={residual:.2e}, norm_mag={norm_magnitude:.6f} for R={R}, k={k}")

    print(f"Radii & Conics NR test: {radii_pass}/{radii_total} passed")

    # 1.2 Near-Grazing Incidence Rays
    # Test angles: 70, 80, 85, 89, 89.9, 89.99 degrees
    grazing_angles_deg = [70.0, 80.0, 85.0, 89.0, 89.9, 89.99]
    grazing_pass = 0
    grazing_total = len(grazing_angles_deg)

    asph_grazing = AsphericSurface(radius=50.0, conic=-1.0, asphere_coeffs=[1e-7, 1e-10])

    for angle_deg in grazing_angles_deg:
        theta_rad = math.radians(angle_deg)
        # Ray direction in Y-Z plane
        dir_vec = np.array([0.0, math.sin(theta_rad), math.cos(theta_rad)])
        # Origin chosen so ray aims towards surface origin
        pos_vec = np.array([0.0, -10.0 * math.sin(theta_rad), -10.0 * math.cos(theta_rad)])
        ray = Ray(pos=pos_vec, dir=dir_vec)

        res = asph_grazing.intersect(ray)
        if res is None:
            results.append(f"[WARN/FAIL] NR returned None for grazing angle {angle_deg} deg")
            continue

        hit_pos, normal = res
        r_hit = float(np.sqrt(hit_pos[0]**2 + hit_pos[1]**2))
        z_sag = asph_grazing.sag(r_hit)
        residual = abs(hit_pos[2] - z_sag)

        if residual < 1e-5:
            grazing_pass += 1
            print(f" Grazing angle {angle_deg:6.2f}° -> Hit: [{hit_pos[0]:.4f}, {hit_pos[1]:.4f}, {hit_pos[2]:.4f}], Residual: {residual:.2e}")
        else:
            results.append(f"[FAIL] Grazing angle {angle_deg}° high residual: {residual:.2e}")

    print(f"Grazing incidence NR test: {grazing_pass}/{grazing_total} passed")

    # 1.3 Aspheric Normal Vector Correctness Check
    asph_norm = AsphericSurface(radius=50.0, conic=-1.0) # parabola
    ray_norm = Ray(pos=np.array([10.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    hit_p, norm_p = asph_norm.intersect(ray_norm)
    expected_nx = -0.2 / np.sqrt(1.04)
    expected_nz = 1.0 / np.sqrt(1.04)

    if np.isclose(norm_p[0], expected_nx, atol=1e-3) and np.isclose(norm_p[2], expected_nz, atol=1e-3):
        print(f" Aspheric surface normal vector check PASSED: [{norm_p[0]:.6f}, {norm_p[1]:.6f}, {norm_p[2]:.6f}]")
    else:
        results.append(f"[FAIL] Aspheric surface normal incorrect: got {norm_p}, expected [{expected_nx:.6f}, 0, {expected_nz:.6f}]")

    return results


def run_scenario_2_grating_equation():
    print("\n=== SCENARIO 2: Grating Equation & Dispersion Vector ===")
    results = []

    # 2.1 Boundary Conditions sin(theta_m) = +-1
    g = DiffractionGrating(lines_per_mm=600.0, order=1)
    pitch_um = 1000.0 / 600.0 # 1.666667 um
    # Solve for incident wavelength/angle giving sin(theta_m) = 1.0 exactly
    # sin(theta_m) = sin(theta_i) + order * lambda / pitch = 1.0
    # Let theta_i = 0 => lambda = pitch = 1.6666666666666667 um
    wl_boundary_pos = pitch_um
    theta_m_pos = g.calculate_diffraction_angle(0.0, wl_boundary_pos)
    print(f" Boundary condition sin(theta_m) = +1.0: theta_m = {math.degrees(theta_m_pos):.6f}°")
    if not np.isclose(math.degrees(theta_m_pos), 90.0, atol=1e-4):
        results.append(f"[FAIL] Boundary +1.0 angle mismatch: {math.degrees(theta_m_pos)} deg")

    # Boundary condition sin(theta_m) = -1.0 with order = -1
    g_neg = DiffractionGrating(lines_per_mm=600.0, order=-1)
    theta_m_neg = g_neg.calculate_diffraction_angle(0.0, wl_boundary_pos)
    print(f" Boundary condition sin(theta_m) = -1.0: theta_m = {math.degrees(theta_m_neg):.6f}°")
    if not np.isclose(math.degrees(theta_m_neg), -90.0, atol=1e-4):
        results.append(f"[FAIL] Boundary -1.0 angle mismatch: {math.degrees(theta_m_neg)} deg")

    # Evanescent wave handling (sin(theta_m) > 1.0)
    wl_evan = pitch_um + 0.1
    try:
        g.calculate_diffraction_angle(0.0, wl_evan)
        results.append("[FAIL] Evanescent wave did not raise ValueError in DiffractionGrating!")
    except ValueError as e:
        print(f" Evanescent wave correctly caught by DiffractionGrating: {e}")

    # 3D Vector evanescent check
    v_out, evan = grating_diffract_3d([0, 0, 1], [0, 0, 1], [1, 0, 0], 600.0, 1, wl_evan)
    if evan and v_out is None:
        print(" 3D Vector grating correctly flagged evanescent wave.")
    else:
        results.append("[FAIL] 3D Vector grating failed to flag evanescent wave!")

    # 2.2 Multi-Wavelength Dispersion Vector Calculation
    wavelengths = [0.400, 0.500, 0.600, 0.700] # um
    line_density = 600.0
    order = 1
    d_um = 1000.0 / line_density

    print(" Multi-wavelength 3D dispersion vectors:")
    dispersion_vectors = []
    for wl in wavelengths:
        v_in = np.array([0.1, 0.2, math.sqrt(1.0 - 0.01 - 0.04)]) # off-axis ray
        v_out, evan = grating_diffract_3d(v_in, [0, 0, 1], [1, 0, 0], line_density, order, wl)
        if evan:
            results.append(f"[FAIL] Wavelength {wl} um resulted in unexpected evanescent wave!")
            continue

        expected_vx = v_in[0] + order * wl / d_um
        expected_vy = v_in[1]
        expected_vz = math.sqrt(1.0 - expected_vx**2 - expected_vy**2)

        norm_vout = np.linalg.norm(v_out)
        print(f"   wl={wl:.3f} um -> v_out=[{v_out[0]:.6f}, {v_out[1]:.6f}, {v_out[2]:.6f}], norm={norm_vout:.6f}")

        if not np.isclose(v_out[0], expected_vx, atol=1e-6):
            results.append(f"[FAIL] Grating X component mismatch for wl={wl}: got {v_out[0]}, exp {expected_vx}")
        if not np.isclose(v_out[1], expected_vy, atol=1e-6):
            results.append(f"[FAIL] Grating Y component modified for wl={wl}: got {v_out[1]}, exp {expected_vy}")
        if not np.isclose(norm_vout, 1.0, atol=1e-6):
            results.append(f"[FAIL] Grating v_out not normalized for wl={wl}: norm={norm_vout}")

        dispersion_vectors.append(v_out)

    # Check angular dispersion separation delta_theta / delta_lambda
    sin_m1 = dispersion_vectors[0][0] # wl = 0.4 um
    sin_m4 = dispersion_vectors[-1][0] # wl = 0.7 um
    spatial_spread = sin_m4 - sin_m1
    expected_spread = order * (0.700 - 0.400) / d_um
    print(f" Wavelength dispersion spread (sin_m(0.7) - sin_m(0.4)): {spatial_spread:.6f} (Expected: {expected_spread:.6f})")
    if not np.isclose(spatial_spread, expected_spread, atol=1e-6):
        results.append(f"[FAIL] Dispersion spread mismatch: got {spatial_spread}, expected {expected_spread}")

    # 2.3 Sign Convention Consistency check
    theta_i_rad = math.radians(20.0)
    sin_i = math.sin(theta_i_rad)
    g_check = DiffractionGrating(lines_per_mm=600.0, order=1)
    calc_angle = g_check.calculate_diffraction_angle(theta_i_rad, 0.5)
    calc_sin_m = math.sin(calc_angle)

    v_in_sign = np.array([sin_i, 0.0, math.cos(theta_i_rad)])
    v_out_sign, _ = grating_diffract_3d(v_in_sign, [0, 0, 1], [1, 0, 0], 600.0, 1, 0.5)

    if np.isclose(calc_sin_m, v_out_sign[0], atol=1e-5):
        print(f" Grating sign convention consistency PASSED (sin_m = {calc_sin_m:.6f})")
    else:
        results.append(f"[FAIL] Grating sign convention mismatch: DiffractionGrating={calc_sin_m}, 3d={v_out_sign[0]}")

    return results


def run_scenario_3_decenters_tilts():
    print("\n=== SCENARIO 3: Decenters & Tilts Accumulation ===")
    results = []

    # 3.1 Rotation Matrix Orthonormality & Composition
    test_angles = [
        (10.0, 20.0, 30.0),
        (45.0, -45.0, 90.0),
        (-30.0, 60.0, -15.0),
        (180.0, 0.0, 0.0)
    ]
    for rx, ry, rz in test_angles:
        R = build_rotation_matrix(rx, ry, rz)
        det_R = np.linalg.det(R)
        ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
        if not (np.isclose(det_R, 1.0, atol=1e-12) and ortho_err < 1e-12):
            results.append(f"[FAIL] Rotation matrix ({rx}, {ry}, {rz}) failed orthonormality: det={det_R}, ortho_err={ortho_err}")
    print(" Rotation matrix orthonormality verified across test angles.")

    # 3.2 Sequential Ray Tracing through Multiple Tilted & Decentered Surfaces
    stack = SurfaceStack()
    s1 = Surface(name="Surface1_TiltedX", radius=100.0, thickness=10.0, glass="N-BK7", semi_diameter=30.0, decenter_x=2.0, tilt_x=15.0)
    s2 = Surface(name="Surface2_TiltedY", radius=-100.0, thickness=15.0, glass="AIR", semi_diameter=30.0, decenter_y=-3.0, tilt_y=-10.0)
    s3 = Surface(name="Surface3_TiltedZ", radius=80.0, thickness=10.0, glass="F2", semi_diameter=30.0, tilt_z=45.0)
    s4 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=40.0)

    stack.add_surface(s1)
    stack.add_surface(s2)
    stack.add_surface(s3)
    stack.add_surface(s4)

    sys_opt = OpticalSystem(stack)
    ray = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.5876)

    try:
        traced = sys_opt.trace_ray(ray)
        print(f" Sequential trace through 4 tilted/decentered surfaces completed:")
        for idx, h in enumerate(traced.history):
            print(f"   Surf {idx} ({stack.surfaces[idx].name}): Local=[{h['x_local']:.4f}, {h['y_local']:.4f}, {h['z_local']:.4f}], Global=[{h['x_global']:.4f}, {h['y_global']:.4f}, {h['z_global']:.4f}]")

        if traced.vignetted:
            results.append("[FAIL] Ray unexpectedly vignetted in decentered/tilted stack test!")
        if np.any(np.isnan(traced.pos)) or np.any(np.isnan(traced.dir)):
            results.append("[FAIL] Traced ray pos/dir contains NaN!")
        if not np.isclose(np.linalg.norm(traced.dir), 1.0, atol=1e-6):
            results.append(f"[FAIL] Final ray direction vector not unit length: norm={np.linalg.norm(traced.dir)}")
    except Exception as e:
        results.append(f"[FAIL] Exception during sequential trace through tilted surfaces: {e}")

    # 3.3 Mathematical Check of Sequential Transformation Consistency
    # Check that pos_global = origin + R_mat @ pos_local at each surface
    for idx, h in enumerate(traced.history):
        surf = stack.surfaces[idx]
        R_mat = build_rotation_matrix(surf.tilt_x, surf.tilt_y, surf.tilt_z)
        origin = np.array([surf.decenter_x, surf.decenter_y, surf.global_z])
        pos_local = np.array([h['x_local'], h['y_local'], 0.0]) # hit in local frame z=0 plane or sag
        hit_pos_local = np.array([h['x_local'], h['y_local'], surf.sag(math.sqrt(h['x_local']**2 + h['y_local']**2))])
        reconstructed_global = origin + R_mat @ hit_pos_local
        hit_global_actual = np.array([h['x_global'], h['y_global'], h['z_global']])
        diff = np.linalg.norm(reconstructed_global - hit_global_actual)
        if diff > 1e-4:
            results.append(f"[FAIL] Global reconstruction mismatch for surface {idx}: diff={diff:.6f}")
        else:
            print(f"   Surface {idx} coordinate reconstruction diff: {diff:.2e}")

    return results


def run_scenario_4_zmx_and_cad_exporter():
    print("\n=== SCENARIO 4: Zemax Parser & 3D STEP Exporter ===")
    results = []

    # 4.1 Zemax .zmx Parsing Test
    sample_zmx = """
    ! Standard Doublet Prescription File
    UNIT MM
    SURF 0
      TYPE OBJECT
      THIC INFINITY
    SURF 1
      TYPE STANDARD
      CURV 0.02
      THIC 6.0
      GLAS N-BK7
      DIAM 25.0
    SURF 2
      TYPE EVENASPH
      CURV -0.015
      THIC 2.0
      GLAS AIR
      DIAM 25.0
      CONI -0.5
      PARM 1 1e-6
      PARM 2 -2e-9
    SURF 3
      TYPE GRATING
      CURV 0.0
      THIC 15.0
      GLAS AIR
      DIAM 20.0
      PARM 1 600.0
    SURF 4
      TYPE DETECTOR
      THIC 0.0
      DIAM 15.0
    """

    try:
        stack = parse_zmx_string(sample_zmx)
        print(f" Parsed .zmx string: {len(stack.surfaces)} surfaces loaded")
        if len(stack.surfaces) != 5:
            results.append(f"[FAIL] Expected 5 surfaces from .zmx string, got {len(stack.surfaces)}")

        # Check surface parameters
        s1 = stack.surfaces[1]
        if not (np.isclose(s1.radius, 50.0) and np.isclose(s1.thickness, 6.0) and s1.glass == "N-BK7"):
            results.append(f"[FAIL] Surface 1 parameters incorrect: R={s1.radius}, t={s1.thickness}, glass={s1.glass}")

        s2 = stack.surfaces[2]
        if not (s2.surface_type == SurfaceType.ASPHERIC and np.isclose(s2.conic, -0.5) and len(s2.asphere_coeffs) == 2):
            results.append(f"[FAIL] Surface 2 aspheric parameters incorrect: conic={s2.conic}, coeffs={s2.asphere_coeffs}")

        s3 = stack.surfaces[3]
        if not (s3.surface_type == SurfaceType.GRATING and np.isclose(s3.grating_lines_per_mm, 600.0)):
            results.append(f"[FAIL] Surface 3 grating parameters incorrect: lines/mm={s3.grating_lines_per_mm}")

    except Exception as e:
        results.append(f"[FAIL] Exception parsing .zmx string: {e}\n{traceback.format_exc()}")

    # 4.2 File Encodings Test for parse_zmx_file
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    for enc in encodings:
        with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.zmx') as tmp:
            tmp.write(sample_zmx.encode(enc))
            tmp_path = tmp.name

        try:
            s_file = parse_zmx_file(tmp_path)
            if len(s_file.surfaces) != 5:
                results.append(f"[FAIL] Encoding {enc} failed to parse all surfaces (got {len(s_file.surfaces)})")
            else:
                print(f" Encoding {enc:10s} successfully parsed .zmx file.")
        except Exception as e:
            results.append(f"[FAIL] Encoding {enc} threw exception: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 4.3 3D STEP Assembly Exporter Test
    try:
        stack_export = SurfaceStack()
        e1 = Surface(name="Lens1_Front", radius=40.0, thickness=5.0, glass="N-BK7", semi_diameter=20.0)
        e2 = Surface(name="Lens1_Back", radius=-50.0, thickness=2.0, glass="AIR", semi_diameter=20.0)
        e3 = Surface(name="Grating_Substrate", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, glass="AIR", semi_diameter=15.0, tilt_y=10.0)
        e3.grating_lines_per_mm = 600.0
        e4 = Surface(name="CCD_Sensor", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=12.0)

        stack_export.add_surface(e1)
        stack_export.add_surface(e2)
        stack_export.add_surface(e3)
        stack_export.add_surface(e4)

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.step') as tmp:
            step_file_path = tmp.name

        export_res = STEPExporter.export(stack_export, step_file_path)
        print(f" STEP export result: is_success={export_res.is_success}, components={export_res.num_components_exported}, size={export_res.file_size_bytes} bytes")

        if not export_res.is_success:
            results.append(f"[FAIL] STEPExporter.export returned is_success=False: {export_res.error_message}")
        if export_res.file_size_bytes < 500:
            results.append(f"[FAIL] Generated STEP file suspiciously small: {export_res.file_size_bytes} bytes")

        with open(step_file_path, 'r', encoding='utf-8') as f:
            step_txt = f.read()

        required_step_tokens = [
            "ISO-10303-21;",
            "HEADER;",
            "FILE_DESCRIPTION",
            "FILE_SCHEMA",
            "DATA;",
            "CARTESIAN_POINT",
            "POLY_LOOP",
            "FACE_OUTER_BOUND",
            "FACE_SURFACE",
            "CLOSED_SHELL",
            "MANIFOLD_SOLID_BREP",
            "PRODUCT",
            "ENDSEC;",
            "END-ISO-10303-21;"
        ]

        for token in required_step_tokens:
            if token not in step_txt:
                results.append(f"[FAIL] STEP file missing required ISO-10303-21 entity/header token: '{token}'")

        print(" ISO-10303-21 STEP structure & syntax validation PASSED.")

        # Test CADExporter compatibility alias
        cad_res_path = CADExporter.export_system_step(stack_export, step_file_path)
        if not os.path.exists(cad_res_path):
            results.append("[FAIL] CADExporter.export_system_step compatibility alias failed!")

        # Test convenience function export_to_step
        convenience_res = export_to_step(stack_export, step_file_path)
        if not convenience_res.is_success:
            results.append("[FAIL] export_to_step convenience function failed!")

        if os.path.exists(step_file_path):
            os.remove(step_file_path)

    except Exception as e:
        results.append(f"[FAIL] Exception in STEP exporter test: {e}\n{traceback.format_exc()}")

    return results


def main():
    print("=" * 70)
    print("EMPIRICAL ADVERSARIAL STRESS TEST HARNESS — MILESTONE 2")
    print("=" * 70)

    all_failures = []

    res1 = run_scenario_1_newton_raphson()
    all_failures.extend(res1)

    res2 = run_scenario_2_grating_equation()
    all_failures.extend(res2)

    res3 = run_scenario_3_decenters_tilts()
    all_failures.extend(res3)

    res4 = run_scenario_4_zmx_and_cad_exporter()
    all_failures.extend(res4)

    print("\n" + "=" * 70)
    print("SUMMARY OF ADVERSARIAL TESTING FINDINGS")
    print("=" * 70)

    if not all_failures:
        print("ALL SCENARIOS PASSED WITH 0 FAILURES OR ANOMALIES!")
        sys.exit(0)
    else:
        print(f"FOUND {len(all_failures)} FAILURES / ANOMALIES:")
        for idx, f in enumerate(all_failures, 1):
            print(f" {idx}. {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
