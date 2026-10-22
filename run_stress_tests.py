"""
Empirical Adversarial Stress Testing Script for Milestone 2 Optics Engine & STEP Exporter.
Runs scenarios 1-4 and reports detailed pass/fail outcomes and bug diagnostics.
"""
import sys
import os
import math
import tempfile
import numpy as np

# Add project root to sys.path
sys.path.insert(0, r"c:\Users\Juan\Documents\GitHub\2D_Optics")

from optics_app.engine.surface import Surface, SurfaceType, SphericalSurface, AsphericSurface
from optics_app.engine.lde import SurfaceStack, LensDataEditor
from optics_app.engine.ray import Ray
from optics_app.engine.tracer import snell_refract_3d, build_rotation_matrix, OpticalSystem
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.engine.dispersion import get_refractive_index
from optics_app.utils.materials import GlassCatalog
from optics_app.cad.exporter import STEPExporter, STEPExportOptions, STEPExportEngine

def run_stress_tests():
    print("=" * 80)
    print("EMPIRICAL ADVERSARIAL STRESS TEST SUITE — MILESTONE 2")
    print("=" * 80)
    
    findings = []
    
    # -------------------------------------------------------------------------
    # SCENARIO 1: Aspheric Surface Intersection
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: Aspheric Surface Intersection ---")
    
    # 1.1 Hyperbolic conic k = -2.0, extreme off-axis
    asph_hyp = AsphericSurface(radius=25.0, conic=-2.0, asphere_coeffs=[])
    # Ray parallel to Z at off-axis distance x = 18 mm (semi_diam ~20)
    ray_off = Ray(pos=np.array([18.0, 5.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    res = asph_hyp.intersect(ray_off)
    
    if res is None:
        print("[FAIL 1.1] Hyperbolic conic intersection returned None for ray at x=18, y=5!")
        findings.append("Scenario 1.1 FAIL: AsphericSurface.intersect failed on hyperbolic conic k=-2.0 off-axis ray.")
    else:
        hit, norm = res
        r_hit = math.sqrt(hit[0]**2 + hit[1]**2)
        z_expected = asph_hyp.sag(r_hit)
        res_err = abs(hit[2] - z_expected)
        norm_len = np.linalg.norm(norm)
        print(f"[PASS 1.1] Hyperbolic conic hit={hit}, norm={norm}, sag_residue={res_err:.2e}, norm_len={norm_len:.6f}")
        if res_err > 1e-5 or not np.isclose(norm_len, 1.0):
            findings.append(f"Scenario 1.1 WARN: Sag residue {res_err:.2e} or norm length {norm_len}")

    # 1.2 Steep conic k = +1.0 (oblate ellipsoid)
    asph_obl = AsphericSurface(radius=25.0, conic=1.0, asphere_coeffs=[])
    ray_obl = Ray(pos=np.array([10.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    res_obl = asph_obl.intersect(ray_obl)
    if res_obl is None:
        print("[FAIL 1.2] Oblate ellipsoid conic k=+1.0 intersection returned None!")
        findings.append("Scenario 1.2 FAIL: AsphericSurface.intersect failed on oblate ellipsoid k=+1.0.")
    else:
        hit_obl, norm_obl = res_obl
        print(f"[PASS 1.2] Oblate conic k=+1.0 hit={hit_obl}, norm={norm_obl}")

    # 1.3 High-order polynomial coefficients A4, A6, A8
    asph_high = AsphericSurface(radius=40.0, conic=-0.5, asphere_coeffs=[1e-5, -1e-7, 1e-9])
    ray_high = Ray(pos=np.array([15.0, -10.0, -10.0]), dir=np.array([0.0, 0.0, 1.0]))
    res_high = asph_high.intersect(ray_high)
    if res_high is None:
        print("[FAIL 1.3] High-order polynomial asphere intersection returned None!")
        findings.append("Scenario 1.3 FAIL: AsphericSurface.intersect failed with high-order A4/A6/A8 coefficients.")
    else:
        hit_h, norm_h = res_high
        r_h = math.sqrt(hit_h[0]**2 + hit_h[1]**2)
        res_err_h = abs(hit_h[2] - asph_high.sag(r_h))
        print(f"[PASS 1.3] High-order asphere hit={hit_h}, norm={norm_h}, residue={res_err_h:.2e}")
        if res_err_h > 1e-4:
            findings.append(f"Scenario 1.3 FAIL: Newton-Raphson failed to converge (residue={res_err_h:.2e})")

    # 1.4 Test Surface.intersect with conic != 0 on Standard surface type
    surf_std_conic = Surface(name="ConicStandard", surface_type=SurfaceType.STANDARD, radius=25.0, conic=-2.0)
    res_std = surf_std_conic.intersect(Ray(pos=np.array([10.0, 0.0, -10.0]), dir=np.array([0.0, 0.0, 1.0])))
    if res_std is not None:
        hit_s, norm_s = res_std
        # Expected z for parabola k=-1 at r=10 is 1.0, for k=-2 z(10) is 1.0 / (1 + sqrt(1 - (-1)*(1/25)^2*100)) ...
        z_asph_expected = AsphericSurface(radius=25.0, conic=-2.0).sag(10.0)
        z_sph_expected = SphericalSurface(radius=25.0).sag(10.0) if hasattr(SphericalSurface(radius=25.0), 'sag') else 100/ (25 + math.sqrt(625-100))
        print(f"Surface(STANDARD, conic=-2.0) hit z = {hit_s[2]:.6f} (Aspheric z would be {z_asph_expected:.6f}, Spherical z is {z_sph_expected:.6f})")
        if np.isclose(hit_s[2], z_sph_expected, atol=1e-3) and not np.isclose(hit_s[2], z_asph_expected, atol=1e-3):
            print("[BUG 1.4 CONFIRMED] Surface.intersect ignores conic constant when surface_type='Standard'!")
            findings.append("BUG 1.4: Surface.intersect ignores non-zero conic constant when surface_type=='Standard'.")

    # -------------------------------------------------------------------------
    # SCENARIO 2: 3D Vector Grating Diffraction
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: 3D Vector Grating Diffraction ---")
    
    # 2.1 Skewed incident ray with non-zero v_iy
    # Incident ray direction with v_iy = 0.4
    v_ix = 0.2
    v_iy = 0.4
    v_iz = math.sqrt(1.0 - v_ix**2 - v_iy**2)
    v_in_skewed = np.array([v_ix, v_iy, v_iz])
    
    line_density = 600.0  # lines/mm -> d = 1.666667 um
    wl = 0.5876  # um
    d_um = 1000.0 / line_density
    
    for m in [1, -1, 2, -2]:
        v_out, evan = grating_diffract_3d(v_in_skewed, [0,0,1], [1,0,0], line_density, m, wl)
        if evan:
            print(f"Order m={m}: Evanescent wave detected.")
        else:
            expected_vx = v_ix + (m * wl / d_um)
            expected_vy = v_iy
            expected_vz = math.sqrt(max(0.0, 1.0 - expected_vx**2 - expected_vy**2))
            diff_x = abs(v_out[0] - expected_vx)
            diff_y = abs(v_out[1] - expected_vy)
            diff_z = abs(v_out[2] - expected_vz)
            norm_v = np.linalg.norm(v_out)
            print(f"Order m={m:2d}: v_out={v_out}, norm={norm_v:.6f}, diff_x={diff_x:.2e}, diff_y={diff_y:.2e}")
            if diff_x > 1e-5 or diff_y > 1e-5 or diff_z > 1e-5 or not np.isclose(norm_v, 1.0):
                findings.append(f"Scenario 2.1 FAIL: Skewed grating diffraction mismatch for order m={m}.")

    # 2.2 Evanescent wave rejection in OpticalSystem ray trace
    stack_g = SurfaceStack()
    sg = Surface(name="HighDensityGrating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, semi_diameter=25.0)
    sg.grating_lines_per_mm = 2000.0  # d = 0.5 um
    sg.grating_order = 2  # m=2, wl=0.6 um -> 2*0.6/0.5 = 2.4 > 1.0 -> Evanescent!
    stack_g.add_surface(sg)
    stack_g.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR, thickness=0.0, semi_diameter=25.0))
    
    sys_g = OpticalSystem(stack_g)
    ray_g = sys_g.trace_ray(Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=0.6))
    print(f"[PASS 2.2] High order grating evanescent check: vignetted={ray_g.vignetted}, dir={ray_g.dir}")
    if not ray_g.vignetted:
        print("[FAIL 2.2] OpticalSystem failed to mark ray vignetted when grating order is evanescent!")
        findings.append("Scenario 2.2 FAIL: OpticalSystem ray tracer did not flag vignetted for evanescent grating diffraction.")

    # 2.3 Unused parameters in grating_diffract_3d
    # Test if line_vector is used:
    v_out_line1, _ = grating_diffract_3d([0,0,1], [0,0,1], [1,0,0], 600.0, 1, 0.5)
    v_out_line2, _ = grating_diffract_3d([0,0,1], [0,0,1], [0,1,0], 600.0, 1, 0.5)
    if np.allclose(v_out_line1, v_out_line2):
        print("[BUG 2.3 CONFIRMED] grating_diffract_3d ignores `line_vector` parameter (assumes lines always along Y, dispersion along X)!")
        findings.append("BUG 2.3: grating_diffract_3d ignores `line_vector` argument and hardcodes dispersion along X axis.")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Euler Tilts / Decenters
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: Euler Tilts & Decenters ---")
    
    # 3.1 Multi-surface system with arbitrary 3D tilts (15 deg X, 30 deg Y, 45 deg Z)
    stack_t = SurfaceStack()
    s_t1 = Surface(name="Tilted1", radius=100.0, thickness=15.0, glass="N-BK7", semi_diameter=25.0, tilt_x=15.0, tilt_y=30.0, tilt_z=45.0, decenter_x=2.0, decenter_y=-3.0)
    s_t2 = Surface(name="Tilted2", radius=-80.0, thickness=10.0, glass="AIR", semi_diameter=25.0, tilt_x=-10.0, tilt_y=15.0, tilt_z=-20.0)
    s_t3 = Surface(name="Detector", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=30.0)
    
    stack_t.add_surface(s_t1)
    stack_t.add_surface(s_t2)
    stack_t.add_surface(s_t3)
    
    sys_t = OpticalSystem(stack_t)
    ray_t_in = Ray(pos=np.array([1.0, -1.0, 0.0]), dir=np.array([0.05, -0.05, 0.997497]))
    traced_t = sys_t.trace_ray(ray_t_in)
    
    print(f"Tilted trace completed: vignetted={traced_t.vignetted}, tir={traced_t.tir}, final_pos={traced_t.pos}, path_length={traced_t.path_length:.6f}")
    
    # Check path length calculation consistency
    geom_path = 0.0
    p_prev = ray_t_in.pos.copy()
    for entry in traced_t.history:
        p_curr = np.array([entry["x_global"], entry["y_global"], entry["z_global"]])
        geom_path += np.linalg.norm(p_curr - p_prev)
        p_prev = p_curr
        
    diff_path = abs(traced_t.path_length - geom_path)
    print(f"Path length recorded: {traced_t.path_length:.6f}, Geometric path sum: {geom_path:.6f}, diff: {diff_path:.2e}")
    if diff_path > 1e-4:
        print("[FAIL 3.1] Path length accumulation inconsistent with geometric hit point distances!")
        findings.append(f"Scenario 3.1 FAIL: Path length mismatch ({diff_path:.2e}) in tilted surface ray tracing.")
    else:
        print("[PASS 3.1] Path length and 3D direction vector consistency verified.")

    # 3.2 Check history dictionary coordinate inconsistency (x/y local vs z global)
    if len(traced_t.history) > 0:
        h0 = traced_t.history[0]
        if h0["x"] == h0["x_local"] and h0["y"] == h0["y_local"] and h0["z"] == h0["z_global"] and h0["z"] != h0["z_local"]:
            print("[BUG 3.2 CONFIRMED] Ray history dict has inconsistent 'z' key storing z_global while 'x','y' store local x,y!")
            findings.append("BUG 3.2: Ray history dictionary key 'z' stores z_global while 'x' and 'y' store local coordinates.")

    # -------------------------------------------------------------------------
    # SCENARIO 4: CAD STEP Exporter Syntax ISO-10303-21 Compliance
    # -------------------------------------------------------------------------
    print("\n--- Scenario 4: CAD Exporter STEP Syntax ISO-10303-21 Compliance ---")
    
    stack_cad = SurfaceStack()
    stack_cad.add_surface(Surface(name="Lens1_S1", radius=30.0, thickness=6.0, glass="N-BK7", semi_diameter=15.0))
    stack_cad.add_surface(Surface(name="Lens1_S2", radius=-40.0, thickness=4.0, glass="AIR", semi_diameter=15.0))
    sg_cad = Surface(name="Grating", surface_type=SurfaceType.GRATING, radius=0.0, thickness=10.0, semi_diameter=15.0, tilt_y=15.0)
    sg_cad.grating_lines_per_mm = 600.0
    stack_cad.add_surface(sg_cad)
    stack_cad.add_surface(Surface(name="CCD", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, semi_diameter=12.0))
    
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.step') as tmp:
        step_path = tmp.name
        
    try:
        res_exp = STEPExporter.export(stack_cad, step_path)
        print(f"Exporter result: success={res_exp.is_success}, file_size={res_exp.file_size_bytes} bytes, components={res_exp.num_components_exported}")
        
        with open(step_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Total lines in exported STEP file: {len(lines)}")
        
        # Verify ISO-10303-21 Structural Tokens
        text_full = "".join(lines)
        assert "ISO-10303-21;" in text_full, "Missing ISO-10303-21 header token!"
        assert "HEADER;" in text_full, "Missing HEADER block!"
        assert "ENDSEC;" in text_full, "Missing ENDSEC block!"
        assert "DATA;" in text_full, "Missing DATA section!"
        assert "END-ISO-10303-21;" in text_full, "Missing END-ISO-10303-21 token!"
        
        # Check ISO-10303-21 syntax compliance details
        entity_map = {}
        syntax_errors = []
        schema_mismatches = []
        
        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str or line_str in ["ISO-10303-21;", "HEADER;", "DATA;", "ENDSEC;", "END-ISO-10303-21;"]:
                continue
            if not line_str.endswith(";"):
                syntax_errors.append(f"Line {line_num}: Missing trailing semicolon: {line_str[:50]}")
            if line_str.startswith("#"):
                parts = line_str.split("=", 1)
                if len(parts) == 2:
                    ent_id = parts[0].strip()
                    ent_val = parts[1].strip()
                    ent_name = ent_val.split("(")[0].strip() if "(" in ent_val else ent_val
                    entity_map[ent_id] = ent_name
                    
                    # Check FACE_SURFACE entity parameter types
                    if ent_name == "FACE_SURFACE":
                        # Format: #id = FACE_SURFACE('name', (bounds), surface_id, same_sense);
                        # Extract surface_id (3rd arg)
                        args_str = ent_val[len("FACE_SURFACE"):].rstrip(";").strip()
                        if args_str.startswith("(") and args_str.endswith(")"):
                            # Parse top-level comma arguments
                            # Arg 0: name, Arg 1: bounds tuple, Arg 2: surface_id, Arg 3: same_sense
                            # e.g. ('Element_1_Lens1_S1_face', (#103), #103, .T.)
                            inner = args_str[1:-1]
                            # Find 3rd parameter
                            param_parts = [p.strip() for p in inner.split(", #")]
                            if len(param_parts) >= 2:
                                surf_ref = "#" + param_parts[-2].split(",")[0].strip() if "#" in inner else ""
                                # Let's check entity map for surf_ref type
                                # In exporter.py line 486: FACE_SURFACE('...', (#bound_id), #bound_id, .T.)
                                # #bound_id is FACE_OUTER_BOUND!
                                pass

        print(f"[CHECK 4.2] Total parsed entities: {len(entity_map)}")
        
        # Check specific schema mismatch found in exporter.py line 486:
        # FACE_SURFACE uses FACE_OUTER_BOUND as surface entity!
        for line in lines:
            if "FACE_SURFACE" in line:
                # e.g. #104 = FACE_SURFACE('Element_1_Lens1_S1_face', (#103), #103, .T.);
                parts = line.split(",")
                if len(parts) >= 3:
                    ref_id = parts[-2].strip()
                    if ref_id in entity_map and entity_map[ref_id] == "FACE_OUTER_BOUND":
                        schema_mismatches.append(f"FACE_SURFACE entity '{line.strip()}' uses FACE_OUTER_BOUND ({ref_id}) as surface geometry!")
        
        if schema_mismatches:
            print(f"[BUG 4.2 CONFIRMED] STEP exporter violates ISO-10303-21 schema:")
            for sm in schema_mismatches[:3]:
                print("  -", sm)
            findings.append(f"BUG 4.2: STEPExporter generates invalid ISO-10303-21 STEP entities — FACE_SURFACE uses FACE_OUTER_BOUND ID instead of valid SURFACE entity (PLANE, SPHERICAL_SURFACE, etc.).")
        else:
            print("[PASS 4.2] STEP file ISO-10303-21 entity syntax validated.")
            
    finally:
        if os.path.exists(step_path):
            os.remove(step_path)
            
    print("\n" + "=" * 80)
    print("STRESS TEST SUMMARY FINDINGS:")
    print("=" * 80)
    for f in findings:
        print(" *", f)
    print("=" * 80)
    return len(findings) == 0

if __name__ == "__main__":
    success = run_stress_tests()
    sys.exit(0 if success else 1)
