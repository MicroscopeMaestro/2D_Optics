import math, sys, os
sys.path.insert(0, ".")
import numpy as np

print("=== ADVERSARIAL STRESS TEST SUITE ===")

# Test 1: Matrix propagation optics & physics
from optics_app.engine.matrix_tracer import (
    RayMatrix, space_matrix, thin_lens_matrix, refraction_matrix,
    reflection_matrix, thick_lens_matrix, calculate_cavity_q_factor,
    calculate_spectrometer_q_factor
)

# 1a. Concave mirror Zemax sign convention R = -100mm
m_concave = reflection_matrix(-100.0)
m_prop = space_matrix(50.0)
m_total = m_prop @ m_concave
y_foc, u_foc = m_total.propagate_ray(1.0, 0.0)
assert abs(y_foc) < 1e-12, f"Concave mirror failed to focus: y={y_foc}"
assert abs(m_concave.effective_focal_length() - 50.0) < 1e-9
print("[PASS] 1a: Concave mirror paraxial focus at f = -R/2 = 50.0 mm")

# 1b. Convex mirror R = +100mm
m_convex = reflection_matrix(100.0)
assert abs(m_convex.effective_focal_length() - (-50.0)) < 1e-9
y_div, u_div = (space_matrix(50.0) @ m_convex).propagate_ray(1.0, 0.0)
assert y_div > 1.0, f"Convex mirror failed to diverge: y={y_div}"
print("[PASS] 1b: Convex mirror diverging paraxial EFL = -50.0 mm")

# 1c. Spherical dielectric interface (Air -> Glass, R = 25mm, n1=1.0, n2=1.5)
m_refr = refraction_matrix(25.0, 1.0, 1.5)
y_rf, u_rf = (space_matrix(75.0) @ m_refr).propagate_ray(1.0, 0.0)
assert abs(y_rf) < 1e-12, f"Refraction focus failed: y={y_rf}"
print("[PASS] 1c: Spherical dielectric refraction focus at d = 75.0 mm")

# 1d. Resonator stability (Confocal cavity L = 100mm, R1 = -100mm, R2 = -100mm)
m_m1 = reflection_matrix(-100.0)
m_m2 = reflection_matrix(-100.0)
m_L = space_matrix(100.0)
m_roundtrip = m_m1 @ m_L @ m_m2 @ m_L
m_param = m_roundtrip.stability_parameter()
assert abs(m_param - (-1.0)) < 1e-9, f"Confocal stability parameter failed: {m_param}"
assert m_roundtrip.is_stable_resonator()
print(f"[PASS] 1d: Confocal cavity roundtrip stability m = {m_param:.4f}")

# 1e. Cavity Q-factor high reflectivity
q_high = calculate_cavity_q_factor(100.0, 1.064, r1=0.999, r2=0.999)
q_low = calculate_cavity_q_factor(100.0, 1.064, r1=0.9, r2=0.9)
assert q_high["q_factor"] > q_low["q_factor"] * 10.0
assert q_high["finesse"] > q_low["finesse"] * 10.0
print(f"[PASS] 1e: Cavity Q-factor scaling: Q(99.9%)={q_high['q_factor']:.2e} vs Q(90%)={q_low['q_factor']:.2e}")

# Test 2: AI Command Interface Natural Language Edge Cases
from optics_app.ai.tools import interpret_and_execute_command, lookup_thorlabs_catalog
from optics_app.engine.models import SpectrometerSystem
from optics_app.ai.generative_design import GenerativeOpticalDesigner

sys_test = GenerativeOpticalDesigner.design_spectrometer(0.785, 600.0, 1.0)

# 2a. Uppercase commands
res_up = interpret_and_execute_command("ROTATE ELEMENT 1 BY 3.2 DEGREES AROUND X AXIS", system=sys_test)
assert res_up["success"] is True
assert abs(sys_test.surfaces[1].tilt_x - 3.2) < 1e-4
print("[PASS] 2a: Uppercase natural language command execution")

# 2b. Mixed punctuation and prepositions
res_mix = interpret_and_execute_command("please tilt element #1 around y by -2.5 degrees", system=sys_test)
assert res_mix["success"] is True
assert abs(sys_test.surfaces[1].tilt_y - (-2.5)) < 1e-4
print("[PASS] 2b: Mixed punctuation / preposition phrasing execution")

# 2c. Unknown command handling
res_unk = interpret_and_execute_command("synthesize quantum teleportation protocol", system=sys_test)
assert res_unk["success"] is False
print("[PASS] 2c: Unknown command handled gracefully without crash")

# Test 3: Generative Optical Designer Edge Cases
# 3a. Extreme magnification (0.25x and 4x)
sys_025 = GenerativeOpticalDesigner.design_spectrometer(0.532, 1200.0, 0.25)
assert abs(sys_025.surfaces[1].radius - (-100.0)) < 1e-5
assert abs(sys_025.surfaces[3].radius - (-25.0)) < 1e-5
print("[PASS] 3a: Generative design 0.25x magnification parameter scaling")

# 3b. UV wavelength parsing
spec_uv = GenerativeOpticalDesigner.parse_specifications("Build spectrometer for 266 nm laser with 1200 grooves/mm")
assert abs(spec_uv["wavelength_um"] - 0.266) < 1e-5
assert spec_uv["grating_lines_per_mm"] == 1200.0
print("[PASS] 3b: Generative design UV wavelength extraction (266nm -> 0.266um)")

# Test 4: Real-world Catalog Query & Zemax File
from optics_app.ai.catalog import OpticalCatalogManager, COMMERCIAL_CATALOG
for part in ["AC254-050-C", "CM254-050-P01", "GR25-0616", "A230-C"]:
    spec = OpticalCatalogManager.query_catalog(part)
    assert spec is not None
    assert "type" in spec
print(f"[PASS] 4a: Verified all catalog parts in database")

# DMD4500 lookup via lookup_thorlabs_catalog
res_dmd = lookup_thorlabs_catalog("DMD4500")
assert "DMD4500" in res_dmd
print("[PASS] 4b: DMD4500 verified in lookup_thorlabs_catalog")

# Nonexistent catalog part raises KeyError
try:
    OpticalCatalogManager.query_catalog("NONEXISTENT-PART-XYZ")
    assert False, "Expected KeyError"
except KeyError:
    print("[PASS] 4c: Nonexistent catalog part raises expected KeyError")

# Real Zemax file parsing
zmx_file = "thorlabs_czerny_turner_785nm.zmx"
surfs = OpticalCatalogManager.parse_zemax_file(zmx_file)
assert len(surfs) >= 4
print(f"[PASS] 4d: Real ZEMAX file {zmx_file} parsed ({len(surfs)} optical surfaces)")

print("=== ALL ADVERSARIAL STRESS TESTS PASSED 100% ===")
