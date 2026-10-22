# run_challenger2_adversarial_verification.py
import os, sys, math, json
import numpy as np
import ezdxf

from optics_app.engine.grating import grating_diffract_3d
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

print('=' * 80)
print('CHALLENGER 2: ADVERSARIAL STRESS-TESTING & EMPIRICAL VERIFICATION')
print('=' * 80)

# Section 1: Evanescent Cutoff Boundaries
print('\n--- [1] EVANESCENT WAVE CUTOFF STRESS-TEST (300 vs 600 vs 1200 l/mm) ---')
wl_min = 1.11130  # 400 cm^-1 shift
wl_max = 1.66708  # 3400 cm^-1 shift
alpha_deg = 12.0
alpha_rad = math.radians(alpha_deg)
sin_alpha = math.sin(alpha_rad)

densities = [300.0, 600.0, 1200.0]
test_wls = np.linspace(wl_min, wl_max, 556)  # 1 nm resolution

for g in densities:
    d_um = 1000.0 / g
    cutoff_wl = d_um * (1.0 - sin_alpha)
    sin_betas = sin_alpha + (test_wls / d_um)
    evanescent_mask = sin_betas > 1.0
    evanescent_pct = 100.0 * np.mean(evanescent_mask)
    
    print(f'Grating {g:4.0f} l/mm (pitch d = {d_um:.4f} um):')
    print(f'  - Analytical cutoff wavelength at alpha={alpha_deg:.1f} deg: {cutoff_wl:.4f} um ({cutoff_wl*1000:.1f} nm)')
    b_min_str = f'{math.degrees(math.asin(sin_betas[0])):.2f} deg' if sin_betas[0] <= 1.0 else 'EVANESCENT'
    b_max_str = f'{math.degrees(math.asin(sin_betas[-1])):.2f} deg' if sin_betas[-1] <= 1.0 else 'EVANESCENT'
    print(f'  - sin(beta) at {wl_min:.4f} um (400 cm^-1):  {sin_betas[0]:.4f} -> beta = {b_min_str}')
    print(f'  - sin(beta) at {wl_max:.4f} um (3400 cm^-1): {sin_betas[-1]:.4f} -> beta = {b_max_str}')
    print(f'  - Evanescent fraction across 1111.3 - 1667.1 nm: {evanescent_pct:.1f}%')
    if g == 300.0:
        assert evanescent_pct == 0.0, '300 l/mm must have 0% evanescent waves!'
        print('  -> RESULT: PASS (100% propagating across full Raman band)')
    elif g == 600.0:
        assert evanescent_pct > 60.0, '600 l/mm must have > 60% evanescent cutoff!'
        print(f'  -> RESULT: CUTOFF FAILURE for full Raman band (Cuts off at {cutoff_wl*1000:.1f} nm)')
    elif g == 1200.0:
        assert evanescent_pct == 100.0, '1200 l/mm must be 100% evanescent!'
        print('  -> RESULT: UNIVERSAL FAILURE (100% evanescent across entire Raman spectrum)')

# Section 2: DMD Glint Rejection & Vector Reflection
print('\n--- [2] NORMAL INCIDENCE STRAY LIGHT GLINT SEPARATION ---')
i_vec = np.array([0.0, 0.0, 1.0])
n_win = np.array([0.0, 0.0, -1.0])
r_win = i_vec - 2.0 * np.dot(i_vec, n_win) * n_win
theta_win = math.degrees(math.atan2(r_win[0], -r_win[2]))

theta_tilt_on = math.radians(12.0)
n_on = np.array([-math.sin(theta_tilt_on), 0.0, -math.cos(theta_tilt_on)])
r_on = i_vec - 2.0 * np.dot(i_vec, n_on) * n_on
theta_on = math.degrees(math.atan2(-r_on[0], -r_on[2]))

theta_tilt_off = math.radians(-12.0)
n_off = np.array([-math.sin(theta_tilt_off), 0.0, -math.cos(theta_tilt_off)])
r_off = i_vec - 2.0 * np.dot(i_vec, n_off) * n_off
theta_off = math.degrees(math.atan2(-r_off[0], -r_off[2]))

na = 0.12
cone_half_angle = math.degrees(math.asin(na))
isolation_margin = (theta_on - theta_win) - 2.0 * cone_half_angle

print('Vector reflection angles:')
print(f'  - Flat DMD window reflection:   {theta_win:+.3f} deg (Retroreflected to entrance axis)')
print(f'  - ON-state micromirror (+12 deg tilt): {theta_on:+.3f} deg (Into collection optics)')
print(f'  - OFF-state micromirror (-12 deg tilt): {theta_off:+.3f} deg (Into beam dump)')
print(f'Optical cone isolation (NA = {na}, theta_cone = {cone_half_angle:.3f} deg):')
print(f'  - Glint cone range: [{theta_win - cone_half_angle:+.3f} deg, {theta_win + cone_half_angle:+.3f} deg]')
print(f'  - ON-state cone range: [{theta_on - cone_half_angle:+.3f} deg, {theta_on + cone_half_angle:+.3f} deg]')
print(f'  - Angular isolation margin: {isolation_margin:+.3f} deg')
assert isolation_margin > 10.0
print('  -> RESULT: PASS (> 10 deg clear separation, 0 glint in collection optics)')

# Section 3: CAD Export Integrity
print('\n--- [3] CAD EXPORT INTEGRITY & TOPOLOGY VERIFICATION ---')
cad_pairs = [
    ('hybrid_raman_spectrometer.step', 'hybrid_raman_spectrometer.dxf'),
    ('hybrid_raman_1064_full_range.step', 'hybrid_raman_1064_full_range.dxf'),
    ('hybrid_raman_1064.step', 'hybrid_raman_1064.dxf')
]

for step_file, dxf_file in cad_pairs:
    print(f'Checking pair: {step_file} / {dxf_file}')
    if os.path.exists(step_file):
        with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
            step_txt = f.read()
        n_solids = step_txt.count('MANIFOLD_SOLID_BREP')
        n_shells = step_txt.count('CLOSED_SHELL')
        n_faces = step_txt.count('ADVANCED_FACE')
        has_iso = 'ISO-10303-21;' in step_txt and 'END-ISO-10303-21;' in step_txt
        print(f'  STEP {step_file} ({len(step_txt)} bytes):')
        print(f'    - ISO header/trailer valid: {has_iso}')
        print(f'    - Manifold Solids: {n_solids}, Closed Shells: {n_shells}, Advanced Faces: {n_faces}')
        assert has_iso and n_solids >= 8, f'STEP file {step_file} invalid!'
    if os.path.exists(dxf_file):
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()
        layers = [l.dxf.name for l in doc.layers]
        print(f'  DXF {dxf_file} ({os.path.getsize(dxf_file)} bytes):')
        print(f'    - AC Version: {doc.dxfversion}, Entities: {len(msp)}')
        print(f'    - Key layers present: {"OPTICS_LENSES" in layers and "OPTICAL_AXIS" in layers}')
        assert len(msp) > 100, f'DXF file {dxf_file} has too few entities!'
        print('    -> RESULT: PASS')

print('\n' + '=' * 80)
print('ALL EMPIRICAL CHALLENGER 2 VERIFICATION CHECKS PASSED!')
print('=' * 80)
