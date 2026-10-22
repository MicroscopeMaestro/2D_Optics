import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.engine.surface import Surface, SurfaceType, AsphericSurface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.grating import grating_diffract_3d, DiffractionGrating
from optics_app.engine.source import SourceConfig, SourceType
from optics_app.analysis.spot import SpotDiagram, calculate_spot_metrics

results = {}

print('=' * 80)
print(' CHALLENGER 2: EMPIRICAL & ADVERSARIAL VERIFICATION SUITE')
print('=' * 80)

# ----------------------------------------------------------------------
# 1. GRATING PHYSICS & EVANESCENT WAVE CUTOFF BOUNDARIES
# ----------------------------------------------------------------------
print('\n[1] GRATING PHYSICS & EVANESCENT WAVE CUTOFF VERIFICATION')
raman_wls = [1.11130, 1.20000, 1.30000, 1.38919, 1.50000, 1.60000, 1.61330, 1.66708]
densities = [300.0, 600.0, 1200.0]
alpha_deg = 12.0
alpha_rad = math.radians(alpha_deg)
sin_alpha = math.sin(alpha_rad)

grating_results = {}
for g in densities:
    d_um = 1000.0 / g
    g_res = {'pitch_um': d_um, 'wavelengths': {}}
    for wl in raman_wls:
        sin_m1 = sin_alpha + (1.0 * wl / d_um)
        is_evanescent_analytical = abs(sin_m1) > 1.0
        
        v_in = np.array([math.sin(alpha_rad), 0.0, math.cos(alpha_rad)])
        N = np.array([0.0, 0.0, 1.0])
        L = np.array([1.0, 0.0, 0.0])
        v_out, is_evanescent_3d = grating_diffract_3d(v_in, N, L, g, 1, wl)
        
        diff_angle_deg = math.degrees(math.asin(sin_m1)) if not is_evanescent_analytical else None
        
        g_res['wavelengths'][wl] = {
            'sin_beta': sin_m1,
            'diff_angle_deg': diff_angle_deg,
            'evanescent_analytical': is_evanescent_analytical,
            'evanescent_3d_engine': is_evanescent_3d
        }
    grating_results[str(int(g)) + '_lpm'] = g_res

results['grating_cutoff_analysis'] = grating_results

g300_all_propagating = all(not v['evanescent_analytical'] for v in grating_results['300_lpm']['wavelengths'].values())
g600_1667_evanescent = grating_results['600_lpm']['wavelengths'][1.66708]['evanescent_analytical']
g1200_all_evanescent = all(v['evanescent_analytical'] for v in grating_results['1200_lpm']['wavelengths'].values())

print(f' -> 300 l/mm: All Raman wavelengths propagate (No evanescent waves)? {g300_all_propagating}')
print(f' -> 600 l/mm: 1667.1 nm undergoes evanescent cutoff (sin_beta={grating_results["600_lpm"]["wavelengths"][1.66708]["sin_beta"]:.4f} > 1)? {g600_1667_evanescent}')
print(f' -> 1200 l/mm: 100% of Raman band is evanescent (sin_beta > 1.33)? {g1200_all_evanescent}')

cutoff_300 = (1000.0 / 300.0) * (1.0 - sin_alpha)
cutoff_600 = (1000.0 / 600.0) * (1.0 - sin_alpha)
cutoff_1200 = (1000.0 / 1200.0) * (1.0 - sin_alpha)
print(f' -> Cutoff wavelength at alpha=12.0 deg:')
print(f'    - 300 l/mm:  lambda_max_prop = {cutoff_300:.4f} um (allows full Raman band up to 1.667 um)')
print(f'    - 600 l/mm:  lambda_max_prop = {cutoff_600:.4f} um (cuts off above 1320 nm)')
print(f'    - 1200 l/mm: lambda_max_prop = {cutoff_1200:.4f} um (cuts off above 660 nm)')

# ----------------------------------------------------------------------
# 2. QUASI-LITTROW CONDITION & ANGULAR DISPERSION MATCH
# ----------------------------------------------------------------------
print('\n[2] QUASI-LITTROW CONDITION & CAMERA FOCAL LENGTH MATCH')
g = 300.0
d_um = 1000.0 / g
wl_mid = 1.38919
sin_littrow = (1.0 * wl_mid) / (2.0 * d_um)
theta_littrow_deg = math.degrees(math.asin(sin_littrow))

wl_min = 1.11130
wl_max = 1.61330
sin_b_min = sin_alpha + (wl_min / d_um)
sin_b_max = sin_alpha + (wl_max / d_um)
beta_min_deg = math.degrees(math.asin(sin_b_min))
beta_max_deg = math.degrees(math.asin(sin_b_max))
delta_beta_rad = math.radians(beta_max_deg - beta_min_deg)

f_cam = 50.0
linear_dispersion_approx_mm = f_cam * delta_beta_rad

print(f' -> Central wavelength {wl_mid:.4f} um exact Littrow angle: {theta_littrow_deg:.3f} deg (Layout angle: {alpha_deg:.1f} deg)')
print(f' -> Angular spread across primary band [1.111 um, 1.613 um]: {math.degrees(delta_beta_rad):.3f} deg ({delta_beta_rad:.4f} rad)')
print(f' -> Approximate linear spread with f={f_cam}mm camera lens: {linear_dispersion_approx_mm:.3f} mm (Target DMD active width: 8.664 mm)')

# ----------------------------------------------------------------------
# 3. DMD WINDOW NORMAL INCIDENCE VS +24 DEG ON-STATE REFLECTION
# ----------------------------------------------------------------------
print('\n[3] DMD WINDOW NORMAL INCIDENCE & GLINT REJECTION')
dmd_tilt_deg = 12.0
dmd_tilt_rad = math.radians(dmd_tilt_deg)
window_refl_deg = 0.0
on_state_refl_deg = 2.0 * dmd_tilt_deg
off_state_refl_deg = -2.0 * dmd_tilt_deg

na = 0.12
cone_half_angle_deg = math.degrees(math.asin(na))
angular_isolation_margin_deg = on_state_refl_deg - (2.0 * cone_half_angle_deg)

print(f' -> DMD Window Glint Angle: {window_refl_deg:.1f} deg (Retroreflected to entrance axis)')
print(f' -> DMD ON-state Reflection Channel: +{on_state_refl_deg:.1f} deg')
print(f' -> DMD OFF-state Reflection Channel: {off_state_refl_deg:.1f} deg (Dumped)')
print(f' -> Beam Cone Half-Angle (NA={na}): +/-{cone_half_angle_deg:.2f} deg')
print(f' -> Glint-to-Signal Angular Isolation Margin: {angular_isolation_margin_deg:.2f} deg (> 0 confirms 100% glint rejection)')

# ----------------------------------------------------------------------
# 4. CAD B-REP SOLID TOPOLOGY & DXF AUDIT
# ----------------------------------------------------------------------
print('\n[4] CAD B-REP SOLID TOPOLOGY & DXF AUDIT')
for step_file in ['hybrid_raman_spectrometer.step', 'hybrid_raman_785nm_spectrometer.step']:
    if not os.path.exists(step_file):
        continue
    with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
        step_content = f.read()
    num_solids = step_content.count('MANIFOLD_SOLID_BREP')
    num_shells = step_content.count('CLOSED_SHELL')
    num_faces = step_content.count('ADVANCED_FACE')
    num_products = step_content.count('PRODUCT(')
    has_header = 'ISO-10303-21;' in step_content and 'HEADER;' in step_content
    print(f' -> STEP File: {step_file} ({len(step_content)} bytes)')
    print(f'    - Valid ISO Header & Schema? {has_header}')
    print(f'    - Solids: {num_solids}, Closed Shells: {num_shells}, Faces: {num_faces}, Products: {num_products}')

for dxf_file in ['hybrid_raman_spectrometer.dxf', 'hybrid_raman_785nm_spectrometer_dual_planes.dxf']:
    if not os.path.exists(dxf_file):
        continue
    with open(dxf_file, 'r', encoding='utf-8', errors='ignore') as f:
        dxf_content = f.read()
    has_header = 'SECTION' in dxf_content and 'HEADER' in dxf_content
    has_entities = 'ENTITIES' in dxf_content
    has_eof = dxf_content.strip().endswith('EOF')
    num_lines = dxf_content.count('LINE\n') + dxf_content.count('LINE\r\n')
    num_circles = dxf_content.count('CIRCLE\n') + dxf_content.count('CIRCLE\r\n')
    num_lwpolylines = dxf_content.count('LWPOLYLINE\n') + dxf_content.count('LWPOLYLINE\r\n')
    num_texts = dxf_content.count('TEXT\n') + dxf_content.count('TEXT\r\n')
    print(f' -> DXF File: {dxf_file} ({len(dxf_content)} bytes)')
    print(f'    - Valid DXF Structure & EOF? {has_header and has_entities and has_eof}')
    print(f'    - Entities: Lines={num_lines}, Polylines={num_lwpolylines}, Circles={num_circles}, Texts={num_texts}')

# ----------------------------------------------------------------------
# 5. FULL RAY TRACING AUDIT
# ----------------------------------------------------------------------
print('\n[5] FULL SYSTEM RAY-TRACING AUDIT')
for layout_name in ['hybrid_raman_spectrometer.json', 'hybrid_raman_1064_full_range.json', 'hybrid_raman_1064.json']:
    if not os.path.exists(layout_name):
        continue
    loaded = SurfaceStack.load_from_json(layout_name)
    stack = loaded['stack']
    opt_sys = OpticalSystem(stack)
    
    dmd_idx = -1
    det_idx = len(stack.surfaces) - 1
    for idx, s in enumerate(stack.surfaces):
        if 'dmd' in s.name.lower():
            dmd_idx = idx
            break
            
    print(f' -> Layout: {layout_name} (Surfaces: {len(stack.surfaces)}, DMD idx: {dmd_idx}, Det idx: {det_idx})')
    
    chief_rays_dmd_x = []
    chief_rays_det_r = []
    trace_summary = {}
    for wl in raman_wls:
        r = Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl)
        traced = opt_sys.trace_ray(r)
        
        dmd_entry = None
        det_entry = None
        for h in traced.history:
            if h.get('surface_index') == dmd_idx:
                dmd_entry = h
            elif h.get('surface_index') == det_idx:
                det_entry = h
                
        if dmd_entry is not None:
            chief_rays_dmd_x.append(dmd_entry.get('x_local', 0.0))
        if det_entry is not None:
            dx = det_entry.get('x_local', 0.0)
            dy = det_entry.get('y_local', 0.0)
            chief_rays_det_r.append(math.sqrt(dx**2 + dy**2))
            
        trace_summary[wl] = {
            'dmd_x_local': dmd_entry.get('x_local', 0.0) if dmd_entry else None,
            'det_r_local': math.sqrt(det_entry.get('x_local', 0.0)**2 + det_entry.get('y_local', 0.0)**2) if det_entry else None,
        }
    
    if len(chief_rays_dmd_x) >= 2:
        dmd_span_mm = abs(max(chief_rays_dmd_x) - min(chief_rays_dmd_x))
        primary_wls = [1.11130, 1.20000, 1.30000, 1.38919, 1.50000, 1.60000, 1.61330]
        prim_xs = [trace_summary[w]['dmd_x_local'] for w in primary_wls if trace_summary[w]['dmd_x_local'] is not None]
        prim_span_mm = abs(max(prim_xs) - min(prim_xs))
        max_det_spot_r = max(chief_rays_det_r) if chief_rays_det_r else 0.0
        
        print(f'    - Chief ray DMD primary span [1.111 um, 1.613 um]: {prim_span_mm:.3f} mm (DMD limit: 8.664 mm)')
        print(f'    - Chief ray DMD full span [1.111 um, 1.667 um]: {dmd_span_mm:.3f} mm')
        print(f'    - Chief ray max detector radial offset: {max_det_spot_r:.4f} mm (Limit: 0.500 mm / 1.0mm dia)')

print('=' * 80)
