import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def get_element_name(surf_name):
    if 'CB_' in surf_name:
        # Extract associated element
        if 'Grating' in surf_name: return 'Grating'
        if 'DMD' in surf_name: return 'DMD'
    if 'Slit' in surf_name: return 'Slit'
    if 'Collimator' in surf_name: return 'Collimator'
    if 'Aperture' in surf_name or 'Stop' in surf_name: return 'Aperture_Stop'
    if 'Grating' in surf_name: return 'Grating'
    if 'Focusing' in surf_name: return 'Focusing'
    if 'DMD' in surf_name: return 'DMD'
    if 'Relay' in surf_name: return 'Relay'
    if 'Cond' in surf_name: return 'Condenser'
    if 'Detector' in surf_name: return 'Detector'
    return surf_name

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

# Group surfaces by optical element
elements = {}
for i, s in enumerate(sys_opt.surfaces):
    elem = get_element_name(s.name)
    if elem not in elements:
        elements[elem] = []
    elements[elem].append((i, s))

print("Optical Elements:")
for elem, surfs in elements.items():
    origins = [s.global_origin for idx, s in surfs]
    center = np.mean(origins, axis=0)
    max_sd = max(s.semi_diameter for idx, s in surfs)
    print(f"  {elem:15s}: center={np.round(center, 2)}, max_sd={max_sd}")

# Trace rays
wls = [1.087, 1.225, 1.350]
rays = []
for wl in wls:
    # Chief ray
    rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
    # Marginal rays
    theta_max = np.arcsin(0.04)
    for theta in [theta_max]:
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

print(f"\nTracing {len(rays)} rays...")
traced = [sys_opt.trace_ray(r) for r in rays]

print("\n--- Clearance between Ray Segments and Unrelated Elements ---")
min_clearance_all = 999.0

for r_idx, tr in enumerate(traced):
    for seg_idx in range(len(tr.history) - 1):
        h1 = tr.history[seg_idx]
        h2 = tr.history[seg_idx+1]
        p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
        p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
        s1 = sys_opt.surfaces[h1['surface_index']]
        s2 = sys_opt.surfaces[h2['surface_index']]
        elem1 = get_element_name(s1.name)
        elem2 = get_element_name(s2.name)
        
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6: continue
        seg_dir = seg_vec / seg_len
        
        for elem_name, surfs in elements.items():
            # If element is one of the endpoints of the segment, skip
            if elem_name == elem1 or elem_name == elem2:
                continue
            # Also skip virtual elements (Slit, Aperture_Stop, Detector) if appropriate, or check physical ones
            if elem_name in ('Slit', 'Detector'):
                continue
            
            # Element center and physical semi_diameter
            elem_center = np.mean([s.global_origin for idx, s in surfs], axis=0)
            elem_sd = max(s.semi_diameter for idx, s in surfs)
            required_clearance = elem_sd + 5.0 # 5 mm housing margin
            
            # Closest approach on line segment
            v = elem_center - p1
            t = np.clip(np.dot(v, seg_dir), 0.0, seg_len)
            closest_pt = p1 + t * seg_dir
            dist = np.linalg.norm(elem_center - closest_pt)
            clearance = dist - required_clearance
            
            if clearance < min_clearance_all:
                min_clearance_all = clearance
                
            if clearance < 0:
                print(f"COLLISION: Ray {r_idx} Seg {elem1}->{elem2} passes near {elem_name}: dist={dist:.2f} mm (required {required_clearance:.2f} mm, diff={clearance:.2f} mm)")

print(f"\nMinimum ray-to-unrelated-housing clearance margin: {min_clearance_all:.2f} mm")
