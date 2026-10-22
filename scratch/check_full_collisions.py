import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray
from scratch.sweep_relay import test_hybrid_relay

sys_opt = test_hybrid_relay(d_dmd_relay=50.0, d_relay_cond=25.0, d_cond_det=5.5, use_30mm_focus=True)

# Let's inspect all surfaces and their positions
print("Surfaces and Global Origins:")
for i, s in enumerate(sys_opt.surfaces):
    pos = s.global_origin
    print(f"{i:2d}: {s.name:22s} | type={s.surface_type} | pos=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] | sd={s.semi_diameter:5.2f}")

# Group surfaces into logical physical components
# Elements:
# 1. Collimator (surfaces 1, 2, 3) - Housing radius 12.7 mm
# 2. Aperture Stop (surface 4) - Stop
# 3. Grating (surface 6) - Housing radius 12.7 mm
# 4. Focusing Lens (surfaces 8, 9, 10) - Housing radius 12.7 mm
# 5. DMD4500 (surface 11) - DMD radius 12.7 mm / active 5.0 mm
# 6. Relay Lens (surfaces 13, 14, 15) - Housing radius 12.7 mm
# 7. Condenser Lens (surfaces 16, 17) - Housing radius 6.35 mm
# 8. Detector (surface 18) - Detector

components = {
    'Collimator': {'surfs': [1, 2, 3], 'radius': 12.7},
    'Grating': {'surfs': [6], 'radius': 12.7},
    'Focusing': {'surfs': [8, 9, 10], 'radius': 12.7},
    'DMD4500': {'surfs': [11], 'radius': 12.7},
    'Relay': {'surfs': [13, 14, 15], 'radius': 12.7},
    'Condenser': {'surfs': [16, 17], 'radius': 6.35},
}

for comp_name, comp_info in components.items():
    centers = [sys_opt.surfaces[idx].global_origin for idx in comp_info['surfs']]
    comp_info['center'] = np.mean(centers, axis=0)
    print(f"Component {comp_name:12s}: center = {np.round(comp_info['center'], 2)}, radius = {comp_info['radius']} mm")

# Trace full ray grid
wls = [1.087, 1.225, 1.350]
rays = []
for wl in wls:
    # Chief ray
    rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([0., 0., 1.]), wavelength=wl))
    # Marginal rays
    theta_max = np.arcsin(0.04)
    for theta in [theta_max/2, theta_max]:
        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = np.sin(theta)*np.cos(phi)
            dy = np.sin(theta)*np.sin(phi)
            dz = np.cos(theta)
            rays.append(Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl))

print(f"\nTracing {len(rays)} rays...")
traced = [sys_opt.trace_ray(r) for r in rays]

# Check clearance from each ray segment to every unrelated component center
min_clearance = 999.0
min_margin = 999.0
violations = []

for r_idx, tr in enumerate(traced):
    if tr.vignetted:
        print(f"Ray {r_idx} vignetted!")
        continue
    for seg_idx in range(len(tr.history) - 1):
        h1 = tr.history[seg_idx]
        h2 = tr.history[seg_idx+1]
        p1 = np.array([h1['x_global'], h1['y_global'], h1['z_global']])
        p2 = np.array([h2['x_global'], h2['y_global'], h2['z_global']])
        s1_idx = h1['surface_index']
        s2_idx = h2['surface_index']
        
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6: continue
        seg_dir = seg_vec / seg_len
        
        for comp_name, comp_info in components.items():
            # If the segment starts or ends at any surface of this component, skip
            # Also if it touches adjacent coordinate break of this component
            comp_surfs = set(comp_info['surfs'])
            if comp_name == 'Grating':
                comp_surfs.update([5, 7]) # CB before and after grating
            elif comp_name == 'DMD4500':
                comp_surfs.update([12]) # CB after DMD
                
            if s1_idx in comp_surfs or s2_idx in comp_surfs:
                continue
                
            comp_center = comp_info['center']
            req_dist = comp_info['radius'] + 5.0 # +5mm margin
            
            # Closest point on segment
            v = comp_center - p1
            t = np.clip(np.dot(v, seg_dir), 0.0, seg_len)
            closest_pt = p1 + t * seg_dir
            dist = np.linalg.norm(comp_center - closest_pt)
            clearance_to_center = dist # Distance to housing center
            margin_clearance = dist - req_dist # Clearance beyond radius + 5mm
            
            if clearance_to_center < min_clearance:
                min_clearance = clearance_to_center
            if margin_clearance < min_margin:
                min_margin = margin_clearance
                
            if margin_clearance < 0:
                violations.append((r_idx, seg_idx, s1_idx, s2_idx, comp_name, dist, req_dist, margin_clearance))

print(f"\nCollision Analysis Results:")
print(f"Minimum distance to any unrelated housing center: {min_clearance:.2f} mm (must be > 0)")
print(f"Minimum clearance beyond physical radius + 5mm:   {min_margin:.2f} mm (must be > 0)")

if violations:
    print(f"Violations ({len(violations)}):")
    for v in violations[:10]:
        s1_name = sys_opt.surfaces[v[2]].name
        s2_name = sys_opt.surfaces[v[3]].name
        print(f"  Ray {v[0]} Seg {s1_name}->{s2_name} vs {v[4]}: dist={v[5]:.2f} mm, req={v[6]:.2f} mm, diff={v[7]:.2f} mm")
else:
    print("SUCCESS: Zero collisions! All ray segments clear unrelated optical housings by > 5 mm margin.")
