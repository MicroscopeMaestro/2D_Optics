import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

def verify_spot_radius(json_path=None, verbose=True):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spectrometer_refractive_1064.json')
        
    sys_opt = OpticalSystem.from_json(json_path)
    sys_opt._update_accumulated_positions()
    
    wls = [1.087, 1.225, 1.350]
    all_hits_x, all_hits_y = [], []
    wl_hits = {wl: [] for wl in wls}
    vignettes = 0
    total_rays = 0
    
    # Trace ray bundle across Raman spectrum and numerical aperture cone
    for wl in wls:
        for theta in np.linspace(0.001, np.arcsin(0.04), 5):
            for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
                r = Ray(pos=np.array([0., 0., 0.]), dir=np.array([dx, dy, dz]), wavelength=wl)
                tr = sys_opt.trace_ray(r)
                total_rays += 1
                if tr.vignetted:
                    vignettes += 1
                else:
                    h = tr.history[-1]
                    all_hits_x.append(h['x_local'])
                    all_hits_y.append(h['y_local'])
                    wl_hits[wl].append((h['x_local'], h['y_local']))
                    
    if not all_hits_x:
        if verbose:
            print("ERROR: All rays were vignetted!")
        return False, 999.0, 0, total_rays

    per_wl_max_r = {}
    per_wl_rms_r = {}
    per_wl_centroids = {}
    
    for wl, pts in wl_hits.items():
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        r_sq = (xs - mx)**2 + (ys - my)**2
        per_wl_centroids[wl] = (mx, my)
        per_wl_max_r[wl] = float(np.max(np.sqrt(r_sq)))
        per_wl_rms_r[wl] = float(np.sqrt(np.mean(r_sq)))

    # Overall max spot radius from wavelength centroid
    max_spot_r = float(max(per_wl_max_r.values())) if per_wl_max_r else 999.0
    
    # Overall max radial distance from detector optical axis center (0,0)
    all_r_from_origin = np.sqrt(np.array(all_hits_x)**2 + np.array(all_hits_y)**2)
    max_r_origin = float(np.max(all_r_from_origin))
    rms_r_origin = float(np.sqrt(np.mean(all_r_from_origin**2)))
    
    # Require 100% throughput across all design wavelengths, max spot radius <= 0.400 mm,
    # and all rays landing within the 0.400 mm detector radius (0.8 mm detector diameter)
    passed = (max_spot_r <= 0.400) and (max_r_origin <= 0.400) and (vignettes == 0) and all(len(pts) > 0 for pts in wl_hits.values())
    
    if verbose:
        print(f"=== Spot Radius Verification ({os.path.basename(json_path)}) ===")
        print(f"  Total Traced Rays:   {total_rays}")
        print(f"  Non-Vignetted Rays:  {len(all_hits_x)}")
        print(f"  Vignetted Rays:      {vignettes}")
        print(f"  Throughput:          {100.0 * len(all_hits_x) / total_rays:.1f}%")
        print("  Per-Wavelength Breakdown:")
        for wl in wls:
            pts = wl_hits[wl]
            c = per_wl_centroids.get(wl, (0.0, 0.0))
            r_max = per_wl_max_r.get(wl, 999.0)
            r_rms = per_wl_rms_r.get(wl, 999.0)
            print(f"    {wl:.3f} µm: {len(pts)} rays | Centroid=({c[0]:.4f}, {c[1]:.4f}) mm | Max Spot R={r_max:.4f} mm | RMS Spot R={r_rms:.4f} mm")
        print(f"  Max Spot Radius (from centroid): {max_spot_r:.4f} mm (Threshold: <= 0.4000 mm)")
        print(f"  Max Radial Distance from (0,0):   {max_r_origin:.4f} mm (Detector Semi-Diameter: 0.4000 mm)")
        print(f"  Overall RMS Spot Radius:         {rms_r_origin:.4f} mm")
        print(f"  Verification Result: {'PASS' if passed else 'FAIL'}")
        
    return passed, max_r_origin, len(all_hits_x), total_rays

if __name__ == '__main__':
    passed, max_r, hits, total = verify_spot_radius()
    assert passed, f"Max spot radius {max_r:.4f} mm exceeds 0.400 mm limit or has vignetting!"
    sys.exit(0 if passed else 1)
