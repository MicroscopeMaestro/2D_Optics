import os
import sys
import math
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray

# Thorlabs C-coating doublets catalog
# Prescriptions: (R1, T1, Glass1, R2, T2, Glass2, R3, BFL)
THORLABS_CATALOG = {
    "AC254-030-C": {
        "f": 30.0, "R1": 20.00, "T1": 12.00, "G1": "N-BAF10", "R2": -17.40, "T2": 3.00, "G2": "N-SF6HT", "R3": -93.10, "BFL": 21.32,
        # Alt N-BK7/SF5: R1: 18.50, T1: 7.50, G1: "N-BK7", R2: -14.80, T2: 2.00, G2: "SF5", R3: -57.20
    },
    "AC254-030-C-BK7": {
        "f": 30.0, "R1": 18.50, "T1": 7.50, "G1": "N-BK7", "R2": -14.80, "T2": 2.00, "G2": "SF5", "R3": -57.20, "BFL": 24.80,
    },
    "AC254-040-C": {
        "f": 40.0, "R1": 23.70, "T1": 10.00, "G1": "N-LAK22", "R2": -20.10, "T2": 2.50, "G2": "N-SF6HT", "R3": -57.70, "BFL": 28.76,
    },
    "AC254-040-C-BK7": {
        "f": 40.0, "R1": 25.10, "T1": 6.80, "G1": "N-BK7", "R2": -20.20, "T2": 2.00, "G2": "SF5", "R3": -84.00, "BFL": 35.40,
    },
    "AC254-050-C": {
        "f": 50.0, "R1": 34.00, "T1": 6.00, "G1": "N-BK7", "R2": -27.80, "T2": 2.00, "G2": "SF5", "R3": -115.00, "BFL": 45.60,
    },
    "AC254-075-C": {
        "f": 75.0, "R1": 49.30, "T1": 8.00, "G1": "N-BK7", "R2": -39.00, "T2": 3.00, "G2": "SF5", "R3": -170.00, "BFL": 69.80,
    },
    "AC254-100-C": {
        "f": 100.0, "R1": 64.50, "T1": 7.00, "G1": "N-BK7", "R2": -50.20, "T2": 2.00, "G2": "SF5", "R3": -220.00, "BFL": 94.90,
    },
    "ACL12708U-C": {
        "f": 8.0, "R1": 4.16, "T1": 4.50, "G1": "B270", "conic": -0.6265, "asphere": [1.5e-5], "R2": 0.0, "BFL": 5.20
    }
}

def build_system_and_eval(condenser_type="doublet_pair", gap_dmd_relay=20.0, gap_relay_cond=5.0, gap_cond_det=10.0, **kwargs):
    stack = SurfaceStack()
    stack.primary_wavelength = 1.36200
    
    # 0. Slit
    stack.add_surface(Surface(name="Slit_Source", surface_type=SurfaceType.OBJECT, thickness=47.5, glass="AIR", semi_diameter=12.7))
    
    # 1-3. Collimator AC254-050-C
    c = THORLABS_CATALOG["AC254-050-C"]
    stack.add_surface(Surface(name="AC254-050-C_Coll_Front", radius=c["R1"], thickness=c["T1"], glass=c["G1"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Mid", radius=c["R2"], thickness=c["T2"], glass=c["G2"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Coll_Back", radius=c["R3"], thickness=20.0, glass="AIR", semi_diameter=12.7))
    
    # 4. Grating GR25-0316
    stack.add_surface(Surface(name="Grating_GR25-0316", surface_type=SurfaceType.GRATING, radius=0.0, thickness=20.0, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0, grating_lines_per_mm=300.0, grating_order=1))
    
    # 5-7. Camera AC254-050-C
    stack.add_surface(Surface(name="AC254-050-C_Focus_Front", radius=c["R1"], thickness=c["T1"], glass=c["G1"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Mid", radius=c["R2"], thickness=c["T2"], glass=c["G2"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-050-C_Focus_Back", radius=c["R3"], thickness=44.0, glass="AIR", semi_diameter=12.7))
    
    # 8. DMD4500
    stack.add_surface(Surface(name="TI_DMD4500", surface_type=SurfaceType.STANDARD, radius=0.0, thickness=gap_dmd_relay, glass="MIRROR", semi_diameter=12.7, tilt_y=12.0))
    
    # 9-11. Relay AC254-075-C
    r = THORLABS_CATALOG["AC254-075-C"]
    stack.add_surface(Surface(name="AC254-075-C_Relay_Front", radius=r["R1"], thickness=r["T1"], glass=r["G1"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Mid", radius=r["R2"], thickness=r["T2"], glass=r["G2"], semi_diameter=12.7))
    stack.add_surface(Surface(name="AC254-075-C_Relay_Back", radius=r["R3"], thickness=gap_relay_cond, glass="AIR", semi_diameter=12.7))
    
    if condenser_type == "asphere":
        asph = THORLABS_CATALOG["ACL12708U-C"]
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Front", surface_type=SurfaceType.ASPHERIC, radius=asph["R1"], thickness=asph["T1"], glass=asph["G1"], semi_diameter=12.7, conic=asph["conic"], asphere_coeffs=asph["asphere"]))
        stack.add_surface(Surface(name="ACL12708U-C_Cond_Back", radius=asph["R2"], thickness=gap_cond_det, glass="AIR", semi_diameter=12.7))
    elif condenser_type == "single_doublet_030":
        d30 = THORLABS_CATALOG["AC254-030-C-BK7"]
        stack.add_surface(Surface(name="AC254-030-C_Cond_Front", radius=d30["R1"], thickness=d30["T1"], glass=d30["G1"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Mid", radius=d30["R2"], thickness=d30["T2"], glass=d30["G2"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond_Back", radius=d30["R3"], thickness=gap_cond_det, glass="AIR", semi_diameter=12.7))
    elif condenser_type == "doublet_pair_030":
        d30 = THORLABS_CATALOG["AC254-030-C-BK7"]
        gap_mid = kwargs.get("gap_mid", 1.0)
        # Doublet 1
        stack.add_surface(Surface(name="AC254-030-C_Cond1_Front", radius=d30["R1"], thickness=d30["T1"], glass=d30["G1"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond1_Mid", radius=d30["R2"], thickness=d30["T2"], glass=d30["G2"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond1_Back", radius=d30["R3"], thickness=gap_mid, glass="AIR", semi_diameter=12.7))
        # Doublet 2
        stack.add_surface(Surface(name="AC254-030-C_Cond2_Front", radius=d30["R1"], thickness=d30["T1"], glass=d30["G1"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond2_Mid", radius=d30["R2"], thickness=d30["T2"], glass=d30["G2"], semi_diameter=12.7))
        stack.add_surface(Surface(name="AC254-030-C_Cond2_Back", radius=d30["R3"], thickness=gap_cond_det, glass="AIR", semi_diameter=12.7))

    # Detector
    stack.add_surface(Surface(name="Detector_InGaAs_1mm", surface_type=SurfaceType.DETECTOR, radius=0.0, thickness=0.0, glass="AIR", semi_diameter=0.50))
    
    stack._update_accumulated_positions()
    opt = OpticalSystem(stack)
    
    # Check Chief Ray span on DMD
    wls = [1.11130, 1.20000, 1.30000, 1.36200, 1.40000, 1.50000, 1.61330, 1.66708]
    chief_rays = [Ray(pos=np.array([0.0, 0.0, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=w) for w in wls]
    chief_traced = opt.trace_rays(chief_rays)
    dmd_idx = next(i for i, s in enumerate(stack.surfaces) if "DMD" in s.name.upper())
    det_idx = len(stack.surfaces) - 1
    
    dmd_hits = {}
    for r in chief_traced:
        hit = next(h for h in r.history if h["surface_index"] == dmd_idx)
        dmd_hits[r.wavelength] = hit["x"]
    span_1613 = abs(dmd_hits[1.61330] - dmd_hits[1.11130])
    
    # Check Detector Spot with f/5 (NA = 0.10) rays
    na = 0.10
    max_theta = math.asin(na)
    slit_r = 0.0125
    test_rays = []
    for wl in wls:
        for px in [-slit_r, 0.0, slit_r]:
            for py in [-slit_r, 0.0, slit_r]:
                for ar in [0.0, 0.5, 1.0]:
                    if ar == 0.0:
                        test_rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([0.0, 0.0, 1.0]), wavelength=wl))
                    else:
                        for phi in np.linspace(0, 2*np.pi, 8, endpoint=False):
                            theta = ar * max_theta
                            vx = math.sin(theta) * math.cos(phi)
                            vy = math.sin(theta) * math.sin(phi)
                            vz = math.cos(theta)
                            test_rays.append(Ray(pos=np.array([px, py, 0.0]), dir=np.array([vx, vy, vz]), wavelength=wl))
                            
    traced = opt.trace_rays(test_rays)
    det_hits = []
    for r in traced:
        if not r.vignetted:
            dh = next((h for h in r.history if h["surface_index"] == det_idx), None)
            if dh:
                det_hits.append((dh["x"], dh["y"]))
                
    if det_hits:
        radii = [math.hypot(x, y) for x, y in det_hits]
        max_r = max(radii)
        max_d = 2.0 * max_r
        rms_r = math.sqrt(np.mean(np.array(radii)**2))
        return {
            "dmd_span": span_1613,
            "max_r": max_r,
            "max_d": max_d,
            "rms_r": rms_r,
            "reached": len(det_hits),
            "total": len(test_rays),
            "stack": stack
        }
    return None

if __name__ == "__main__":
    for ctype in ["asphere", "single_doublet_030", "doublet_pair_030"]:
        for g_det in np.linspace(3.0, 15.0, 13):
            res = build_system_and_eval(condenser_type=ctype, gap_cond_det=g_det)
            if res:
                print(f"Type: {ctype:20s} | GapDet: {g_det:5.2f} | DMD Span: {res['dmd_span']:.3f} mm | Spot Max D: {res['max_d']:.4f} mm | Spot RMS: {res['rms_r']:.4f} mm | Reached: {res['reached']}/{res['total']}")
