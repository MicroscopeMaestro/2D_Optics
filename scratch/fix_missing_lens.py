import json
import os
import copy

def main():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'fiber_spectrometer_portable.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    surfaces = data["surfaces"]
    
    # Let's find the DMD Mirror index
    dmd_idx = -1
    for i, s in enumerate(surfaces):
        if s["name"] == "DMD_Mirror":
            dmd_idx = i
            break
            
    if dmd_idx == -1:
        print("DMD_Mirror not found!")
        return

    # Keep everything up to DMD_Mirror
    new_surfaces = surfaces[:dmd_idx+1]
    
    # We want to add L3 (AC127-019-A)
    l1_front = next(s for s in surfaces if s["name"] == "L1 Front")
    l1_mid = next(s for s in surfaces if s["name"] == "L1 Middle")
    l1_back = next(s for s in surfaces if s["name"] == "L1 Back")
    
    l3_front = copy.deepcopy(l1_front)
    l3_front["name"] = "L3 Front"
    l3_front["thickness"] = l1_front["thickness"]
    
    l3_mid = copy.deepcopy(l1_mid)
    l3_mid["name"] = "L3 Middle"
    l3_mid["thickness"] = l1_mid["thickness"]
    
    l3_back = copy.deepcopy(l1_back)
    l3_back["name"] = "L3 Back"
    l3_back["thickness"] = 10.0 # Initial air gap to Condenser
    
    new_surfaces.extend([l3_front, l3_mid, l3_back])
    
    # Condenser (ACL12708U: r=4.16, d=4.0, N-BK7, conic=-0.6265, asphere=1.5e-5)
    condenser_front = copy.deepcopy(l3_front)
    condenser_front["name"] = "Condenser Front"
    condenser_front["surface_type"] = "Aspheric"
    condenser_front["radius"] = 4.16
    condenser_front["thickness"] = 4.0
    condenser_front["glass"] = "N-BK7"
    condenser_front["conic"] = -0.6265
    condenser_front["asphere_coeffs"] = [1.5e-05]
    
    condenser_back = copy.deepcopy(l3_back)
    condenser_back["name"] = "Condenser Back"
    condenser_back["surface_type"] = "Standard"
    condenser_back["radius"] = 0.0
    condenser_back["thickness"] = 8.0 # Initial air gap to detector
    
    new_surfaces.extend([condenser_front, condenser_back])
    
    # Detector
    det = next(s for s in surfaces if s["name"] == "Detector")
    det_new = copy.deepcopy(det)
    new_surfaces.append(det_new)
    
    data["surfaces"] = new_surfaces
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Successfully inserted L3 AC127-019-A and combined Condenser ACL12708U!")

if __name__ == "__main__":
    main()
