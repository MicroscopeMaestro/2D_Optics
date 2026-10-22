import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from optics_app.engine.tracer import OpticalSystem

sys_old = OpticalSystem.from_json('spectrometer_yz_40mm_100mm_slit.json')
sys_new = OpticalSystem.from_json('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json')

def get_dist(sys_t, i, j):
    p1 = sys_t.surfaces[i].global_origin
    p2 = sys_t.surfaces[j].global_origin
    return np.linalg.norm(p1 - p2)

print("Distance Comparisons (Old vs New):")
# Front end
print(f"Sample to First Lens (S0 -> S1): {get_dist(sys_old, 0, 1):.2f} mm -> {get_dist(sys_new, 0, 1):.2f} mm")

# Focus to DMD
print(f"Focus Lens to DMD (S17 -> S18): {get_dist(sys_old, 17, 18):.2f} mm -> {get_dist(sys_new, 17, 18):.2f} mm")

# DMD to Relay
print(f"DMD to Relay Lens (S18 -> S20): {get_dist(sys_old, 18, 20):.2f} mm -> {get_dist(sys_new, 18, 20):.2f} mm")

# Relay to Condenser
print(f"Relay Lens to Condenser (S22 -> S23): {get_dist(sys_old, 22, 23):.2f} mm -> {get_dist(sys_new, 22, 23):.2f} mm")

# Condenser to Detector
print(f"Condenser to Detector (S24 -> S25): {get_dist(sys_old, 24, 25):.2f} mm -> {get_dist(sys_new, 24, 25):.2f} mm")
