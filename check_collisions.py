
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

sys_opt = OpticalSystem.from_json('spectrometer_refractive_1064.json')
sys_opt._update_accumulated_positions()

def dist(s1, s2):
    return np.linalg.norm(s1.global_origin - s2.global_origin)

for i in range(len(sys_opt.surfaces)):
    for j in range(i+1, len(sys_opt.surfaces)):
        d = dist(sys_opt.surfaces[i], sys_opt.surfaces[j])
        name1, name2 = sys_opt.surfaces[i].name, sys_opt.surfaces[j].name
        if 'CB_' in name1 or 'CB_' in name2 or 'Slit' in name1 or 'Slit' in name2 or 'Detector' in name1 or 'Detector' in name2: continue
        # Ignore adjacent surfaces of the same lens (thickness < 10)
        if 'Collimator' in name1 and 'Collimator' in name2: continue
        if 'Focusing' in name1 and 'Focusing' in name2: continue
        if 'Relay' in name1 and 'Relay' in name2: continue
        if 'Cond' in name1 and 'Cond' in name2: continue
        
        # Check if they are colliding
        sd1 = sys_opt.surfaces[i].semi_diameter
        sd2 = sys_opt.surfaces[j].semi_diameter
        if d < (sd1 + sd2 + 5.0): # 5mm housing tolerance
            print(f'Collision! {name1} and {name2} are only {d:.1f} mm apart (need {sd1+sd2+5.0:.1f} mm)')

