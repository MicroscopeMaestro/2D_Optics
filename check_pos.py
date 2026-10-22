
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem

stack = SurfaceStack.load_from_json('C:/Users/Juan/Desktop/Spectrometer_V2_2inch_Optimized.json')
if isinstance(stack, dict): stack = stack['stack']
sys_opt = OpticalSystem(stack)

print('Surface Global Positions (Y, Z):')
for i, s in enumerate(sys_opt.surfaces):
    pos = s.global_origin
    print(f'{i:2d} {s.name:25s}: Y={pos[1]:8.2f} Z={pos[2]:8.2f} Radius={s.semi_diameter}')

