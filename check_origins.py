
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem
stack = SurfaceStack()
stack.load_from_json(r'C:\Users\Juan\.gemini\antigravity\brain\49fd158d-b0e1-421c-b15b-b237b2e3ad7f\Spectrometer_V2_Zemax.json')
sys_opt = OpticalSystem(stack)
for s in sys_opt.surfaces:
    print(s.name, s.global_origin)

