
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem

with open('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json', 'r') as f:
    data = json.load(f)

# Apply 5 degree board tilt.
# The DMD surface is index 16.
# It currently has tilt_x = 12.0
data['surfaces'][16]['tilt_x'] = 17.0

# The coordinate break AFTER the DMD is index 17.
# It previously had tilt_x = 0.0 (Wait, in debug it had tilt_x = 0).
# Let's see what the tilt was originally!
# If DMD is 12, the beam reflects at 24.

