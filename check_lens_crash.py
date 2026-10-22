
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

sys_opt = OpticalSystem.from_json('teamwork_yz_spectrometer/spectrometer_yz_1064_opt.json')
sys_opt._update_accumulated_positions()

# Find the centers of the Focusing Lens and Relay Lens
focus_c = sys_opt.surfaces[15].global_origin # Focusing_S3
relay_c = sys_opt.surfaces[18].global_origin # Relay_S1

dist = np.linalg.norm(focus_c - relay_c)
print(f'Distance between Focusing Lens and Relay Lens centers: {dist:.2f} mm')

# The lenses are 1-inch (25.4 mm) diameter.
# So their radii are 12.7 mm.
# To avoid touching, the distance must be > 25.4 mm.

