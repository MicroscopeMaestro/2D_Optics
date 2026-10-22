
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.tracer import OpticalSystem
from v2.engine.ray import Ray

# Load the optimized 40mm model
data = json.load(open('spectrometer_yz_40mm_100mm_optimized.json'))

# Update wavelengths
data['wavelengths'] = [1.1, 1.225, 1.35]
data['primary_wavelength'] = 1.225
data['system_config']['wavelengths'] = [
    {'wavelength_um': 1.1, 'weight': 1.0, 'is_primary': False, 'color': '#0000FF'},
    {'wavelength_um': 1.225, 'weight': 1.0, 'is_primary': True, 'color': '#00FF00'},
    {'wavelength_um': 1.35, 'weight': 1.0, 'is_primary': False, 'color': '#FF0000'}
]

# Save updated model
json.dump(data, open('spectrometer_yz_40mm_100mm_optimized.json', 'w'), indent=4)

