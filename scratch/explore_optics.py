import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from verify_layout import RAMAN_WAVELENGTHS_UM, generate_fiber_phase_space_rays

# Catalog definitions of AC254 doublets (Thorlabs standard):
DOUBLETS = {
    "AC254-030-C": [
        {"radius": 18.5, "thickness": 7.5, "glass": "N-BK7", "semi_diameter": 12.7},
        {"radius": -14.8, "thickness": 2.0, "glass": "SF5", "semi_diameter": 12.7},
        {"radius": -57.2, "thickness": 10.0, "glass": "AIR", "semi_diameter": 12.7}
    ],
    "AC254-040-C": [
        {"radius": 25.1, "thickness": 6.8, "glass": "N-BK7", "semi_diameter": 12.7},
        {"radius": -20.2, "thickness": 2.0, "glass": "SF5", "semi_diameter": 12.7},
        {"radius": -84.0, "thickness": 10.0, "glass": "AIR", "semi_diameter": 12.7}
    ],
    "AC254-050-C": [
        {"radius": 34.0, "thickness": 6.0, "glass": "N-BK7", "semi_diameter": 12.7},
        {"radius": -27.8, "thickness": 2.0, "glass": "SF5", "semi_diameter": 12.7},
        {"radius": -115.0, "thickness": 10.0, "glass": "AIR", "semi_diameter": 12.7}
    ],
    "AC254-075-C": [
        {"radius": 49.3, "thickness": 5.2, "glass": "N-BK7", "semi_diameter": 12.7},
        {"radius": -39.0, "thickness": 2.0, "glass": "SF5", "semi_diameter": 12.7},
        {"radius": -170.0, "thickness": 10.0, "glass": "AIR", "semi_diameter": 12.7}
    ],
    "AC254-100-C": [
        {"radius": 61.2, "thickness": 5.0, "glass": "N-BK7", "semi_diameter": 12.7},
        {"radius": -44.5, "thickness": 2.0, "glass": "SF5", "semi_diameter": 12.7},
        {"radius": -198.0, "thickness": 10.0, "glass": "AIR", "semi_diameter": 12.7}
    ]
}

# Condenser ACL12708U-C (Thorlabs Aspheric Condenser Lens, 0.5" / 12.7mm OD, EFL=8.0mm, NA=0.78):
CONDENSER = [
    {"radius": 4.16, "thickness": 4.5, "glass": "B270", "conic": -0.6265, "asphere_coeffs": [1.5e-5], "semi_diameter": 6.35},
    {"radius": 0.0, "thickness": 5.2, "glass": "AIR", "semi_diameter": 6.35}
]

print("Doublets and Condenser catalog ready.")
