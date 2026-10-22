
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath('.'))
from v2.engine.surface import Surface, SurfaceType
from v2.engine.lde import SurfaceStack
from v2.engine.tracer import OpticalSystem

t_coll = 25.0
t_foc = 25.0
t_rel1 = 30.0  # Fold mirror right after DMD
t_rel2 = 60.0  # Relay Lens far away out-of-plane
t_cond = 10.0

data = {
    'surfaces': [
        {'name': 'Input_Slit_30um', 'surface_type': 'Object', 'thickness': t_coll},
        {'name': 'Collimator_S1', 'surface_type': 'Standard', 'radius': 71.12, 'thickness': 1.8, 'glass': 'N-SF6HT', 'semi_diameter': 12.7},
        {'name': 'Collimator_S2', 'surface_type': 'Standard', 'radius': 15.24, 'thickness': 13.0, 'glass': 'N-BAF10', 'semi_diameter': 12.7},
        {'name': 'Collimator_S3', 'surface_type': 'Standard', 'radius': -21.09, 'thickness': 30.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        
        {'name': 'Grating_600lpmm', 'surface_type': 'Grating', 'thickness': 0.0, 'grating_lines_per_mm': 600.0, 'grating_order': 1, 'grating_primary_wavelength_um': 1.225, 'tilt_x': -21.5, 'semi_diameter': 12.7, 'glass': ''},
        {'name': 'CB_After_Grating', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'tilt_x': -43.1230625, 'thickness': 30.0},
        
        {'name': 'Focusing_S1', 'surface_type': 'Standard', 'radius': 21.09, 'thickness': 13.0, 'glass': 'N-BAF10', 'semi_diameter': 12.7},
        {'name': 'Focusing_S2', 'surface_type': 'Standard', 'radius': -15.24, 'thickness': 1.8, 'glass': 'N-SF6HT', 'semi_diameter': 12.7},
        {'name': 'Focusing_S3', 'surface_type': 'Standard', 'radius': -71.12, 'thickness': t_foc, 'glass': 'AIR', 'semi_diameter': 12.7},
        
        # Use DMD in its standard configuration (reflects beam 24 degrees)
        {'name': 'DMD4500_Micromirrors', 'surface_type': 'Standard', 'thickness': 0.0, 'tilt_x': -12.0, 'semi_diameter': 5.0, 'glass': 'MIRROR'},
        {'name': 'CB_After_DMD', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'tilt_x': 0.0, 'thickness': t_rel1},
        
        # Add a flat fold mirror at 45 degrees to bend the beam 90 degrees out of the way!
        # We tilt around Y so it goes sideways and avoids the Grating (which is below it in Y)
        {'name': 'Fold_Mirror', 'surface_type': 'Standard', 'thickness': 0.0, 'tilt_y': 45.0, 'semi_diameter': 12.7, 'glass': 'MIRROR'},
        {'name': 'CB_After_Fold', 'surface_type': 'Coordinate Break', 'is_coord_break': True, 'tilt_x': 0.0, 'thickness': t_rel2},
        
        {'name': 'Imaging_Relay_S1', 'surface_type': 'Standard', 'radius': 61.8, 'thickness': 7.5, 'glass': 'N-BK7', 'semi_diameter': 25.4},
        {'name': 'Imaging_Relay_S2', 'surface_type': 'Standard', 'radius': -49.2, 'thickness': 3.5, 'glass': 'SF5', 'semi_diameter': 25.4},
        {'name': 'Imaging_Relay_S3', 'surface_type': 'Standard', 'radius': -250.0, 'thickness': t_cond, 'glass': 'AIR', 'semi_diameter': 25.4},
        {'name': 'Imaging_Cond_S1', 'surface_type': 'Aspheric', 'radius': 8.0, 'conic': -1.0, 'thickness': 10.0, 'glass': 'B270', 'semi_diameter': 12.7},
        {'name': 'Imaging_Cond_S2', 'surface_type': 'Standard', 'radius': 0.0, 'thickness': 16.0, 'glass': 'AIR', 'semi_diameter': 12.7},
        {'name': 'Detector', 'surface_type': 'Detector', 'thickness': 0.0, 'semi_diameter': 0.4}
    ]
}

stack = SurfaceStack()
for sd in data['surfaces']:
    s = Surface(name=sd.get('name', ''))
    for k,v in sd.items():
        if k == 'surface_type':
            if 'grating' in v.lower(): s.surface_type = SurfaceType.GRATING
            elif 'object' in v.lower(): s.surface_type = SurfaceType.OBJECT
            elif 'detector' in v.lower(): s.surface_type = SurfaceType.DETECTOR
            elif 'aspheric' in v.lower(): 
                s.surface_type = SurfaceType.ASPHERIC
                s.is_aspheric = True
            elif 'coordinate' in v.lower():
                s.surface_type = SurfaceType.COORDINATE_BREAK
                s.is_coord_break = True
            else: s.surface_type = SurfaceType.STANDARD
        elif hasattr(s, k): setattr(s, k, v)
    stack.add_surface(s)

stack.save_to_json('spectrometer_refractive_1064.json', wavelengths=[1.087, 1.225, 1.350])
print('Aligned JSON constructed and saved successfully!')

