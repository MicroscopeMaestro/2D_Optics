import re

with open('optics_app/gui/visualizer_3d.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add build_cylinder_geometry if it doesn't exist
cylinder_func = '''
def build_cylinder_geometry(radius: float, thickness: float, radial_samples: int = 12, angular_samples: int = 32):
    import numpy as np
    theta = np.linspace(0, 2 * np.pi, angular_samples)
    r = np.linspace(0, radius, radial_samples)
    T, R = np.meshgrid(theta, r)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    
    # Front face (z=0)
    Z_front = np.zeros_like(X)
    # Rear face (z=thickness)
    Z_rear = np.full_like(X, thickness)
    
    # Side cylinder
    Z_side = np.array([np.zeros_like(theta), np.full_like(theta, thickness)])
    X_side = np.array([radius * np.cos(theta), radius * np.cos(theta)])
    Y_side = np.array([radius * np.sin(theta), radius * np.sin(theta)])
    
    return (X, Y, Z_front), (X, Y, Z_rear), (X_side, Y_side, Z_side)
'''

if 'def build_cylinder_geometry' not in content:
    content = content.replace('def build_lens_solid_geometry(', cylinder_func + '\n\ndef build_lens_solid_geometry(')

# 2. Modify is_air_region rendering
air_replace_target = '''                if is_air_region:
                    color = "#b0bec5"
                    rim_color = "#90a4ae"
                    glass_alpha = 0.15 if not self.no_transparency else 1.0
                    edge_width = 0.0
                    edge_color = 'none'
'''

new_air_logic = '''                if is_air_region:
                    color = "#4fc3f7"  # Light transparent blue for air gaps
                    rim_color = "#29b6f6"
                    glass_alpha = 0.10 if not self.no_transparency else 0.8
                    edge_width = 0.5
                    edge_color = "#0288d1"
                    
                    # Force pure cylinder for air gap instead of sagging
                    front_mesh, rear_mesh, side_mesh = build_cylinder_geometry(
                        radius=max(getattr(front, 'semi_diameter', 10.0), getattr(rear, 'semi_diameter', 10.0)),
                        thickness=float(getattr(front, 'thickness', 10.0)),
                        radial_samples=mesh_radial, angular_samples=mesh_angular
                    )
'''

if 'Force pure cylinder' not in content:
    content = content.replace(air_replace_target, new_air_logic)

with open('optics_app/gui/visualizer_3d.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated visualizer_3d.py with Air Gap Cylinders.")
