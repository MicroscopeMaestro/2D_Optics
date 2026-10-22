import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def build_rect_aperture_2d(hx, hy, sd, n_arc=16):
    # Ensure sd is large enough to contain the rectangle corners
    sd = max(sd, np.hypot(hx, hy) + 0.001)
    
    faces = []
    
    # Angles of the 4 intersections of the horizontal/vertical lines with the circle
    # Right edge y in [-hy, hy]
    ang_tr = np.arcsin(hy / sd)
    ang_br = np.arcsin(-hy / sd)
    
    # Top edge x in [-hx, hx]
    ang_tl = np.pi - np.arcsin(hy / sd)
    
    # Left edge y in [-hy, hy]
    ang_bl = np.pi - np.arcsin(-hy / sd)
    
    # We can define 4 chunks: Right, Top, Left, Bottom
    # RIGHT CHUNK
    arc_right = np.linspace(ang_br, ang_tr, n_arc)
    pts_right = [(hx, hy), (hx, -hy)] + [(sd*np.cos(a), sd*np.sin(a)) for a in arc_right]
    faces.append(pts_right)
    
    # TOP CHUNK
    arc_top = np.linspace(ang_tr, ang_tl, n_arc)
    pts_top = [(-hx, hy), (hx, hy)] + [(sd*np.cos(a), sd*np.sin(a)) for a in arc_top]
    faces.append(pts_top)
    
    # LEFT CHUNK
    ang_bl_real = np.pi + np.arcsin(hy/sd)
    ang_tl_real = np.pi - np.arcsin(hy/sd)
    arc_left = np.linspace(ang_tl_real, ang_bl_real, n_arc)
    pts_left = [(-hx, -hy), (-hx, hy)] + [(sd*np.cos(a), sd*np.sin(a)) for a in arc_left]
    faces.append(pts_left)
    
    # BOTTOM CHUNK
    ang_bl_bot = np.pi + np.arcsin(hy/sd)
    ang_br_bot = 2*np.pi - np.arcsin(hy/sd)
    arc_bot = np.linspace(ang_bl_bot, ang_br_bot, n_arc)
    pts_bot = [(hx, -hy), (-hx, -hy)] + [(sd*np.cos(a), sd*np.sin(a)) for a in arc_bot]
    faces.append(pts_bot)
    
    return faces

faces = build_rect_aperture_2d(0.5, 0.2, 2.0)
print('Generated', len(faces), 'faces')
