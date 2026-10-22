import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Let's import the actual function to test it visually
import sys
sys.path.append('.')
from optics_app.engine.surface import Surface
from optics_app.gui.visualizer_3d import build_rectangular_aperture_geometry

surf = Surface(name="TestSlit", semi_diameter=12.7, aperture_x_half=0.5, aperture_y_half=2.0)
faces, in_rim, out_rim, verts = build_rectangular_aperture_geometry(surf)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

poly = Poly3DCollection(faces, alpha=0.5, facecolors='b', edgecolors='k')
ax.add_collection3d(poly)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-5, 5)

plt.savefig('scratch/test_rect_cad.png')
