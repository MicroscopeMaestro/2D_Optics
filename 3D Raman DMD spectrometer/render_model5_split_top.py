"""Static inspection views of the exported split STEP solids."""
from pathlib import Path
import sys
import FreeCAD
import Part
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

split_only = '--split-only' in sys.argv
out = Path(__file__).resolve().parent / ('model5_aperture_split' if split_only else 'model5_split_top')
names = ['Top_remaining.step', 'Top_DMD_aperture.step'] if split_only else ['Top_main_M4.step', 'Top_DMD_M4.step']
shapes = [Part.read(str(out / name)) for name in names]
fig = plt.figure(figsize=(14, 8), facecolor='#f5f7fa')
for index, exploded in enumerate([False, True]):
    ax = fig.add_subplot(1, 2, index+1, projection='3d')
    for shape, color, is_dmd in zip(shapes, ['#99b3c9', '#e9a44c'], [False, True]):
        vertices, faces = shape.tessellate(0.15)
        dy = -20 if exploded and is_dmd else 0
        dx = 12 if exploded and is_dmd else 0
        points = [(v.z, v.y+dy, v.x+dx) for v in vertices]
        polygons = [[points[i] for i in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=color,
                                           shade=True,
                                           linewidths=0, zsort='average'))
    ax.set(xlim=(10,240), ylim=(-110,30), zlim=(-5,35))
    ax.set_box_aspect((230,140,55))
    ax.view_init(elev=62, azim=-80)
    ax.set_xlabel('Z / mm')
    ax.set_ylabel('Y / mm')
    ax.set_zlabel('X / mm')
    ax.set_title('Exploded joint view' if exploded else 'Assembled top', fontsize=16)
    ax.set_facecolor('#f5f7fa')
fig.suptitle('Model 5 top — DMD aperture section / unchanged geometry' if split_only else 'Model 5 top — separate DMD section with two M4 insert joints', fontsize=19)
fig.text(0.5,0.04,'Orange: DMD aperture + existing mounting holes    |    Blue: remaining top    |    Split only' if split_only else 'Orange: DMD section    |    Blue: main housing    |    Provisional insert pockets: Ø5.6 × 6 mm',ha='center',fontsize=12)
plt.savefig(out / 'split_top_preview.png', dpi=160, bbox_inches='tight')
