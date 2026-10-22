
import build123d as bd
b1 = bd.Solid.make_box(1, 1, 1).move(bd.Location((0, 0, 0)))
b1.label = 'Box_One'
b2 = bd.Solid.make_box(1, 1, 1).move(bd.Location((2, 0, 0)))
b2.label = 'Box_Two'
comp = bd.Compound(children=[b1, b2])
comp.label = 'Optical_System'
bd.export_step(comp, 'test_labels.step')
with open('test_labels.step') as f:
    step = f.read()
print('Box_One found:', 'Box_One' in step)
print('Box_Two found:', 'Box_Two' in step)
print('Optical_System found:', 'Optical_System' in step)

