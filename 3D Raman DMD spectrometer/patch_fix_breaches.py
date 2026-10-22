with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    ( 17.5, 100.0),  # RIGHT MID 1
    (-70.0, 115.0),  # LEFT MID 1
    ( 17.5, 150.0),  # RIGHT MID 2
    (-50.0, 170.0),  # LEFT MID 2
    (-15.0, 227.0),  # EXIT RIGHT
    (-46.0, 227.0)   # EXIT LEFT
]'''
new_screws = '''MANUAL_SCREW_POSITIONS = [
    ( 17.5, 20.0),   # ENTRANCE RIGHT
    (-17.5, 20.0),   # ENTRANCE LEFT
    ( 17.5, 100.0),  # RIGHT MID 1
    (-90.0, 115.0),  # LEFT MID 1 (moved outward)
    ( 17.5, 150.0),  # RIGHT MID 2
    (-65.0, 170.0),  # LEFT MID 2 (moved outward)
    (-15.0, 230.0),  # EXIT RIGHT (moved behind sensor)
    (-45.0, 230.0)   # EXIT LEFT (moved behind sensor)
]'''
content = content.replace(old_screws, new_screws)

with open('build_spectrometer.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Displaced all breaching screws deeply into solid plastic!")
