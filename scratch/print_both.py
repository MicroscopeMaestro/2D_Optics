import json

def load(path):
    with open(path) as f:
        return json.load(f)

manual = load('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual.json')
current = load('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json')

print("=== MANUAL EDITION ===")
print(f"{'Idx':>3}  {'Name':35}  {'Radius':>10}  {'Thickness':>10}  {'Glass':>12}")
print('-' * 80)
for i, s in enumerate(manual['surfaces']):
    print(f"{i:>3}  {s.get('name',''):35}  {s.get('radius',0):>10.3f}  {s.get('thickness',0):>10.3f}  {s.get('glass','AIR'):>12}")

print()
print("=== CURRENT (OPTIMIZED) EDITION ===")
print(f"{'Idx':>3}  {'Name':35}  {'Radius':>10}  {'Thickness':>10}  {'Glass':>12}")
print('-' * 80)
for i, s in enumerate(current['surfaces']):
    print(f"{i:>3}  {s.get('name',''):35}  {s.get('radius',0):>10.3f}  {s.get('thickness',0):>10.3f}  {s.get('glass','AIR'):>12}")
