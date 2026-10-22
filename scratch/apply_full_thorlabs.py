import json

with open('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json') as f:
    config = json.load(f)

s = config['surfaces']

# ═══════════════════════════════════════════════════════════════════
#  Thorlabs Official Prescriptions
#  (radii, center thickness, glass, BFL)
# ═══════════════════════════════════════════════════════════════════

# ── ACL25416U  f=16mm aspheric  CT=14.0mm  BFL=7.3mm ────────────
#    Collection side (light hits flat side first)
s[1].update({'name':'Collect_ACL25416U_S1',  'radius':  0.0,  'glass':'B270', 'thickness':14.0})
s[2].update({'name':'Collect_ACL25416U_S2',  'radius': -8.32, 'glass':'AIR',  'thickness':30.0})
# (30 mm gap to collection doublet — user adjusts this)

# ── AC254-050-C  f=50mm  CT=10.8mm  BFL=41.2mm ──────────────────
#    R1=22.9 N-LAK22 t=9.0 / R2=-25.9 N-SF6 t=1.8 / R3=194.5
s[3].update({'name':'Collect_AC254-050-C_S1', 'radius':  22.9,  'glass':'N-LAK22', 'thickness': 9.0})
s[4].update({'name':'Collect_AC254-050-C_S2', 'radius': -25.9,  'glass':'N-SF6',   'thickness': 1.8})
s[5].update({'name':'Collect_AC254-050-C_S3', 'radius': 194.5,  'glass':'AIR',     'thickness':41.2})
# 41.2 mm = BFL to Entrance Slit

# ── Entrance Slit ────────────────────────────────────────────────
s[6]['thickness'] = 22.2   # = BFL of 30mm collimator (slit at front focal point)

# ── AC254-030-C REVERSED  f=30mm  CT=13.0mm  BFL=22.2mm ─────────
#    Reversed: R1→64.3 F2 t=2.0 / R2→14.2 N-BAF10 t=7.0 / R3→-19.4
#    (curved side faces slit for better collimation of diverging beam)
s[7].update({'name':'Collimator_AC254-030-C_S1', 'radius':  64.3, 'glass':'F2',      'thickness': 2.0})
s[8].update({'name':'Collimator_AC254-030-C_S2', 'radius':  14.2, 'glass':'N-BAF10', 'thickness': 7.0})
s[9].update({'name':'Collimator_AC254-030-C_S3', 'radius': -19.4, 'glass':'AIR',     'thickness':10.0})

# ── Edge Filter (flat glass, N-BK7) ──────────────────────────────
s[10].update({'thickness': 3.5})
s[11].update({'thickness':11.5})

# ── Aperture Stop → Grating ──────────────────────────────────────
s[12]['thickness'] = 41.482  # keep original

# ── AC254-030-C  f=30mm  CT=13.0mm  BFL=22.2mm ──────────────────
#    Standard orientation: R1=19.4 N-BAF10 t=7.0 / R2=-14.2 F2 t=2.0 / R3=-64.3
s[15].update({'name':'Focusing_AC254-030-C_S1', 'radius':  19.4, 'glass':'N-BAF10', 'thickness': 7.0})
s[16].update({'name':'Focusing_AC254-030-C_S2', 'radius': -14.2, 'glass':'F2',      'thickness': 2.0})
s[17].update({'name':'Focusing_AC254-030-C_S3', 'radius': -64.3, 'glass':'AIR',     'thickness':22.2})
# 22.2 mm = BFL of 30mm lens → DMD surface

# ── AC254-100-C  f=100mm  CT=8.3mm  BFL=90.4mm ──────────────────
#    R1=270.4 N-SF6HT t=2.0 / R2=44.2 N-BAF10 t=6.5 / R3=-51.5
#    NOTE: Thorlabs says individual elements t1=6.5 t2=1.8, not 2.0/4.0
s[20].update({'name':'Relay_AC254-100-C_S1', 'radius': 270.4, 'glass':'N-SF6HT', 'thickness': 1.8})
s[21].update({'name':'Relay_AC254-100-C_S2', 'radius':  44.2, 'glass':'N-BAF10', 'thickness': 6.5})
s[22].update({'name':'Relay_AC254-100-C_S3', 'radius': -51.5, 'glass':'AIR',     'thickness':90.4})
# 90.4 mm = BFL to condenser

# ── ACL25416U  f=16mm  CT=14.0mm  BFL=7.3mm ─────────────────────
#    Condenser side (curved side faces incoming beam)
s[23].update({'name':'Condenser_ACL25416U_S1', 'radius':  8.32, 'glass':'B270', 'thickness':14.0})
s[24].update({'name':'Condenser_ACL25416U_S2', 'radius':  0.0,  'glass':'AIR',  'thickness': 7.3})
# 7.3 mm = BFL to detector

with open('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json', 'w') as f:
    json.dump(config, f, indent=4)

# Print result
print(f"{'Idx':>3}  {'Name':35}  {'R':>8}  {'t (mm)':>8}  {'Glass'}")
print('-' * 78)
for i, sf in enumerate(config['surfaces']):
    print(f"{i:>3}  {sf.get('name',''):35}  {sf.get('radius',0):>8.3f}  {sf.get('thickness',0):>8.3f}  {sf.get('glass','AIR')}")
