import json

with open('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json') as f:
    config = json.load(f)

s = config['surfaces']

# ── Collection: ACL25416U aspheric (unchanged, already correct) ──────────
# S1: flat, B270, t=8.0
# S2: R=-8.32, AIR, t=30.0

# ── Collection doublet: AC254-050-C (50 mm, C-coat) ──────────────────────
# Thorlabs: R1=22.9 N-LAK22 / R2=-25.9 N-SF6 / R3=194.5  BFL=41.2
s[3].update({'name':'Collect_AC254-050-C_S1', 'radius': 22.9,   'glass':'N-LAK22', 'thickness': 9.0})
s[4].update({'name':'Collect_AC254-050-C_S2', 'radius':-25.9,   'glass':'N-SF6',   'thickness': 1.8})
s[5].update({'name':'Collect_AC254-050-C_S3', 'radius': 194.5,  'glass':'AIR'})
# leave s[5]['thickness'] as-is (user sets it)

# ── Collimator: AC254-030-C reversed (30 mm, C-coat) ─────────────────────
# Reversed so flat/concave side faces diverging slit beam
# Thorlabs: R1=19.4 N-BAF10 / R2=-14.2 F2 / R3=-64.3 → reversed: 64.3 F2 / 14.2 N-BAF10 / -19.4
s[7].update({'name':'Collimator_AC254-030-C_S1', 'radius': 64.3,  'glass':'F2',      'thickness': 2.0})
s[8].update({'name':'Collimator_AC254-030-C_S2', 'radius': 14.2,  'glass':'N-BAF10', 'thickness': 7.0})
s[9].update({'name':'Collimator_AC254-030-C_S3', 'radius':-19.4,  'glass':'AIR'})

# ── Focusing doublet: AC254-030-C (30 mm, C-coat) ────────────────────────
# Thorlabs: R1=19.4 N-BAF10 / R2=-14.2 F2 / R3=-64.3  BFL≈23.5
s[15].update({'name':'Focusing_AC254-030-C_S1', 'radius': 19.4,  'glass':'N-BAF10', 'thickness': 7.0})
s[16].update({'name':'Focusing_AC254-030-C_S2', 'radius':-14.2,  'glass':'F2',      'thickness': 2.0})
s[17].update({'name':'Focusing_AC254-030-C_S3', 'radius':-64.3,  'glass':'AIR'})
# leave s[17]['thickness'] as-is (user sets it)

# ── Relay: AC254-100-C (100 mm, C-coat) ──────────────────────────────────
# Thorlabs: R1=270.4 N-SF6HT / R2=44.2 N-BAF10 / R3=-51.5
s[20].update({'name':'Relay_AC254-100-C_S1', 'radius': 270.4, 'glass':'N-SF6HT', 'thickness': 2.0})
s[21].update({'name':'Relay_AC254-100-C_S2', 'radius':  44.2, 'glass':'N-BAF10', 'thickness': 4.0})
s[22].update({'name':'Relay_AC254-100-C_S3', 'radius': -51.5, 'glass':'AIR'})
# leave s[22]['thickness'] as-is (user sets it)

# ── Condenser: ACL25416U aspheric (unchanged, already correct) ────────────
# S23: R=8.32, B270, t=9.0
# S24: flat, AIR

# ── Restore distances that got broken by optimizer ──────────────────────
s[14]['thickness'] = 48.026   # CB_After_Grating (restore original)
s[17]['thickness'] = 23.54    # Focusing BFL → DMD (Thorlabs 30mm BFL)

with open('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json', 'w') as f:
    json.dump(config, f, indent=4)

# Print result for confirmation
print(f"{'Idx':>3}  {'Name':35}  {'Radius':>8}  {'Thickness':>9}  {'Glass'}")
print('-' * 75)
for i, sf in enumerate(config['surfaces']):
    print(f"{i:>3}  {sf.get('name',''):35}  {sf.get('radius',0):>8.3f}  {sf.get('thickness',0):>9.3f}  {sf.get('glass','AIR')}")
