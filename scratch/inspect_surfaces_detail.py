import json

with open('spectrometer_raman_50mm_100mm_manual.json', 'r') as f:
    d = json.load(f)

for i, s in enumerate(d['surfaces']):
    uid = s.get('component_uid')
    grp = s.get('group_name')
    stype = s.get('surface_type')
    glass = s.get('glass')
    r = s.get('radius')
    t = s.get('thickness')
    semi = s.get('semi_diameter')
    mech = s.get('mechanical_state')
    ap_type = s.get('aperture_type')
    ap_max = s.get('aperture_a_semi')
    print(f"{i:2d}: {s['name']:<22} | uid={uid[:8] if uid else 'none':<8} | grp={grp:<14} | stype={stype:<10} | g={glass:<8} | r={r:<8} | t={t:<6} | semi={semi} | ap={ap_type}({ap_max})")
