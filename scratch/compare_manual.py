import json

def load(path):
    with open(path) as f:
        return json.load(f)

manual = load('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm_manual.json')
current = load('c:/Users/Juan/Desktop/spectrometer_raman_50mm_100mm.json')

ms = manual['surfaces']
cs = current['surfaces']

print(f"{'Idx':>3}  {'Name (manual)':30}  {'R_manual':>10}  {'t_manual':>9}  {'glass_m':>12}  |  {'R_current':>10}  {'t_current':>10}  {'glass_c':>12}  MATCH?")
print('-' * 130)

n = max(len(ms), len(cs))
for i in range(n):
    m = ms[i] if i < len(ms) else {}
    c = cs[i] if i < len(cs) else {}
    rm = m.get('radius', 0); rc = c.get('radius', 0)
    tm = m.get('thickness', 0); tc = c.get('thickness', 0)
    gm = m.get('glass', 'AIR'); gc = c.get('glass', 'AIR')
    name = m.get('name', c.get('name', ''))
    match = ('OK' if (abs(rm-rc)<0.01 and abs(tm-tc)<0.1 and gm==gc) else '*** DIFF ***')
    diffs = []
    if abs(rm-rc) >= 0.01: diffs.append(f'R:{rm:.2f}vs{rc:.2f}')
    if abs(tm-tc) >= 0.1:  diffs.append(f't:{tm:.2f}vs{tc:.2f}')
    if gm != gc:            diffs.append(f'glass:{gm}vs{gc}')
    diff_str = ' '.join(diffs)
    print(f"{i:>3}  {name:30}  {rm:>10.3f}  {tm:>9.3f}  {gm:>12}  |  {rc:>10.3f}  {tc:>10.3f}  {gc:>12}  {match}  {diff_str}")
