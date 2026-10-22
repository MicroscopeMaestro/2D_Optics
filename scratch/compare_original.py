import json

with open("spectrometer_raman_50mm_100mm_manual.original_2026-10-04.json", "r") as f:
    d_orig = json.load(f)

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    d_cur = json.load(f)

for k in set(list(d_orig.keys()) + list(d_cur.keys())):
    if k not in d_orig:
        print(f"Key added in cur: {k}")
    elif k not in d_cur:
        print(f"Key removed in cur: {k}")
    elif d_orig[k] != d_cur[k]:
        print(f"Key changed: {k}")
        if k == "surfaces":
            for i in range(len(d_cur["surfaces"])):
                s_o = d_orig["surfaces"][i]
                s_c = d_cur["surfaces"][i]
                diffs = [f"{sk}: orig={s_o.get(sk)} vs cur={s_c.get(sk)}" for sk in s_o if s_o.get(sk) != s_c.get(sk)]
                if diffs:
                    print(f"  S{i:02d}: " + ", ".join(diffs))
