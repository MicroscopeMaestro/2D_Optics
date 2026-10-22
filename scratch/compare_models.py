import json

with open("spectrometer_raman_50mm_100mm.json", "r") as f:
    d_auto = json.load(f)

with open("spectrometer_raman_50mm_100mm_manual.json", "r") as f:
    d_man = json.load(f)

print("Auto keys:", list(d_auto.keys()))
print("Manual keys:", list(d_man.keys()))
print("Auto surfaces:", len(d_auto.get("surfaces", [])))
print("Manual surfaces:", len(d_man.get("surfaces", [])))

print("\n--- Compare Surfaces ---")
for i in range(max(len(d_auto.get("surfaces", [])), len(d_man.get("surfaces", [])))):
    s_a = d_auto["surfaces"][i] if i < len(d_auto["surfaces"]) else {}
    s_m = d_man["surfaces"][i] if i < len(d_man["surfaces"]) else {}
    diffs = []
    for k in ["name", "radius", "thickness", "glass", "tilt_x", "tilt_y", "tilt_z", "semi_diameter"]:
        va = s_a.get(k)
        vm = s_m.get(k)
        if va != vm:
            diffs.append(f"{k}: auto={va} vs man={vm}")
    if diffs:
        print(f"S{i:02d} ({s_m.get('name', s_a.get('name'))}): " + ", ".join(diffs))
    else:
        print(f"S{i:02d} ({s_m.get('name')}): IDENTICAL")
