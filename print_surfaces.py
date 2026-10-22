
import json
with open("teamwork_yz_spectrometer/spectrometer_yz_1064.json", "r") as f:
    data = json.load(f)
for i, s in enumerate(data["surfaces"]):
    print(f"{i}: {s.get(\"name\",\"\")} - {s.get(\"surface_type\",\"\")} - th={s.get(\"thickness\",\"\")} - tilt_x={s.get(\"tilt_x\",\"\")}")

