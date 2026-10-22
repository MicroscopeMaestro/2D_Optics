import json
filename = 'spectrometer_yz_40mm_100mm_slit.json'
with open(filename, 'r') as f:
    config = json.load(f)

# Update Sample Object thickness to 10.61 mm to maintain collimation
config['surfaces'][0]['thickness'] = 10.61

with open(filename, 'w') as f:
    json.dump(config, f, indent=4)

print("Updated Sample_Object thickness to 10.61 mm")
