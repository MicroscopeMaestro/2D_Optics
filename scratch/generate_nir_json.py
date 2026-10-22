import json

with open('extract.txt', 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('{')
try:
    data, _ = json.JSONDecoder().raw_decode(text[start:])
except Exception as e:
    print("Failed to decode JSON:", e)
    exit(1)

# Apply modifications from the implementation plan
for surf in data.get('surfaces', []):
    name = surf.get('name', '')
    
    if "DMD4500" in name:
        surf['thickness'] = 100.0
        
    elif name.startswith("M2_"):
        surf['name'] = "M2_CM254-100-E02_Collector"
        surf['radius'] = -200.0
        
    elif "AC254-030-C_Cond_S3" in name:
        surf['thickness'] = 53.44
        
    elif name == 'Surface_3':
        surf['thickness'] = 7.26
            
    elif name == 'Surface_5':
        surf['semi_diameter'] = 0.50

# Remove Surface_1
new_surfaces = [s for s in data.get('surfaces', []) if s.get('name') != 'Surface_1']
data['surfaces'] = new_surfaces

out_path = "dmd4500_nir_fingerprint_0p8mm_spectrometer.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Successfully wrote {out_path}")
