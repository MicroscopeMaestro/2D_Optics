import json
import glob

for f in glob.glob('*.json'):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            
        modified = False
        surfs = data.get('surfaces', [])
        new_surfs = []
        for s in surfs:
            if 'slit' in str(s.get('name', '')).lower():
                print('Found slit in ' + f + ': ' + str(s.get('name')))
                modified = True
            else:
                new_surfs.append(s)
                
        if modified:
            data['surfaces'] = new_surfs
            with open(f, 'w') as file:
                json.dump(data, file, indent=4)
            print('Removed slit from ' + f)
    except Exception as e:
        pass
