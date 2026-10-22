
import json
data = json.load(open('spectrometer_yz_40mm_100mm_final.json'))
data['surfaces'][19]['thickness'] = 154.83189439
data['surfaces'][22]['thickness'] = 5.00011578
data['surfaces'][24]['thickness'] = 5.10242021
json.dump(data, open('spectrometer_yz_40mm_100mm_optimized.json', 'w'), indent=4)

