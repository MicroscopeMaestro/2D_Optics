
import sys, os, json, numpy as np
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'r') as f: data = json.load(f)
data['surfaces'][6]['thickness'] = 23.3 # True Collimator distance!
with open('teamwork_yz_spectrometer/spectrometer_yz_1064_final.json', 'w') as f: json.dump(data, f, indent=4)

