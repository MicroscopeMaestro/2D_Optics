import re

with open('build_spectrometer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will just revert to the original ordering using git checkout, then re-apply the patch!
import subprocess
subprocess.run(['git', 'checkout', '--', 'build_spectrometer.py'])
