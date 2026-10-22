
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath('')))
from v2.engine.materials import get_refractive_index
print(get_refractive_index('N-BAK4', 1.225))

