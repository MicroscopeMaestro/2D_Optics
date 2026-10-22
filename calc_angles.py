
import numpy as np
wls = np.array([0.8, 1.75, 2.7])
d = 1.0 / 150 * 1000 # 6.666 um
theta_i = np.radians(15.0)

for wl in wls:
    sin_theta_d = np.sin(theta_i) - wl / d
    theta_d = np.degrees(np.arcsin(sin_theta_d))
    print(f'{wl} um: {theta_d:.2f} deg')

