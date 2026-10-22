import numpy as np

hx, hy = 0.5, 2.0
corners = np.array([
    np.arctan2(hy, hx),
    np.arctan2(hy, -hx),
    np.arctan2(-hy, -hx),
    np.arctan2(-hy, hx)
])
print('Original corners:', corners)

# Fix: convert to [0, 2pi]
corners = np.mod(corners, 2 * np.pi)
print('Fixed corners:', corners)

axes_ang = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
theta = np.sort(np.unique(np.concatenate([np.linspace(0, 2*np.pi, 64), corners, axes_ang])))
print('Theta min/max:', theta.min(), theta.max())
