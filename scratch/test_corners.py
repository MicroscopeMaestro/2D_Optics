import numpy as np

hx, hy, sd = 0.5, 0.2, 2.0
corners = np.array([
    np.arctan2(hy, hx),
    np.arctan2(hy, -hx),
    np.arctan2(-hy, -hx),
    np.arctan2(-hy, hx)
])
axes_ang = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
theta = np.sort(np.unique(np.concatenate([np.linspace(0, 2*np.pi, 64), corners, axes_ang])))

cos_t = np.cos(theta)
sin_t = np.sin(theta)
cos_t[np.abs(cos_t) < 1e-9] = 1e-9 * np.sign(cos_t[np.abs(cos_t) < 1e-9] + 1e-12)
sin_t[np.abs(sin_t) < 1e-9] = 1e-9 * np.sign(sin_t[np.abs(sin_t) < 1e-9] + 1e-12)

r_x = np.abs(hx / cos_t)
r_y = np.abs(hy / sin_t)
r_in = np.minimum(r_x, r_y)

x_in = r_in * np.cos(theta)
y_in = r_in * np.sin(theta)

print('Max deviation from box:', np.max(np.maximum(np.abs(x_in) - hx, np.abs(y_in) - hy)))
