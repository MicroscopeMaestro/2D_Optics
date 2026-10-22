import numpy as np

def test_rect_mesh(hx, hy, sd, angular_samples=64):
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    
    # Calculate r_inner(theta) for a rectangle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Avoid div by zero
    cos_t[np.abs(cos_t) < 1e-9] = 1e-9 * np.sign(cos_t[np.abs(cos_t) < 1e-9] + 1e-12)
    sin_t[np.abs(sin_t) < 1e-9] = 1e-9 * np.sign(sin_t[np.abs(sin_t) < 1e-9] + 1e-12)
    
    r_x = np.abs(hx / cos_t)
    r_y = np.abs(hy / sin_t)
    r_in = np.minimum(r_x, r_y)
    
    # Outer radius is semi-diameter
    r_out = np.full_like(theta, sd)
    
    # If the hole is bigger than the mount in some directions, clamp it so we don't draw inside-out
    r_in = np.minimum(r_in, r_out - 0.001)
    
    # Create radial grid
    n_rad = 4
    radii_frac = np.linspace(0.0, 1.0, n_rad)
    # radius_grid[i, j] = r_in[j] + radii_frac[i] * (r_out[j] - r_in[j])
    radius_grid = r_in[np.newaxis, :] + radii_frac[:, np.newaxis] * (r_out - r_in)[np.newaxis, :]
    
    x_local = radius_grid * np.cos(theta)[np.newaxis, :]
    y_local = radius_grid * np.sin(theta)[np.newaxis, :]
    
    return x_local, y_local

x, y = test_rect_mesh(0.5, 0.2, 12.7)
print("Mesh generated successfully with shape:", x.shape)
