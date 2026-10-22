import sys

path = 'optics_app/gui/visualizer_3d.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.startswith('def build_rectangular_aperture_geometry('):
        start_idx = i
    if start_idx != -1 and line.strip() == 'return world_quads, inner_rim, outer_rim, all_verts':
        end_idx = i
        break

if start_idx == -1 or end_idx == -1:
    print('Could not find function')
    sys.exit(1)

new_func = '''def build_rectangular_aperture_geometry(
    surf: Surface,
    thickness: float = 1.5,
    outer_scale: float = 1.0,
    surrounding_max_sd: float = 0.0,
    z_offset_mode: str = "auto",
):
    """
    Build 3D closed solid plate for a rectangular aperture.
    Draws a circular outer disk (semi_diameter) with a rectangular inner hole (aperture_x_half, aperture_y_half).
    """
    hx = max(float(getattr(surf, "aperture_x_half", 0.0)), 0.001)
    hy = max(float(getattr(surf, "aperture_y_half", 0.0)), 0.001)
    sd = max(float(getattr(surf, "semi_diameter", 0.0)), 0.1)

    t = float(getattr(surf, "aperture_thickness", 0.0) or getattr(surf, "stop_thickness", 0.0) or thickness)
    t = max(t, 0.4)

    if z_offset_mode == "backward":
        z_front_local = -t
        z_back_local = 0.0
    elif z_offset_mode == "centered":
        z_front_local = -t * 0.5
        z_back_local = t * 0.5
    else:
        z_front_local = 0.0
        z_back_local = t

    center = np.asarray(getattr(surf, "global_origin", [0.0, 0.0, 0.0]), dtype=float)
    rotation = np.asarray(getattr(surf, "global_R", np.eye(3)), dtype=float)

    angular_samples = 64
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_t[np.abs(cos_t) < 1e-9] = 1e-9 * np.sign(cos_t[np.abs(cos_t) < 1e-9] + 1e-12)
    sin_t[np.abs(sin_t) < 1e-9] = 1e-9 * np.sign(sin_t[np.abs(sin_t) < 1e-9] + 1e-12)
    
    r_x = np.abs(hx / cos_t)
    r_y = np.abs(hy / sin_t)
    r_in = np.minimum(r_x, r_y)
    r_out = np.full_like(theta, max(sd, np.max(r_in) + 0.001))
    
    r_in = np.minimum(r_in, r_out - 0.001)

    n_rad = 4
    radii_frac = np.linspace(0.0, 1.0, n_rad)
    radius_grid = r_in[np.newaxis, :] + radii_frac[:, np.newaxis] * (r_out - r_in)[np.newaxis, :]
    
    x_local = radius_grid * np.cos(theta)[np.newaxis, :]
    y_local = radius_grid * np.sin(theta)[np.newaxis, :]
    
    z_front_grid = np.full_like(x_local, z_front_local)
    z_back_grid = np.full_like(x_local, z_back_local)
    
    def transform_grid(x_g, y_g, z_g):
        pts = np.column_stack([x_g.ravel(), y_g.ravel(), z_g.ravel()])
        world = (rotation @ pts.T).T + center
        return world[:, 0].reshape(x_g.shape), world[:, 1].reshape(y_g.shape), world[:, 2].reshape(z_g.shape)

    xf, yf, zf = transform_grid(x_local, y_local, z_front_grid)
    xb, yb, zb = transform_grid(x_local, y_local, z_back_grid)

    world_quads = []
    
    for i in range(n_rad - 1):
        for j in range(angular_samples - 1):
            world_quads.append([
                (xf[i, j], yf[i, j], zf[i, j]),
                (xf[i, j+1], yf[i, j+1], zf[i, j+1]),
                (xf[i+1, j+1], yf[i+1, j+1], zf[i+1, j+1]),
                (xf[i+1, j], yf[i+1, j], zf[i+1, j])
            ])
            world_quads.append([
                (xb[i, j], yb[i, j], zb[i, j]),
                (xb[i, j+1], yb[i, j+1], zb[i, j+1]),
                (xb[i+1, j+1], yb[i+1, j+1], zb[i+1, j+1]),
                (xb[i+1, j], yb[i+1, j], zb[i+1, j])
            ])

    for j in range(angular_samples - 1):
        world_quads.append([
            (xf[-1, j], yf[-1, j], zf[-1, j]),
            (xf[-1, j+1], yf[-1, j+1], zf[-1, j+1]),
            (xb[-1, j+1], yb[-1, j+1], zb[-1, j+1]),
            (xb[-1, j], yb[-1, j], zb[-1, j])
        ])
        world_quads.append([
            (xf[0, j], yf[0, j], zf[0, j]),
            (xf[0, j+1], yf[0, j+1], zf[0, j+1]),
            (xb[0, j+1], yb[0, j+1], zb[0, j+1]),
            (xb[0, j], yb[0, j], zb[0, j])
        ])

    inner_rim = (xf[0, :], yf[0, :], zf[0, :])
    outer_rim = (xf[-1, :], yf[-1, :], zf[-1, :])

    all_verts = np.column_stack([xf.ravel(), yf.ravel(), zf.ravel()])
    return world_quads, inner_rim, outer_rim, all_verts
'''

new_lines = lines[:start_idx] + [new_func] + lines[end_idx+1:]
with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Replaced!')
