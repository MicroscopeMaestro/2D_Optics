with open('build_nir_mirror.py', 'r') as f:
    content = f.read()

content = content.replace("sys_opt.source_config.wavelengths = \"0.8, 1.75, 2.7\"", "")

new_trace = '''
res_rays = []
for wl in [0.8, 1.75, 2.7]:
    for oy in np.linspace(-0.1, 0.1, 5):
        for phi in np.linspace(-0.1, 0.1, 5):
            r = Ray(pos=np.array([0., oy, 0.]), dir=np.array([0., np.sin(phi), np.cos(phi)]), wavelength=wl)
            tr = sys_opt.trace_ray(r)
            if not tr.vignetted:
                res_rays.append(tr)

class MockRes:
    def __init__(self, rays):
        self.ray_paths = [r.history_points for r in rays]

res = MockRes(res_rays)
'''

content = content.replace("tracer = DualSpeedTracer(sys_opt)\nres = tracer.trace_full()", new_trace)
content = content.replace("print(f\"Centroid X: {res.centroid_x:.4f}, Y: {res.centroid_y:.4f}\")", "")
content = content.replace("print(f\"RMS Radius: {res.rms_radius:.4f}\")", "")

with open('build_nir_mirror.py', 'w') as f:
    f.write(content)
