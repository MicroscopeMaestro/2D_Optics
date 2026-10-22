import numpy as np
from optics_app.engine.surface import Surface
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.optimizer import OpticalOptimizer, compute_system_efl

s0 = Surface(name="OBJ", radius=0.0, thickness=0.0, glass="AIR")
s1 = Surface(name="S1", radius=22.0, thickness=4.0, glass="N-BAF10")
s2 = Surface(name="S2", radius=0.0, thickness=2.0, glass="SF10")
s3 = Surface(name="S3", radius=-22.0, thickness=45.0, glass="AIR")
s4 = Surface(name="DET", radius=0.0, thickness=0.0, glass="AIR")

stack = SurfaceStack()
stack.surfaces = [s0, s1, s2, s3, s4]
system = OpticalSystem(stack)

target_efl = 50.0
optimizer = OpticalOptimizer(
    system=system,
    wavelengths=[0.5876],
    method="Nelder-Mead",
    max_iter=200,
    target="Spot Size (Focus)",
    num_rays=15,
    entrance_pupil_radius=5.0,
    target_efl=target_efl,
    efl_weight=100.0
)

var_indices = {
    "radius": [1, 3],
    "thickness": [3]
}

x0 = [22.0, -22.0, 45.0]
pure, pen = optimizer._evaluate_merit_and_penalty(x0, var_indices)
print(f"Initial x0: pure={pure:.4f}, pen={pen:.4f}")

# What if x has R1=45, R3=-45, T3=45?
x_50 = [45.0, -45.0, 45.0]
pure50, pen50 = optimizer._evaluate_merit_and_penalty(x_50, var_indices)
print(f"x_50: pure={pure50:.4f}, pen={pen50:.4f}")

res = optimizer.optimize(variable_indices=var_indices)
print(f"Optimized x: {optimizer.best_x}")
efl, bfl = compute_system_efl(res.optimized_stack.surfaces, wavelength_um=0.5876)
print(f"Optimized EFL: {efl:.3f} mm, BFL: {bfl:.3f} mm, final_rms: {res.final_rms:.4f}")
