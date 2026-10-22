import numpy as np

def test_grating_diffraction(groove_vector, N, ray_dir, lines_per_mm, order, wavelength_um):
    v_in = np.array(ray_dir, dtype=float)
    v_in /= np.linalg.norm(v_in)
    
    N = np.array(N, dtype=float)
    N /= np.linalg.norm(N)
    
    L = np.array(groove_vector, dtype=float)
    L_proj = L - np.dot(L, N) * N
    e_groove = L_proj / np.linalg.norm(L_proj)
    
    # Dispersion direction in surface plane is perpendicular to grooves and normal
    e_disp = np.cross(N, e_groove) # in Y-Z plane
    e_disp /= np.linalg.norm(e_disp)
    
    v_in_groove = np.dot(v_in, e_groove)
    v_in_disp = np.dot(v_in, e_disp)
    v_in_N = np.dot(v_in, N)
    print("e_groove:", e_groove)
    print("e_disp:", e_disp)
    print("v_in_disp:", v_in_disp)
    print("v_in_N:", v_in_N)
    
    d_um = 1000.0 / lines_per_mm
    v_out_groove = v_in_groove
    # Note: grating equation sign convention
    # If order is +1 or -1, depending on whether diffracted towards specular or away
    v_out_disp = v_in_disp + (order * wavelength_um / d_um)
    print("v_out_disp with +order:", v_out_disp)
    print("v_out_disp with -order:", v_in_disp - (order * wavelength_um / d_um))
    
    sum_sq = v_out_groove**2 + v_out_disp**2
    print("sum_sq:", sum_sq)
    if sum_sq > 1.0:
        return None, True
        
    v_out_N = np.sign(v_in_N) * np.sqrt(max(0.0, 1.0 - sum_sq))
    v_out = v_out_groove * e_groove + v_out_disp * e_disp + v_out_N * N
    v_out /= np.linalg.norm(v_out)
    return v_out, False

# Incident ray along Z [0,0,1], surface tilted around X by 30 deg -> N = [0, -sin(30), cos(30)] = [0, -0.5, 0.866]
N = [0.0, -0.5, 0.866025]
grooves = [1.0, 0.0, 0.0]
ray_in = [0.0, 0.0, 1.0]
v_out, evan = test_grating_diffraction(grooves, N, ray_in, 600.0, 1, 0.85)
print("v_out:", v_out)
print("X component is 0?", abs(v_out[0]) < 1e-10)
print("Y component:", v_out[1])
print("Z component:", v_out[2])
