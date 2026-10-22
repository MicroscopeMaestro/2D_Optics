"""
Patch script to integrate Target EFL constraint, paraxial solves resolution, and anti-vignetting spot size optimization
across optimizer.py, optimizer_dock.py, and main_window.py (for both optics_app and v2).
"""
import os

def patch_optimizer_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add compute_system_efl and resolve_solves_and_pickups helpers
    helper_code = '''
def compute_system_efl(
    surfaces: List[Surface],
    wavelength_um: float = 0.5876,
    eval_surface_idx: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute paraxial Effective Focal Length (EFL) and Back Focal Length (BFL) in mm
    using ABCD matrix transfer formalism.
    """
    from optics_app.engine.matrix_tracer import MatrixTracer
    if not surfaces:
        return 0.0, 0.0
    
    end_idx = eval_surface_idx if (eval_surface_idx is not None and eval_surface_idx > 0) else len(surfaces) - 1
    opt_surfaces = surfaces[:end_idx]
    if len(opt_surfaces) > 1 and opt_surfaces[0].surface_type == SurfaceType.OBJECT and not opt_surfaces[0].glass:
        opt_surfaces = opt_surfaces[1:]

    try:
        tracer = MatrixTracer()
        mat = tracer.build_from_system(opt_surfaces, wavelength_um=wavelength_um)
        if abs(mat.C) > 1e-12:
            efl = -1.0 / mat.C
            bfl = -mat.A / mat.C
            return float(efl), float(bfl)
    except Exception:
        pass
    return float('inf'), float('inf')


def resolve_solves_and_pickups(
    stack: SurfaceStack,
    primary_wl: float = 0.5876,
    eval_idx: Optional[int] = None
) -> None:
    """
    Dynamically resolve all Zemax Pickups and Solves across the SurfaceStack:
    - Radius Pickups: R_i = scale * R_ref + offset
    - Thickness Pickups: T_i = scale * T_ref + offset
    - Marginal Ray Solves: solve back thickness to paraxial focus (y = 0)
    """
    if not stack or not hasattr(stack, 'surfaces'):
        return
    n_surfs = len(stack.surfaces)
    for i, s in enumerate(stack.surfaces):
        if getattr(s, 'radius_solve_type', '') == 'Pickup':
            ref_idx = getattr(s, 'radius_pickup_surface', -1)
            if 0 <= ref_idx < n_surfs and ref_idx != i:
                ref_r = stack.surfaces[ref_idx].radius
                scale = getattr(s, 'radius_pickup_scale', 1.0)
                offset = getattr(s, 'radius_pickup_offset', 0.0)
                s.radius = scale * ref_r + offset

        if getattr(s, 'thickness_solve_type', '') == 'Pickup':
            ref_idx = getattr(s, 'thickness_pickup_surface', -1)
            if 0 <= ref_idx < n_surfs and ref_idx != i:
                ref_t = stack.surfaces[ref_idx].thickness
                scale = getattr(s, 'thickness_pickup_scale', 1.0)
                offset = getattr(s, 'thickness_pickup_offset', 0.0)
                s.thickness = scale * ref_t + offset

        if getattr(s, 'thickness_solve_type', '') == 'MarginalRayFocus':
            try:
                from optics_app.engine.matrix_tracer import MatrixTracer
                opt_surfs = stack.surfaces[:i + 1]
                tracer = MatrixTracer()
                mat = tracer.build_from_system(opt_surfs, wavelength_um=primary_wl)
                if abs(mat.C) > 1e-12:
                    bfl = -mat.A / mat.C
                    if bfl > 0.01:
                        s.thickness = float(bfl)
            except Exception:
                pass
'''

    if "def compute_system_efl" not in content:
        # Insert before class OptimizationResult or class OpticalOptimizer
        idx = content.find("class OpticalOptimizer:")
        if idx != -1:
            content = content[:idx] + helper_code + "\n\n" + content[idx:]

    # 2. Add target_efl and efl_weight to OpticalOptimizer.__init__
    old_init = """                 target_surface_index: Optional[int] = None,
                 target_spot_size: float = 0.0,
                 merit_function: Optional[Any] = None,"""

    new_init = """                 target_surface_index: Optional[int] = None,
                 target_spot_size: float = 0.0,
                 target_efl: Optional[float] = None,
                 efl_weight: float = 100.0,
                 merit_function: Optional[Any] = None,"""

    if old_init in content:
        content = content.replace(old_init, new_init, 1)

    # In __init__ body:
    old_init_body = """        self.target_spot_size = float(target_spot_size)
        self.fields = fields if fields is not None else []"""

    new_init_body = """        self.target_spot_size = float(target_spot_size)
        self.target_efl = float(target_efl) if target_efl is not None and float(target_efl) > 0 else None
        self.efl_weight = float(efl_weight)
        self.fields = fields if fields is not None else []"""

    if old_init_body in content and "self.target_efl" not in content:
        content = content.replace(old_init_body, new_init_body, 1)

    # 3. In _evaluate_merit_and_penalty:
    # Resolve solves/pickups and update positions
    old_eval_stack = """        if self.opt_tilt and var_idx < len(variable_values) and (0 <= det_idx < len(eval_stack.surfaces)):
            eval_stack.surfaces[det_idx].tilt_x = float(variable_values[var_idx])

        eval_stack._update_accumulated_positions()
        eval_sys = OpticalSystem(eval_stack)"""

    new_eval_stack = """        if self.opt_tilt and var_idx < len(variable_values) and (0 <= det_idx < len(eval_stack.surfaces)):
            eval_stack.surfaces[det_idx].tilt_x = float(variable_values[var_idx])

        eval_idx = self.target_surface_index if (self.target_surface_index is not None and self.target_surface_index >= 0) else det_idx
        resolve_solves_and_pickups(eval_stack, primary_wl=self.wavelengths[0], eval_idx=eval_idx)

        eval_stack._update_accumulated_positions()
        eval_sys = OpticalSystem(eval_stack)"""

    if old_eval_stack in content:
        content = content.replace(old_eval_stack, new_eval_stack, 1)

    # Anti-vignetting loss penalty and EFL penalty in _evaluate_merit_and_penalty
    old_ray_loss = """        # Handle complete ray loss
        if valid_rays == 0:
            return 0.0, 1e5"""

    new_ray_loss = """        # Anti-Vignetting / Ray Loss Penalty:
        loss_penalty = 0.0
        if total_rays > 0:
            loss_ratio = (total_rays - valid_rays) / float(total_rays)
            if loss_ratio > 0.0:
                loss_penalty = 2000.0 * loss_ratio + 5000.0 * (loss_ratio ** 2)

        if valid_rays < 3:
            return 1000.0, 10000.0 + loss_penalty

        # Evaluate EFL Constraint (if defined or target specifies EFL)
        efl_penalty = 0.0
        if self.target_efl is not None and self.target_efl > 0:
            curr_efl, curr_bfl = compute_system_efl(eval_stack.surfaces, wavelength_um=self.wavelengths[0], eval_surface_idx=eval_idx)
            if math.isinf(curr_efl) or curr_efl <= 0.0:
                efl_penalty = 10000.0
            else:
                efl_rel_err = abs(curr_efl - self.target_efl) / self.target_efl
                efl_penalty = self.efl_weight * (100.0 * efl_rel_err + 500.0 * (efl_rel_err ** 2))"""

    if old_ray_loss in content:
        content = content.replace(old_ray_loss, new_ray_loss, 1)

    # And in Spot Size calculation:
    old_spot_calc = """        else:
            # Default RMS Spot Radius
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for r in traced_rays:
                    if abs(r.wavelength - wl) < 1e-4 and not r.vignetted:
                        for h in r.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    wl_rms_list.append(spot.calculate_rms_radius())
            if wl_rms_list:
                pure_merit = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list))
            else:
                pure_merit = 1000.0"""

    new_spot_calc = """        else:
            # Default RMS Spot Radius (with primary wavelength double-weighting)
            wl_rms_list = []
            for wl in self.wavelengths:
                wl_x, wl_y = [], []
                for r in traced_rays:
                    if abs(r.wavelength - wl) < 1e-4 and not r.vignetted:
                        for h in r.history:
                            if h.get("surface_index") == eval_idx:
                                wl_x.append(h["x"])
                                wl_y.append(h["y"])
                                break
                if wl_x:
                    spot = SpotDiagram(wl_x, wl_y)
                    w_factor = 2.0 if abs(wl - self.wavelengths[0]) < 1e-4 else 1.0
                    wl_rms_list.append(w_factor * spot.calculate_rms_radius())
            if wl_rms_list:
                pure_merit = math.sqrt(sum(r**2 for r in wl_rms_list) / len(wl_rms_list))
            else:
                pure_merit = 1000.0"""

    if old_spot_calc in content:
        content = content.replace(old_spot_calc, new_spot_calc, 1)

    # Also include efl_penalty and loss_penalty in merit return:
    old_ret_penalty = """        # Numerical physical boundary constraints
        for val, (b_min, b_max) in zip(variable_values, bounds):
            if val < b_min:
                dev = b_min - val
                penalty += self.penalty_weight * (dev + dev**2)
            elif val > b_max:
                dev = val - b_max
                penalty += self.penalty_weight * (dev + dev**2)

        return pure_merit, penalty"""

    new_ret_penalty = """        # Numerical physical boundary constraints
        for val, (b_min, b_max) in zip(variable_values, bounds):
            if val < b_min:
                dev = b_min - val
                penalty += self.penalty_weight * (dev + dev**2)
            elif val > b_max:
                dev = val - b_max
                penalty += self.penalty_weight * (dev + dev**2)

        return pure_merit + efl_penalty, penalty + loss_penalty"""

    if old_ret_penalty in content:
        content = content.replace(old_ret_penalty, new_ret_penalty, 1)

    # 4. In OptimizationResult, compute and record final_efl, final_bfl
    old_res_ret = """        return OptimizationResult(
            success=result.success,
            initial_rms=init_pure,
            final_rms=final_pure,
            rms_reduction_pct=pct_gain,
            optimized_thicknesses=final_thick,
            optimized_radii=final_radii,
            optimized_tilts=final_tilts,
            optimized_decenters=final_decenters,
            optimized_stack=opt_stack,
            iterations=result.nit if hasattr(result, 'nit') else self.current_iter,
            message=result.message if hasattr(result, 'message') else "Optimization finished successfully",
            status=TerminationStatus.CONVERGED if result.success else TerminationStatus.MAX_ITERATIONS,
            penalty_value=final_pen,
            per_wavelength_metrics=per_wavelength_metrics,
            validation_result=validation
        )"""

    new_res_ret = """        res_efl, res_bfl = compute_system_efl(opt_stack.surfaces, wavelength_um=self.wavelengths[0], eval_surface_idx=self.target_surface_index)
        return OptimizationResult(
            success=result.success or (final_pure < init_pure),
            initial_rms=init_pure,
            final_rms=final_pure,
            rms_reduction_pct=pct_gain,
            optimized_thicknesses=final_thick,
            optimized_radii=final_radii,
            optimized_tilts=final_tilts,
            optimized_decenters=final_decenters,
            optimized_stack=opt_stack,
            iterations=result.nit if hasattr(result, 'nit') else self.current_iter,
            message=f"{result.message} (Final EFL: {res_efl:.2f} mm, BFL: {res_bfl:.2f} mm)" if (res_efl > 0 and not math.isinf(res_efl)) else (result.message if hasattr(result, 'message') else "Optimization finished successfully"),
            status=TerminationStatus.CONVERGED if result.success else TerminationStatus.MAX_ITERATIONS,
            penalty_value=final_pen,
            per_wavelength_metrics=per_wavelength_metrics,
            validation_result=validation
        )"""

    if old_res_ret in content:
        content = content.replace(old_res_ret, new_res_ret, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Patched optimizer in {file_path}")


def patch_optimizer_dock_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add Target Metric item
    old_targets = """"Spot Size (Focus)",
                "RMS Spot X (Spectral Focus)","""
    new_targets = """"Spot Size (Focus)",
                "Spot Size + Constrain EFL (Focus @ Target EFL)",
                "RMS Spot X (Spectral Focus)","""

    if old_targets in content and "Spot Size + Constrain EFL" not in content:
        content = content.replace(old_targets, new_targets, 1)

    # 2. Add EFL constraint UI controls in _setup_ui
    target_spot_block = """            self.target_spot_label = QLabel("<b>Target Spot Radius (µm):</b>")
            layout.addWidget(self.target_spot_label)
            self.target_spot_spin = QDoubleSpinBox(container)
            self.target_spot_spin.setRange(0.0, 10000.0)
            self.target_spot_spin.setDecimals(1)
            self.target_spot_spin.setValue(0.0)
            layout.addWidget(self.target_spot_spin)
            self.target_spot_label.hide()
            self.target_spot_spin.hide()"""

    replacement_spot_block = """            self.target_spot_label = QLabel("<b>Target Spot Radius (µm):</b>")
            layout.addWidget(self.target_spot_label)
            self.target_spot_spin = QDoubleSpinBox(container)
            self.target_spot_spin.setRange(0.0, 10000.0)
            self.target_spot_spin.setDecimals(1)
            self.target_spot_spin.setValue(0.0)
            layout.addWidget(self.target_spot_spin)
            self.target_spot_label.hide()
            self.target_spot_spin.hide()

            # EFL Constraint Controls
            efl_box = QHBoxLayout()
            self.chk_constrain_efl = QCheckBox("Constrain EFL", container)
            self.chk_constrain_efl.setChecked(False)
            self.chk_constrain_efl.setToolTip("Enforce an exact target Effective Focal Length during optimization")
            self.chk_constrain_efl.toggled.connect(self._on_constrain_efl_toggled)
            efl_box.addWidget(self.chk_constrain_efl)

            self.target_efl_spin = QDoubleSpinBox(container)
            self.target_efl_spin.setRange(0.1, 10000.0)
            self.target_efl_spin.setValue(50.0)
            self.target_efl_spin.setDecimals(2)
            self.target_efl_spin.setSuffix(" mm")
            self.target_efl_spin.setToolTip("Target system Effective Focal Length (EFL)")
            self.target_efl_spin.setEnabled(False)
            efl_box.addWidget(self.target_efl_spin)
            layout.addLayout(efl_box)"""

    if target_spot_block in content and "self.chk_constrain_efl" not in content:
        content = content.replace(target_spot_block, replacement_spot_block, 1)

    # 3. Add _on_constrain_efl_toggled helper and update _on_target_changed
    old_target_changed = """    def _on_target_changed(self, text: str):
        if hasattr(self, 'target_spread_label'):"""

    new_target_changed = """    def _on_constrain_efl_toggled(self, checked: bool):
        if hasattr(self, 'target_efl_spin'):
            self.target_efl_spin.setEnabled(checked)

    def _on_target_changed(self, text: str):
        if "Constrain EFL" in text and hasattr(self, 'chk_constrain_efl'):
            self.chk_constrain_efl.setChecked(True)
        if hasattr(self, 'target_spread_label'):"""

    if old_target_changed in content:
        content = content.replace(old_target_changed, new_target_changed, 1)

    # 4. In on_optimize_clicked, extract target_efl
    old_extract = """                target_spot_um = self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0
                target_spot_mm = float(target_spot_um) / 1000.0"""

    new_extract = """                target_spot_um = self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0
                target_spot_mm = float(target_spot_um) / 1000.0
                target_efl = float(self.target_efl_spin.value()) if (hasattr(self, 'chk_constrain_efl') and (self.chk_constrain_efl.isChecked() or "Constrain EFL" in target)) else None"""

    if old_extract in content and "target_efl" not in content:
        content = content.replace(old_extract, new_extract, 1)

    # In emit:
    old_emit = """self.optimize_requested.emit(self.get_variable_indices(), method, max_iter, target, rays, spread, opt_tilt, target_surf, target_spot_mm)"""
    new_emit = """self.optimize_requested.emit(self.get_variable_indices(), method, max_iter, target, rays, spread, opt_tilt, target_surf, target_spot_mm, target_efl)"""
    if old_emit in content:
        content = content.replace(old_emit, new_emit, 1)

    # 5. In get_config and set_config:
    old_get_cfg = """            "opt_tilt": self.opt_tilt_check.isChecked() if hasattr(self, 'opt_tilt_check') else False,
            "target_spot_um": self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0
        }"""
    new_get_cfg = """            "opt_tilt": self.opt_tilt_check.isChecked() if hasattr(self, 'opt_tilt_check') else False,
            "target_spot_um": self.target_spot_spin.value() if hasattr(self, 'target_spot_spin') else 0.0,
            "constrain_efl": self.chk_constrain_efl.isChecked() if hasattr(self, 'chk_constrain_efl') else False,
            "target_efl": self.target_efl_spin.value() if hasattr(self, 'target_efl_spin') else 50.0
        }"""
    if old_get_cfg in content:
        content = content.replace(old_get_cfg, new_get_cfg, 1)

    old_set_cfg = """        if hasattr(self, 'target_spot_spin') and "target_spot_um" in cfg:
            self.target_spot_spin.setValue(float(cfg["target_spot_um"]))"""
    new_set_cfg = """        if hasattr(self, 'target_spot_spin') and "target_spot_um" in cfg:
            self.target_spot_spin.setValue(float(cfg["target_spot_um"]))
        if hasattr(self, 'chk_constrain_efl') and "constrain_efl" in cfg:
            self.chk_constrain_efl.setChecked(bool(cfg["constrain_efl"]))
        if hasattr(self, 'target_efl_spin') and "target_efl" in cfg:
            self.target_efl_spin.setValue(float(cfg["target_efl"]))"""
    if old_set_cfg in content:
        content = content.replace(old_set_cfg, new_set_cfg, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Patched optimizer dock in {file_path}")


def patch_main_window_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    old_sig = """    def on_optimize(
        self,
        variable_indices: Optional[dict] = None,
        method: str = "Nelder-Mead",
        max_iter: int = 100,
        target: str = "Spot Size (Focus)",
        num_rays: int = 21,
        target_spread: float = 12.0,
        opt_tilt: bool = False,
        target_surface_index: int = -1,
        target_spot_size: float = 0.0,
        blocking: bool = True
    ) -> OptimizationResult:"""

    new_sig = """    def on_optimize(
        self,
        variable_indices: Optional[dict] = None,
        method: str = "Nelder-Mead",
        max_iter: int = 100,
        target: str = "Spot Size (Focus)",
        num_rays: int = 21,
        target_spread: float = 12.0,
        opt_tilt: bool = False,
        target_surface_index: int = -1,
        target_spot_size: float = 0.0,
        target_efl: Optional[float] = None,
        efl_weight: float = 50.0,
        blocking: bool = True
    ) -> OptimizationResult:"""

    if old_sig in content:
        content = content.replace(old_sig, new_sig, 1)

    # In on_optimize OpticalOptimizer initialization:
    old_opt_init = """        optimizer = OpticalOptimizer(
            system=system,
            wavelengths=wavelengths,
            method=method,
            max_iter=max_iter,
            target=target,
            num_rays=num_rays,
            target_spread=target_spread,
            opt_tilt=opt_tilt,
            target_surface_index=target_surface_index if target_surface_index >= 0 else None,
            target_spot_size=target_spot_size,
            fields=getattr(self, 'system_config', None).fields if getattr(self, 'system_config', None) else None
        )"""

    new_opt_init = """        optimizer = OpticalOptimizer(
            system=system,
            wavelengths=wavelengths,
            method=method,
            max_iter=max_iter,
            target=target,
            num_rays=num_rays,
            target_spread=target_spread,
            opt_tilt=opt_tilt,
            target_surface_index=target_surface_index if target_surface_index >= 0 else None,
            target_spot_size=target_spot_size,
            target_efl=target_efl,
            efl_weight=efl_weight,
            fields=getattr(self, 'system_config', None).fields if getattr(self, 'system_config', None) else None
        )"""

    if old_opt_init in content:
        content = content.replace(old_opt_init, new_opt_init, 1)

    # In on_optimize_system:
    old_call = """            res = self.on_optimize(variable_indices=var_indices, method=method, max_iter=max_iter, target=target, num_rays=rays, target_spread=target_spread, opt_tilt=opt_tilt, target_surface_index=target_surf_idx)"""
    new_call = """            target_efl = None
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'chk_constrain_efl') and (self.optimizer_dock.chk_constrain_efl.isChecked() or "Constrain EFL" in target):
                    target_efl = float(self.optimizer_dock.target_efl_spin.value())
            res = self.on_optimize(variable_indices=var_indices, method=method, max_iter=max_iter, target=target, num_rays=rays, target_spread=target_spread, opt_tilt=opt_tilt, target_surface_index=target_surf_idx, target_efl=target_efl)"""

    if old_call in content:
        content = content.replace(old_call, new_call, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Patched main window in {file_path}")

if __name__ == "__main__":
    patch_optimizer_file("optics_app/engine/optimizer.py")
    patch_optimizer_file("v2/engine/optimizer.py")
    patch_optimizer_dock_file("optics_app/gui/optimizer_dock.py")
    patch_optimizer_dock_file("v2/gui/optimizer_dock.py")
    patch_main_window_file("optics_app/gui/main_window.py")
    patch_main_window_file("v2/gui/main_window.py")
