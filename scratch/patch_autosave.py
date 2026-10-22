"""
Patch script to add 5-minute Auto-Save functionality to MainWindow in both v2 and optics_app.
"""
import os
import re

def patch_main_window(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Ensure shutil and datetime are imported
    if "import shutil" not in content:
        content = content.replace("import os\nimport sys", "import os\nimport sys\nimport shutil\nfrom datetime import datetime")

    # 2. Add autosave configuration to __init__
    target_init = """        # Current file tracking & dirty flag
        self.current_file_path: Optional[str] = None
        self.is_dirty: bool = False
        self.act_mech_toggle: Optional[Any] = None"""

    replacement_init = """        # Current file tracking & dirty flag
        self.current_file_path: Optional[str] = None
        self.is_dirty: bool = False
        self.act_mech_toggle: Optional[Any] = None

        # Auto-Save Configuration (Default 5 min = 300,000 ms)
        self.autosave_interval_ms: int = 5 * 60 * 1000
        self.autosave_enabled: bool = True
        self.last_autosave_time: Optional[datetime] = None
        self.last_autosave_path: Optional[str] = None

        if QT_AVAILABLE:
            self._autosave_timer = QTimer(self)
        else:
            self._autosave_timer = DummyQTimer(self)

        if self._autosave_timer is not None:
            self._autosave_timer.setInterval(self.autosave_interval_ms)
            self._autosave_timer.setSingleShot(False)
            self._autosave_timer.timeout.connect(self.on_autosave)
            if self.autosave_enabled:
                self._autosave_timer.start()"""

    if target_init in content and "self.autosave_interval_ms" not in content:
        content = content.replace(target_init, replacement_init, 1)
    else:
        print(f"Warning: target_init not found or already patched in {file_path}")

    # 3. Add model signal dirty connections in __init__
    target_signals = """        # 8. Bind Model Signals to Views
        if hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.on_trace_rays)
            except Exception:
                pass"""

    replacement_signals = """        # 8. Bind Model Signals to Views & Track Dirty State
        if hasattr(self.model, "system_updated"):
            try:
                self.model.system_updated.connect(self.on_trace_rays)
            except Exception:
                pass
        if hasattr(self.model, "dataChanged"):
            try:
                self.model.dataChanged.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "layoutChanged"):
            try:
                self.model.layoutChanged.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "rowsInserted"):
            try:
                self.model.rowsInserted.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass
        if hasattr(self.model, "rowsRemoved"):
            try:
                self.model.rowsRemoved.connect(lambda *a, **k: self.set_dirty(True))
            except Exception:
                pass"""

    if target_signals in content and "lambda *a, **k: self.set_dirty(True)" not in content:
        content = content.replace(target_signals, replacement_signals, 1)
    else:
        print(f"Warning: target_signals not found or already patched in {file_path}")

    # 4. Add Auto-Save menu items to File menu
    target_menu = """        act_save_sys_cfg.triggered.connect(lambda: self.on_save_system_config())
        file_menu.addAction(act_save_sys_cfg)

        file_menu.addSeparator()"""

    replacement_menu = """        act_save_sys_cfg.triggered.connect(lambda: self.on_save_system_config())
        file_menu.addAction(act_save_sys_cfg)

        file_menu.addSeparator()

        # Auto-Save Submenu (Zemax / IDE-style background persistence)
        autosave_menu = file_menu.addMenu("⏰ Auto-Save") if hasattr(file_menu, 'addMenu') else None
        target_as_menu = autosave_menu if autosave_menu else file_menu

        self.act_autosave = QAction("⏰ Auto-Save Enabled (Every 5 Min)", self)
        self.act_autosave.setCheckable(True)
        self.act_autosave.setChecked(self.autosave_enabled)
        self.act_autosave.setStatusTip("Toggle automatic project background saving every 5 minutes")
        self.act_autosave.toggled.connect(self._on_toggle_autosave)
        target_as_menu.addAction(self.act_autosave)

        act_as_now = QAction("💾 Save Auto-Save Snapshot Now", self)
        act_as_now.setStatusTip("Immediately create an autosave snapshot of the current design")
        act_as_now.triggered.connect(lambda: self.on_autosave(manual_trigger=True))
        target_as_menu.addAction(act_as_now)

        if autosave_menu and hasattr(autosave_menu, 'addMenu'):
            iv_menu = autosave_menu.addMenu("⏱ Interval Presets")
            for mins in [1, 2, 5, 10, 15, 30]:
                lbl = f"{mins} Minute" + ("s" if mins > 1 else "") + (" (Default)" if mins == 5 else "")
                act_iv = QAction(lbl, self)
                act_iv.triggered.connect(lambda checked=False, m=mins: self.set_autosave_interval(m))
                iv_menu.addAction(act_iv)

        act_rec_as = QAction("🔄 Recover from Auto-Save...", self)
        act_rec_as.setStatusTip("Restore the latest auto-saved project recovery snapshot")
        act_rec_as.triggered.connect(lambda: self.on_recover_autosave())
        target_as_menu.addAction(act_rec_as)

        file_menu.addSeparator()"""

    if target_menu in content and "act_autosave" not in content:
        content = content.replace(target_menu, replacement_menu, 1)
    else:
        print(f"Warning: target_menu not found or already patched in {file_path}")

    # 5. Add dirty & title helper methods, and update _execute_save and on_autosave
    target_execute_save = """    def _execute_save(self, file_path: str) -> bool:
        try:
            stack = self.model.get_surface_stack()"""

    replacement_execute_save = """    def set_dirty(self, dirty: bool = True) -> None:
        \"\"\"Mark design as modified since last save and update window title.\"\"\"
        self.is_dirty = dirty
        self._update_window_title()

    def _update_window_title(self) -> None:
        base_name = os.path.basename(self.current_file_path) if self.current_file_path else "Untitled"
        dirty_mark = " *" if self.is_dirty else ""
        if QT_AVAILABLE and hasattr(self, 'setWindowTitle'):
            self.setWindowTitle(f"2D Optics Simulation & Design Studio - [{base_name}{dirty_mark}]")

    def _on_toggle_autosave(self, enabled: bool) -> None:
        self.autosave_enabled = enabled
        if hasattr(self, '_autosave_timer') and self._autosave_timer:
            if enabled:
                self._autosave_timer.start(self.autosave_interval_ms)
            else:
                self._autosave_timer.stop()
        msg = f"⏰ Auto-save {'enabled (5 min)' if enabled else 'disabled'}"
        if QT_AVAILABLE and hasattr(self, 'statusBar'):
            self.statusBar().showMessage(msg, 3000)
        else:
            self._status_msg = msg

    def set_autosave_interval(self, minutes: float) -> None:
        \"\"\"Set the autosave interval in minutes.\"\"\"
        if minutes <= 0:
            self._on_toggle_autosave(False)
            if hasattr(self, 'act_autosave') and self.act_autosave is not None:
                self.act_autosave.setChecked(False)
            return

        self.autosave_interval_ms = int(minutes * 60 * 1000)
        self.autosave_enabled = True
        if hasattr(self, '_autosave_timer') and self._autosave_timer:
            self._autosave_timer.setInterval(self.autosave_interval_ms)
            if not self._autosave_timer.isActive():
                self._autosave_timer.start()
        if hasattr(self, 'act_autosave') and self.act_autosave is not None:
            self.act_autosave.setChecked(True)
            min_str = int(minutes) if minutes == int(minutes) else minutes
            self.act_autosave.setText(f"⏰ Auto-Save Enabled (Every {min_str} Min)")
        msg = f"⏰ Auto-save interval set to {minutes} min"
        if QT_AVAILABLE and hasattr(self, 'statusBar'):
            self.statusBar().showMessage(msg, 3000)
        else:
            self._status_msg = msg

    def _execute_silent_save(self, file_path: str) -> bool:
        \"\"\"Helper to save project state to file without modifying current_file_path or pushing undo state.\"\"\"
        try:
            stack = self.model.get_surface_stack()
            if hasattr(self, 'system_config_dock') and self.system_config_dock is not None:
                if hasattr(self.system_config_dock, 'sync_from_ui'):
                    self.system_config_dock.sync_from_ui()

            sys_cfg = getattr(self, 'system_config', None)
            if sys_cfg is not None:
                self.model.system_config = sys_cfg
                stack.system_config = sys_cfg

            wavelengths = None
            if sys_cfg is not None and sys_cfg.wavelengths:
                wavelengths = [w.wavelength_um for w in sys_cfg.wavelengths]

            source_cfg = getattr(self, 'source_config', None)
            if hasattr(self, 'source_editor') and self.source_editor is not None:
                if hasattr(self.source_editor, 'get_source_config'):
                    source_cfg = self.source_editor.get_source_config()

            opt_cfg = None
            if hasattr(self, 'optimizer_dock') and self.optimizer_dock is not None:
                if hasattr(self.optimizer_dock, 'get_config'):
                    opt_cfg = self.optimizer_dock.get_config()

            analysis_cfg = None
            if hasattr(self, 'analysis_tabs') and self.analysis_tabs is not None:
                if hasattr(self.analysis_tabs, 'get_analysis_config'):
                    analysis_cfg = self.analysis_tabs.get_analysis_config()

            if source_cfg is not None:
                stack.source_config = source_cfg

            stack.save_to_json(
                filepath=file_path,
                wavelengths=wavelengths,
                system_config=sys_cfg,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg
            )
            return True
        except Exception as e:
            print(f"[MainWindow] Silent save failed: {e}")
            return False

    def on_autosave(self, manual_trigger: bool = False) -> Optional[str]:
        \"\"\"
        Periodically auto-saves the current project in the background without interrupting the user.
        - If current_file_path is set: saves directly and archives a snapshot to .history and .autosave.
        - If untitled (current_file_path is None): saves to a dedicated recovery autosave file
          (.autosave/autosave_untitled.json) without popping up a blocking file dialog.
        \"\"\"
        if not self.autosave_enabled and not manual_trigger:
            return None

        # Check if there is anything to save
        stack = self.model.get_surface_stack() if hasattr(self, 'model') else None
        if stack is None or not hasattr(stack, 'surfaces') or len(stack.surfaces) == 0:
            return None

        # If project is not dirty and not manually triggered, skip duplicate saving
        if not self.is_dirty and not manual_trigger and self.last_autosave_path and os.path.exists(self.last_autosave_path):
            return None

        try:
            now_dt = datetime.now()
            time_str = now_dt.strftime("%H:%M:%S")

            if self.current_file_path:
                abs_path = os.path.abspath(self.current_file_path)
                proj_dir = os.path.dirname(abs_path)
                base_name = os.path.basename(abs_path)
                stem, ext = os.path.splitext(base_name)
                ext = ext or ".json"

                autosave_dir = os.path.join(proj_dir, ".autosave")
                os.makedirs(autosave_dir, exist_ok=True)
                autosave_file = os.path.join(autosave_dir, f"{stem}_autosave{ext}")

                self._execute_save(self.current_file_path, is_autosave=True)

                try:
                    shutil.copy2(self.current_file_path, autosave_file)
                except Exception:
                    pass

                self.last_autosave_path = self.current_file_path
                self.last_autosave_time = now_dt
                self.is_dirty = False
                self._update_window_title()

                msg = f"⏰ [Auto-Save] Saved project to {base_name} ({time_str})"
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(msg, 5000)
                else:
                    self._status_msg = msg
                return self.current_file_path
            else:
                autosave_dir = os.path.join(os.getcwd(), ".autosave")
                os.makedirs(autosave_dir, exist_ok=True)
                recovery_file = os.path.join(autosave_dir, "autosave_untitled.json")

                success = self._execute_silent_save(recovery_file)
                if not success:
                    return None

                self.last_autosave_path = recovery_file
                self.last_autosave_time = now_dt
                self.is_dirty = False
                self._update_window_title()

                msg = f"⏰ [Auto-Save] Recovery backup saved to .autosave/autosave_untitled.json ({time_str})"
                if QT_AVAILABLE and hasattr(self, 'statusBar'):
                    self.statusBar().showMessage(msg, 5000)
                else:
                    self._status_msg = msg
                return recovery_file
        except Exception as e:
            err_msg = f"⚠️ [Auto-Save] Failed to auto-save: {e}"
            if QT_AVAILABLE and hasattr(self, 'statusBar'):
                self.statusBar().showMessage(err_msg, 5000)
            print(f"[MainWindow] {err_msg}")
            return None

    def on_recover_autosave(self) -> bool:
        \"\"\"Search for and restore the most recent autosaved project recovery snapshot.\"\"\"
        candidates = []
        if self.current_file_path:
            p_dir = os.path.dirname(os.path.abspath(self.current_file_path))
            stem = os.path.splitext(os.path.basename(self.current_file_path))[0]
            auto_file = os.path.join(p_dir, ".autosave", f"{stem}_autosave.json")
            if os.path.exists(auto_file):
                candidates.append(auto_file)

        cwd_untitled = os.path.join(os.getcwd(), ".autosave", "autosave_untitled.json")
        if os.path.exists(cwd_untitled):
            candidates.append(cwd_untitled)

        if self.current_file_path:
            v_list = VersionManager.get_archived_versions(self.current_file_path)
            if v_list:
                candidates.append(v_list[-1]["path"])

        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = candidates[0]
            mtime_str = datetime.fromtimestamp(os.path.getmtime(chosen)).strftime("%Y-%m-%d %H:%M:%S")
            if QT_AVAILABLE:
                reply = QMessageBox.question(
                    self,
                    "Recover Auto-Saved Project",
                    f"Found recent auto-saved snapshot:\\n\\n{chosen}\\nModified: {mtime_str}\\n\\nWould you like to recover this design?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    return self.on_load_project(file_path=chosen)
                return False
            else:
                return self.on_load_project(file_path=chosen)
        else:
            if QT_AVAILABLE:
                QMessageBox.information(self, "No Auto-Save Found", "No auto-save recovery snapshots were found for this workspace.")
            return False

    def _execute_save(self, file_path: str, is_autosave: bool = False) -> bool:
        try:
            stack = self.model.get_surface_stack()"""

    if target_execute_save in content and "def on_autosave" not in content:
        content = content.replace(target_execute_save, replacement_execute_save, 1)
    else:
        print(f"Warning: target_execute_save not found or already patched in {file_path}")

    # 6. Update VersionManager.archive_version description and undo_manager.push_state inside _execute_save
    old_archive_call = """            # Automatically create version archive copy and log past modifications
            archive_path = VersionManager.archive_version(
                file_path=file_path,
                stack=stack,
                system_config=sys_cfg,
                wavelengths=wavelengths,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg,
                description=f"Auto-saved snapshot of {base_name}"
            )

            if QT_AVAILABLE and hasattr(self, 'setWindowTitle'):
                self.setWindowTitle(f"2D Optics Simulation & Design Studio - [{base_name}]")

            if hasattr(self, 'undo_manager') and self.undo_manager is not None:
                self.undo_manager.push_state(stack, sys_cfg, f"Saved Project ({base_name})")

            msg = f"Saved project: {base_name}" + (f" (Archived version to .history)" if archive_path else "")
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg)
            else:
                self._status_msg = msg"""

    new_archive_call = """            # Automatically create version archive copy and log past modifications
            archive_path = VersionManager.archive_version(
                file_path=file_path,
                stack=stack,
                system_config=sys_cfg,
                wavelengths=wavelengths,
                optimizer_config=opt_cfg,
                analysis_config=analysis_cfg,
                description=f"{'Auto-saved' if is_autosave else 'Saved'} snapshot of {base_name}"
            )

            self._update_window_title()

            if hasattr(self, 'undo_manager') and self.undo_manager is not None and not is_autosave:
                self.undo_manager.push_state(stack, sys_cfg, f"Saved Project ({base_name})")

            prefix = "⏰ [Auto-Save] Saved project: " if is_autosave else "Saved project: "
            msg = f"{prefix}{base_name}" + (f" (Archived version to .history)" if archive_path else "")
            if QT_AVAILABLE:
                self.statusBar().showMessage(msg, 5000 if is_autosave else 3000)
            else:
                self._status_msg = msg"""

    if old_archive_call in content:
        content = content.replace(old_archive_call, new_archive_call, 1)
    else:
        print(f"Warning: old_archive_call not found in {file_path}")

    # Also update on_load_project to call _update_window_title
    old_load_title = """            self.current_file_path = file_path
            self.is_dirty = False
            base_name = os.path.basename(file_path)
            if QT_AVAILABLE and hasattr(self, 'setWindowTitle'):
                self.setWindowTitle(f"2D Optics Simulation & Design Studio - [{base_name}]")"""

    new_load_title = """            self.current_file_path = file_path
            self.is_dirty = False
            base_name = os.path.basename(file_path)
            self._update_window_title()"""

    if old_load_title in content:
        content = content.replace(old_load_title, new_load_title, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully patched {file_path}")

if __name__ == "__main__":
    patch_main_window("v2/gui/main_window.py")
    patch_main_window("optics_app/gui/main_window.py")
