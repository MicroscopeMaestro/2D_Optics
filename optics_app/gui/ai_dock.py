"""
AI Assistant, Zemax System Design Panel, Matrix Propagation & Q-Factor Dock.

Provides an integrated AI interface for:
1. Natural language command interpretation & execution (NVIDIA Nemotron & Google Gemma).
2. Live Zemax-style system design parameters (wavelengths, fields, apertures, track).
3. Paraxial matrix propagation ray tracing and optical Q-factor calculations.
"""

import os
import json
import math
from typing import Dict, Any, Optional, List, Tuple, Union

try:
    from optics_app.gui.qt_compat import (
        QtWidgets, QtCore, QtGui, Qt, QThread, pyqtSignal, QT_AVAILABLE
    )
except ImportError:
    from optics_app.gui.qt_compat import (
        QtWidgets, QtCore, QtGui, Qt, QThread, pyqtSignal, QT_AVAILABLE
    )

QDockWidget = getattr(QtWidgets, "QDockWidget", object)
QWidget = getattr(QtWidgets, "QWidget", object)
QVBoxLayout = getattr(QtWidgets, "QVBoxLayout", object)
QHBoxLayout = getattr(QtWidgets, "QHBoxLayout", object)
QFormLayout = getattr(QtWidgets, "QFormLayout", object)
QTabWidget = getattr(QtWidgets, "QTabWidget", object)
QTextEdit = getattr(QtWidgets, "QTextEdit", object)
QLineEdit = getattr(QtWidgets, "QLineEdit", object)
QPushButton = getattr(QtWidgets, "QPushButton", object)
QComboBox = getattr(QtWidgets, "QComboBox", object)
QLabel = getattr(QtWidgets, "QLabel", object)
QDoubleSpinBox = getattr(QtWidgets, "QDoubleSpinBox", object)
QSpinBox = getattr(QtWidgets, "QSpinBox", object)
QGroupBox = getattr(QtWidgets, "QGroupBox", object)
QTableWidget = getattr(QtWidgets, "QTableWidget", object)
QTableWidgetItem = getattr(QtWidgets, "QTableWidgetItem", object)
QHeaderView = getattr(QtWidgets, "QHeaderView", object)
QScrollArea = getattr(QtWidgets, "QScrollArea", object)
QMessageBox = getattr(QtWidgets, "QMessageBox", object)
QDialog = getattr(QtWidgets, "QDialog", object)

try:
    from optics_app.ai.nemotron_client import NemotronClient, GemmaClient, CloudAiClient, ModelTier, ModelFamily
    from optics_app.ai.agent import NemotronAgent
    from optics_app.ai.tools import get_default_tool_registry, set_active_optical_system, get_active_optical_system, interpret_and_execute_command
    from optics_app.engine.matrix_tracer import MatrixTracer, calculate_cavity_q_factor, calculate_spectrometer_q_factor
except ImportError:
    from optics_app.ai.nemotron_client import NemotronClient, GemmaClient, CloudAiClient, ModelTier, ModelFamily
    from optics_app.ai.agent import NemotronAgent
    from optics_app.ai.tools import get_default_tool_registry, set_active_optical_system, get_active_optical_system, interpret_and_execute_command
    from optics_app.engine.matrix_tracer import MatrixTracer, calculate_cavity_q_factor, calculate_spectrometer_q_factor



import logging
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.expanduser("~/.2d_optics_ai.json")

def load_saved_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load AI config: {e}")
    return {}

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save AI config: {e}")

class CommandWorker(QThread if QT_AVAILABLE else object):
    if QT_AVAILABLE:
        finished = pyqtSignal(object)
        error = pyqtSignal(str)

    def __init__(self, text, system, client=None, file_path=None, agent=None, agentic_mode=False):
        super().__init__()
        self.text = text
        self.system = system
        self.client = client
        self.file_path = file_path
        self.agent = agent
        self.agentic_mode = bool(agentic_mode)

    def execute(self):
        """Execute one command and normalize single-action and agentic results."""
        if self.agentic_mode and self.agent is not None:
            agent_result = self.agent.run(self.text, verbose=False)
            steps = agent_result.get("steps", [])
            return {
                "command": self.text,
                "assistant_reply": agent_result.get("final_answer", ""),
                "tool_calls": [],
                "executed_results": [
                    {
                        "tool": step.get("tool", "unknown_tool"),
                        "arguments": step.get("arguments", {}),
                        "result": step.get("output", ""),
                    }
                    for step in steps
                ],
                "success": bool(agent_result.get("success")),
                "agentic_mode": True,
                "iterations": int(agent_result.get("iterations", 0)),
            }
        result = interpret_and_execute_command(
            self.text,
            system=self.system,
            client=self.client,
            json_path=self.file_path,
        )
        result["agentic_mode"] = False
        return result

    def run(self):
        try:
            res = self.execute()
            if QT_AVAILABLE:
                self.finished.emit(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if QT_AVAILABLE:
                self.error.emit(str(e))


class ConfigDialog(QDialog if QT_AVAILABLE else object):
    def __init__(self, parent=None):
        if not QT_AVAILABLE:
            return
        super().__init__(parent)
        self.setWindowTitle("Cloud AI Configuration (Nemotron & Gemma)")
        self.layout = QFormLayout(self)

        # Provider / Family
        self.family_combo = QComboBox()
        self.family_combo.addItems([ModelFamily.NEMOTRON.value, ModelFamily.GEMMA.value, "Custom OpenAI-Compatible"])
        self.layout.addRow("AI Provider / Family:", self.family_combo)

        # Base API URL
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("https://integrate.api.nvidia.com/v1 (Leave blank for default)")
        self.layout.addRow("Base API URL:", self.api_base_input)

        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("NVIDIA / Google / Custom Cloud API Key")
        self.layout.addRow("API Key:", self.api_key_input)

        # Model Name
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("nvidia/nemotron-4-340b-instruct (Leave blank for tier default)")
        self.layout.addRow("Model Name:", self.model_name_input)

        # Tier
        self.tier_combo = QComboBox()
        self.tier_combo.addItems([ModelTier.HIGH.value, ModelTier.MEDIUM.value])
        self.layout.addRow("Fallback Model Tier:", self.tier_combo)

        # Demo mode toggle
        self.demo_mode_combo = QComboBox()
        self.demo_mode_combo.addItems(["Auto (Live if key set, Demo fallback)", "Always Demo Mode", "Live API Only"])
        self.layout.addRow("Integrity Mode:", self.demo_mode_combo)

        # Custom system prompt
        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setMaximumHeight(80)
        self.layout.addRow("System Prompt:", self.system_prompt_input)

        self._load_current()

        # Action Buttons
        btn_layout = QHBoxLayout()
        
        test_btn = QPushButton("🔌 Test Connection")
        test_btn.clicked.connect(self.test_connection)
        btn_layout.addWidget(test_btn)
        
        apply_btn = QPushButton("Apply Configuration")
        apply_btn.clicked.connect(self.accept)
        btn_layout.addWidget(apply_btn)
        
        self.layout.addRow(btn_layout)

    def _load_current(self):
        cfg = load_saved_config()
        if not cfg: return
        
        family = cfg.get("family", ModelFamily.NEMOTRON.value)
        idx = self.family_combo.findText(family)
        if idx >= 0: self.family_combo.setCurrentIndex(idx)
        
        self.api_base_input.setText(cfg.get("base_url", ""))
        self.api_key_input.setText(cfg.get("api_key", ""))
        self.model_name_input.setText(cfg.get("model", ""))
        
        tier = cfg.get("tier", ModelTier.HIGH.value)
        idx = self.tier_combo.findText(tier)
        if idx >= 0: self.tier_combo.setCurrentIndex(idx)
        
        if cfg.get("demo_mode", False):
            self.demo_mode_combo.setCurrentIndex(1)
        
        self.system_prompt_input.setText(cfg.get("prompt", ""))

    def test_connection(self):
        cfg = self.get_config()
        family = cfg.get("family", ModelFamily.NEMOTRON.value)
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "")
        model = cfg.get("model", "")
        demo = cfg.get("demo_mode", False)
        
        try:
            if family.lower() == ModelFamily.GEMMA.value.lower():
                client = GemmaClient(api_key=api_key or None, model=model or None, base_url=base_url or None, demo_mode=demo)
            else:
                client = NemotronClient(api_key=api_key or None, model=model or None, base_url=base_url or None, demo_mode=demo)
            
            res = client.chat_completion([{"role": "user", "content": "ping"}])
            
            if res and isinstance(res, str):
                QMessageBox.information(self, "Connection Test", f"Success! API responded:\n{res[:100]}")
            else:
                QMessageBox.warning(self, "Connection Test", "Success, but empty response.")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n{str(e)}")

    def get_config(self) -> dict:
        return {
            "family": self.family_combo.currentText(),
            "base_url": self.api_base_input.text().strip(),
            "api_key": self.api_key_input.text().strip(),
            "model": self.model_name_input.text().strip(),
            "tier": self.tier_combo.currentText(),
            "demo_mode": self.demo_mode_combo.currentIndex() == 1,
            "prompt": self.system_prompt_input.toPlainText()
        }

class AiChatDock(QDockWidget if QT_AVAILABLE else object):
    """
    AI Assistant & Zemax System Design Panel with Matrix Propagation & Q-Factor.
    """
    def __init__(self, main_window=None, parent=None):
        if QT_AVAILABLE:
            super().__init__("AI System Design & Assistant", parent)
            self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)

        self.main_window = main_window
        self.client = None
        self.agent = None
        self.dark_mode: bool = False

        if QT_AVAILABLE:
            self.init_ui()
            self._update_theme_styles()
            self.init_default_ai()

    def set_dark_mode(self, enabled: bool) -> None:
        """Switch between dark and light background themes."""
        self.dark_mode = bool(enabled)
        self._update_theme_styles()
        self.update_current_file_label()

    def _update_theme_styles(self) -> None:
        """Update styled elements to match the current theme."""
        if not QT_AVAILABLE:
            return
        is_dark = getattr(self, 'dark_mode', False)
        if hasattr(self, 'label_abcd') and self.label_abcd:
            if is_dark:
                self.label_abcd.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background: #1e293b; color: #f1f5f9; "
                    "padding: 6px; border-radius: 4px; border: 1px solid #334155;"
                )
            else:
                self.label_abcd.setStyleSheet(
                    "font-family: monospace; font-size: 11px; background: #f0f0f0; color: #0f172a; "
                    "padding: 6px; border-radius: 4px; border: 1px solid #cbd5e1;"
                )
        if hasattr(self, 'label_q_result') and self.label_q_result:
            if is_dark:
                self.label_q_result.setStyleSheet(
                    "font-weight: bold; color: #38bdf8; background: #0f172a; padding: 6px; "
                    "border-radius: 4px; border: 1px solid #1e293b;"
                )
            else:
                self.label_q_result.setStyleSheet(
                    "font-weight: bold; color: #1a5276; background: #eaf2f8; padding: 6px; "
                    "border-radius: 4px; border: 1px solid #90caf9;"
                )

    def init_default_ai(self):
        """Initialize AI agent in demo/live mode."""
        try:
            cfg = load_saved_config()
            if cfg:
                self.apply_config(cfg)
            else:
                self.client = NemotronClient()
                tools = get_default_tool_registry()
                self.agent = NemotronAgent(client=self.client, tool_registry=tools)
        except Exception as e:
            logger.debug(f"AI auto-init: {e}")

    def test_main_connection(self):
        if not self.client:
            QMessageBox.warning(self, "No Client", "AI Client is not initialized. Please configure it first.")
            return
        try:
            res = self.client.chat_completion([{"role": "user", "content": "ping"}])
            if res and isinstance(res, str):
                QMessageBox.information(self, "Connection Test", f"Success! AI responded:\n{res[:100]}")
            else:
                QMessageBox.warning(self, "Connection Test", "Success, but empty response.")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect:\n{str(e)}")

    def init_ui(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # -------------------------------------------------------------------
        # TAB 1: AI Assistant & Natural Language Commands
        # -------------------------------------------------------------------
        tab_ai = QWidget()
        tab_ai_layout = QVBoxLayout(tab_ai)

        # Top Bar (Model / Tier / Config)
        top_bar = QHBoxLayout()
        self.model_label = QLabel("🤖 <b>Model:</b> Nemotron 70B (Demo Ready)")
        top_bar.addWidget(self.model_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Agentic (Multi-Step)", "agentic")
        self.mode_combo.addItem("Single Action", "single")
        self.mode_combo.setToolTip(
            "Agentic mode can plan and execute several application tools in sequence. "
            "Single Action executes one interpreted command."
        )
        top_bar.addWidget(self.mode_combo)
        top_bar.addStretch()

        self.test_conn_btn = QPushButton("🔌 Test API Connection")
        self.test_conn_btn.clicked.connect(self.test_main_connection)
        top_bar.addWidget(self.test_conn_btn)

        self.config_btn = QPushButton("⚙️ Config")
        self.config_btn.clicked.connect(self.open_config)
        top_bar.addWidget(self.config_btn)
        tab_ai_layout.addLayout(top_bar)

        # Current File Status
        self.current_file_label = QLabel("📄 Current File: <i>None</i>")
        self.current_file_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
        tab_ai_layout.addWidget(self.current_file_label)

        # Chat Stream
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        tab_ai_layout.addWidget(self.chat_history)

        # Input Row
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText(
            "Try: replace Camera Lens with a 50 mm Thorlabs achromat; optimize focus at Detector"
        )
        self.input_box.returnPressed.connect(self.send_message)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)
        tab_ai_layout.addLayout(input_layout)

        self.tabs.addTab(tab_ai, "AI Assistant")

        # -------------------------------------------------------------------
        # TAB 2: Zemax System Design Parameters
        # -------------------------------------------------------------------
        tab_zemax = QWidget()
        zemax_layout = QVBoxLayout(tab_zemax)

        # Wavelengths Group
        wl_group = QGroupBox("Wavelengths (µm)")
        wl_form = QFormLayout(wl_group)

        self.spin_primary_wl = QDoubleSpinBox()
        self.spin_primary_wl.setRange(0.2, 20.0)
        self.spin_primary_wl.setDecimals(4)
        self.spin_primary_wl.setValue(1.064)
        self.spin_primary_wl.setSingleStep(0.01)
        wl_form.addRow("Primary Wavelength:", self.spin_primary_wl)

        self.edit_wavelength_list = QLineEdit("1.100, 1.225, 1.350")
        wl_form.addRow("Active Wavelengths:", self.edit_wavelength_list)
        zemax_layout.addWidget(wl_group)

        # Fields Group
        field_group = QGroupBox("Field Data (Zemax Conventions)")
        field_form = QFormLayout(field_group)

        self.spin_field_angle = QDoubleSpinBox()
        self.spin_field_angle.setRange(-45.0, 45.0)
        self.spin_field_angle.setValue(0.0)
        self.spin_field_angle.setSuffix("°")
        field_form.addRow("Field Angle Y:", self.spin_field_angle)

        self.spin_field_height = QDoubleSpinBox()
        self.spin_field_height.setRange(0.0, 100.0)
        self.spin_field_height.setValue(0.0)
        self.spin_field_height.setSuffix(" mm")
        field_form.addRow("Field Height:", self.spin_field_height)
        zemax_layout.addWidget(field_group)

        # Apertures Group
        ap_group = QGroupBox("Apertures & Pupil")
        ap_form = QFormLayout(ap_group)

        self.spin_epd = QDoubleSpinBox()
        self.spin_epd.setRange(0.1, 200.0)
        self.spin_epd.setValue(10.0)
        self.spin_epd.setSuffix(" mm")
        ap_form.addRow("Entrance Pupil (EPD):", self.spin_epd)

        self.spin_na = QDoubleSpinBox()
        self.spin_na.setRange(0.01, 0.95)
        self.spin_na.setDecimals(3)
        self.spin_na.setValue(0.12)
        ap_form.addRow("Numerical Aperture (NA):", self.spin_na)

        self.label_total_track = QLabel("Total Track: -- mm")
        ap_form.addRow("System Physical Track:", self.label_total_track)
        zemax_layout.addWidget(ap_group)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_apply_zemax = QPushButton("Apply to System")
        self.btn_apply_zemax.setStyleSheet("font-weight: bold; background-color: #2b5b84; color: white;")
        self.btn_apply_zemax.clicked.connect(self.apply_zemax_parameters)

        self.btn_sync_zemax = QPushButton("Sync from Active System")
        self.btn_sync_zemax.clicked.connect(self.sync_from_system)

        btn_box.addWidget(self.btn_sync_zemax)
        btn_box.addWidget(self.btn_apply_zemax)
        zemax_layout.addLayout(btn_box)
        zemax_layout.addStretch()

        self.tabs.addTab(tab_zemax, "Zemax Parameters")

        # -------------------------------------------------------------------
        # TAB 3: Matrix Propagation Ray Tracing & Q-Factor
        # -------------------------------------------------------------------
        tab_matrix = QWidget()
        matrix_layout = QVBoxLayout(tab_matrix)

        # ABCD Matrix Display
        abcd_box = QGroupBox("Paraxial Ray Transfer Matrix [A B; C D]")
        abcd_layout = QVBoxLayout(abcd_box)
        self.label_abcd = QLabel("A = 1.0000    B = 0.0000 mm\nC = 0.0000 mm⁻¹ D = 1.0000\nDeterminant = 1.0000")
        self.label_abcd.setStyleSheet("font-family: monospace; font-size: 11px; background: #f0f0f0; padding: 6px;")
        abcd_layout.addWidget(self.label_abcd)

        self.label_cardinal = QLabel("EFL: -- mm | BFL: -- mm | FFL: -- mm\nStability m: -- (Stable: Yes)")
        abcd_layout.addWidget(self.label_cardinal)
        matrix_layout.addWidget(abcd_box)

        # Ray Propagation Test
        ray_box = QGroupBox("Paraxial Ray Propagation")
        ray_form = QFormLayout(ray_box)

        self.spin_ray_y = QDoubleSpinBox()
        self.spin_ray_y.setRange(-50.0, 50.0)
        self.spin_ray_y.setValue(1.0)
        self.spin_ray_y.setSuffix(" mm")
        ray_form.addRow("Input Height y0:", self.spin_ray_y)

        self.spin_ray_u = QDoubleSpinBox()
        self.spin_ray_u.setRange(-1.0, 1.0)
        self.spin_ray_u.setDecimals(4)
        self.spin_ray_u.setValue(0.0)
        self.spin_ray_u.setSuffix(" rad")
        ray_form.addRow("Input Slope u0:", self.spin_ray_u)

        self.label_ray_out = QLabel("Output: y_out = -- mm, u_out = -- rad")
        ray_form.addRow("Result:", self.label_ray_out)

        btn_trace_matrix = QPushButton("Trace Paraxial Ray")
        btn_trace_matrix.clicked.connect(self.run_matrix_trace)
        ray_form.addRow(btn_trace_matrix)
        matrix_layout.addWidget(ray_box)

        # Q-Factor Calculation Box
        q_box = QGroupBox("Optical Quality Factor (Q-Factor)")
        q_form = QFormLayout(q_box)

        self.combo_q_mode = QComboBox()
        self.combo_q_mode.addItems(["Optical Resonator / Cavity", "Grating Spectrometer"])
        q_form.addRow("Q-Factor Mode:", self.combo_q_mode)

        self.spin_reflectivity = QDoubleSpinBox()
        self.spin_reflectivity.setRange(0.5, 0.99999)
        self.spin_reflectivity.setDecimals(4)
        self.spin_reflectivity.setValue(0.9900)
        q_form.addRow("Mirror Reflectivity R:", self.spin_reflectivity)

        self.label_q_result = QLabel("Q-Factor: --\nFinesse: --\nLinewidth: -- Hz")
        self.label_q_result.setStyleSheet("font-weight: bold; color: #1a5276; background: #eaf2f8; padding: 6px;")
        q_form.addRow("Q-Metrics:", self.label_q_result)

        btn_calc_q = QPushButton("Compute Q-Factor & Update")
        btn_calc_q.clicked.connect(self.run_q_factor_calc)
        q_form.addRow(btn_calc_q)

        matrix_layout.addWidget(q_box)
        matrix_layout.addStretch()

        self.tabs.addTab(tab_matrix, "Matrix Ray Trace & Q-Factor")

        self.setWidget(container)
        self.setMinimumWidth(320)
        
        # Initialize current file label
        self.update_current_file_label()

    def open_config(self):
        if not QT_AVAILABLE:
            return
        dialog = ConfigDialog(self)
        if dialog.exec_():
            cfg = dialog.get_config()
            save_config(cfg)
            self.apply_config(cfg)

    def apply_config(self, cfg: dict):
        family = cfg.get("family", ModelFamily.NEMOTRON.value)
        tier = cfg.get("tier", ModelTier.HIGH.value)
        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "")
        model = cfg.get("model", "")
        demo = cfg.get("demo_mode", False)
        prompt = cfg.get("prompt", "")

        if api_key:
            os.environ["NVIDIA_API_KEY"] = api_key
            os.environ["GEMMA_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key

        try:
            if family.lower() == ModelFamily.GEMMA.value.lower():
                self.client = GemmaClient(api_key=api_key or None, model=model or None, base_url=base_url or None, tier=tier, demo_mode=demo)
            else:
                self.client = NemotronClient(api_key=api_key or None, model=model or None, base_url=base_url or None, tier=tier, demo_mode=demo)

            tools = get_default_tool_registry()
            kwargs = {"client": self.client, "tool_registry": tools}
            if prompt:
                kwargs["system_prompt"] = prompt

            self.agent = NemotronAgent(**kwargs)
            self.model_label.setText(f"🤖 <b>Model:</b> {self.client.model} ({tier})")
            self.chat_history.append(f"<b>System:</b> Connected to {family.capitalize()} AI ({tier}).")
        except Exception as e:
            self.chat_history.append(f"<font color='red'><b>Config Error:</b> {e}</font>")

    def quick_action(self, prompt: str):
        self.input_box.setText(prompt)
        self.send_message()

    def quick_action_for_selected(self, prompt_template: str):
        """Run an element action against the currently selected LDE surface."""
        table = getattr(self.main_window, "table_view", None) if self.main_window else None
        index = table.currentIndex() if table is not None and hasattr(table, "currentIndex") else None
        row = index.row() if index is not None and hasattr(index, "isValid") and index.isValid() else -1
        if row < 0:
            self.chat_history.append(
                "<font color='orange'><b>Action warning:</b> Select a lens surface in the LDE first.</font>"
            )
            return
        self.quick_action(prompt_template.format(index=row))

    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.chat_history.append(f"<b>You:</b> {text}")
        self.input_box.clear()
        
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(False)
        self.input_box.setEnabled(False)

        # Update active optical system pointer
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        # Get current file path for auto-saving
        file_path = getattr(self.main_window, "current_file_path", None) if self.main_window else None
        if system is not None:
            set_active_optical_system(system)
            # AI tools mutate the shared stack in a worker thread. Capture the
            # pre-action state here so the completed action is one undoable step.
            model = getattr(self.main_window, "model", None) if self.main_window else None
            if model is not None and hasattr(model, "_push_undo"):
                model._push_undo(f"Before AI command: {text[:60]}")

        # Update current file label
        self.update_current_file_label(file_path)

        # Execute in background thread to prevent UI freeze
        agentic_mode = (
            hasattr(self, "mode_combo") and self.mode_combo.currentData() == "agentic"
        )
        self.chat_history.append(
            "<font color='#555'><i>Agentic multi-step execution enabled.</i></font>"
            if agentic_mode else
            "<font color='#555'><i>Single-action execution enabled.</i></font>"
        )
        self.worker = CommandWorker(
            text,
            system,
            client=self.client,
            file_path=file_path,
            agent=self.agent,
            agentic_mode=agentic_mode,
        )
        self.worker.finished.connect(self.on_command_finished)
        self.worker.error.connect(self.on_command_error)
        self.worker.start()

    def on_command_finished(self, res):
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)
        
        system = getattr(self.worker, 'system', None)
        file_path = getattr(self.worker, 'file_path', None)
        
        # Write back changes to synchronize wavelengths, fields, apertures
        if self.main_window and system:
            self.main_window.current_system = system
            
        # Auto-save to current file if path exists
        if file_path and system and hasattr(system, "save_to_json"):
            try:
                system.save_to_json(file_path)
                self.chat_history.append(f"<font color='blue'><b>Auto-saved:</b> {file_path}</font>")
                self.update_current_file_label(file_path)
            except Exception as e:
                self.chat_history.append(f"<font color='orange'><b>Save warning:</b> {e}</font>")
            
        if res.get("executed_results"):
            for step in res["executed_results"]:
                self.chat_history.append(f"<font color='green'><b>Action:</b> {step['tool']}</font>")
                self.chat_history.append(f"<pre>{step['result']}</pre>")
            if hasattr(self.main_window, "update_all_views"):
                # The pre-action snapshot was captured before the worker ran;
                # pushing the post-action state here would make the first Undo a no-op.
                self.main_window.update_all_views(action_name="AI Action", record_history=False)
        reply = res.get('assistant_reply', '')
        if reply:
            prefix = "Agent" if res.get("agentic_mode") else "AI"
            iterations = f" ({res.get('iterations')} iterations)" if res.get("agentic_mode") else ""
            self.chat_history.append(f"<b>{prefix}{iterations}:</b> {reply}")
        elif not res.get("executed_results"):
            self.chat_history.append("<b>AI:</b> Understood.")

    def on_command_error(self, err_msg):
        if hasattr(self, 'send_btn'):
            self.send_btn.setEnabled(True)
        self.input_box.setEnabled(True)
        self.chat_history.append(f"<font color='red'><b>Error:</b> {err_msg}</font>")

    def save_current_file(self):
        """Manually save the current optical system to the current file."""
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        file_path = getattr(self.main_window, "current_file_path", None) if self.main_window else None
        
        if not system:
            self.chat_history.append("<font color='red'><b>Error:</b> No active optical system to save.</font>")
            return
            
        if not file_path:
            self.chat_history.append("<font color='red'><b>Error:</b> No current file path. Use 'Save As' from File menu first.</font>")
            return
            
        try:
            system.save_to_json(file_path)
            self.chat_history.append(f"<font color='green'><b>Saved:</b> {file_path}</font>")
            self.update_current_file_label(file_path)
        except Exception as e:
            self.chat_history.append(f"<font color='red'><b>Save Error:</b> {e}</font>")

    def update_current_file_label(self, file_path: str = None):
        """Update the current file label in the UI."""
        if file_path is None:
            file_path = getattr(self.main_window, "current_file_path", None) if self.main_window else None
        
        is_dark = getattr(self, 'dark_mode', False)
        if file_path:
            import os
            filename = os.path.basename(file_path)
            c = "#4ade80" if is_dark else "#27ae60"
            self.current_file_label.setText(f"📄 Current File: <b>{filename}</b>")
            self.current_file_label.setStyleSheet(f"color: {c}; font-size: 11px; padding: 2px;")
        else:
            c = "#94a3b8" if is_dark else "#666"
            self.current_file_label.setText("📄 Current File: <i>None</i>")
            self.current_file_label.setStyleSheet(f"color: {c}; font-size: 11px; padding: 2px;")

    def sync_from_system(self):
        """Read parameters from active system into GUI controls."""
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        if not system:
            return

        # Wavelengths
        if hasattr(system, "primary_wavelength"):
            self.spin_primary_wl.setValue(float(system.primary_wavelength))
        if hasattr(system, "wavelengths") and system.wavelengths:
            self.edit_wavelength_list.setText(", ".join(f"{w:.4f}" for w in system.wavelengths))

        # Apertures
        if hasattr(system, "source_config") and system.source_config:
            cfg = system.source_config
            br = float(cfg.get("beam_radius", 5.0)) if isinstance(cfg, dict) else float(getattr(cfg, "beam_radius", 5.0))
            self.spin_epd.setValue(br * 2.0)
            na = float(cfg.get("numerical_aperture", 0.12)) if isinstance(cfg, dict) else float(getattr(cfg, "numerical_aperture", 0.12))
            self.spin_na.setValue(na)

        # Fields
        if hasattr(system, "field_angle") and system.field_angle is not None:
            self.spin_field_angle.setValue(float(system.field_angle))
        elif hasattr(system, "system_config") and system.system_config:
            cfg = system.system_config
            if isinstance(cfg, dict):
                self.spin_field_angle.setValue(float(cfg.get("field_angle", 0.0)))
                self.spin_field_height.setValue(float(cfg.get("field_height", 0.0)))
            elif hasattr(cfg, "fields") and cfg.fields:
                f0 = cfg.fields[0]
                ang = getattr(f0, "angle_y_deg", getattr(f0, "y", 0.0))
                h = getattr(f0, "height_y", getattr(f0, "y", 0.0))
                self.spin_field_angle.setValue(float(ang))
                self.spin_field_height.setValue(float(h))

        # Track
        surfs = getattr(system, "surfaces", [])
        total_len = sum(getattr(s, "thickness", 0.0) for s in surfs)
        self.label_total_track.setText(f"Total Track: {total_len:.2f} mm ({len(surfs)} surfaces)")

        # Run matrix trace sync
        self.run_matrix_trace()
        self.run_q_factor_calc()

    def apply_zemax_parameters(self):
        """Apply Zemax parameters from GUI controls to active system."""
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        if not system:
            return

        # Update primary wavelength
        prim_wl = float(self.spin_primary_wl.value())
        system.primary_wavelength = prim_wl

        # Update active wavelengths
        raw_wls = self.edit_wavelength_list.text().split(",")
        parsed_wls = []
        for w in raw_wls:
            try:
                parsed_wls.append(float(w.strip()))
            except ValueError:
                pass
        if parsed_wls:
            system.wavelengths = parsed_wls

        # Update aperture
        epd = float(self.spin_epd.value())
        if hasattr(system, "source_config") and system.source_config:
            if isinstance(system.source_config, dict):
                system.source_config["beam_radius"] = epd / 2.0
                system.source_config["numerical_aperture"] = float(self.spin_na.value())
            else:
                system.source_config.beam_radius = epd / 2.0
                system.source_config.numerical_aperture = float(self.spin_na.value())

        # Update field
        field_ang = float(self.spin_field_angle.value())
        if hasattr(system, "field_angle"):
            system.field_angle = field_ang
        if hasattr(system, "system_config") and system.system_config:
            if isinstance(system.system_config, dict):
                system.system_config["field_angle"] = field_ang
            elif hasattr(system.system_config, "fields") and system.system_config.fields:
                f0 = system.system_config.fields[0]
                f0.angle_y_deg = field_ang
                if hasattr(f0, "y"):
                    f0.y = field_ang

        self.chat_history.append(f"<b>System:</b> Applied Zemax parameters (Primary: {prim_wl} µm, EPD: {epd} mm, Field: {field_ang}°).")
        
        # Write back changes to synchronize wavelengths, fields, apertures across docks
        if self.main_window and system:
            self.main_window.current_system = system
            
        if hasattr(self.main_window, "update_all_views"):
            self.main_window.update_all_views()

        self.run_matrix_trace()
        self.run_q_factor_calc()

    def run_matrix_trace(self):
        """Compute paraxial transfer matrix and ray trace."""
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        if not system:
            return

        tracer = MatrixTracer()
        mat = tracer.build_from_system(system, wavelength_um=float(self.spin_primary_wl.value()))

        self.label_abcd.setText(
            f"A = {mat.A:.4f}    B = {mat.B:.4f} mm\n"
            f"C = {mat.C:.6f} mm⁻¹ D = {mat.D:.4f}\n"
            f"Determinant = {mat.determinant():.6f}"
        )

        efl = mat.effective_focal_length()
        bfl = mat.back_focal_length()
        ffl = mat.front_focal_length()
        m_stab = mat.stability_parameter()
        is_stab = mat.is_stable_resonator()

        efl_str = f"{efl:.2f}" if not math.isinf(efl) else "∞"
        bfl_str = f"{bfl:.2f}" if not math.isinf(bfl) else "∞"
        ffl_str = f"{ffl:.2f}" if not math.isinf(ffl) else "∞"

        self.label_cardinal.setText(
            f"EFL: {efl_str} mm | BFL: {bfl_str} mm | FFL: {ffl_str} mm\n"
            f"Stability m: {m_stab:.4f} (Stable: {'Yes' if is_stab else 'No'})"
        )

        # Propagate ray
        y0 = float(self.spin_ray_y.value())
        u0 = float(self.spin_ray_u.value())
        traj = tracer.trace_ray(y0, u0)
        y_out, u_out = traj[-1]["y"], traj[-1]["u"]

        self.label_ray_out.setText(f"Output: y_out = {y_out:.4f} mm, u_out = {u_out:.6f} rad")

    def run_q_factor_calc(self):
        """Calculate cavity or spectrometer Q-factor."""
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        if not system:
            return

        wl = float(self.spin_primary_wl.value())
        r = float(self.spin_reflectivity.value())
        mode = "spectrometer" if self.combo_q_mode.currentIndex() == 1 else "cavity"

        surfs = getattr(system, "surfaces", [])
        total_len = sum(getattr(s, "thickness", 0.0) for s in surfs)
        if total_len <= 0:
            total_len = 50.0

        if mode == "spectrometer":
            # Grating resolution mode
            g_lines = 600.0
            for s in surfs:
                st_str = str(getattr(s, "surface_type", "")).upper()
                if "GRATING" in st_str or getattr(s, "grating_lines_per_mm", 0.0) > 0 or "GRATING" in getattr(s, "name", "").upper():
                    g_lines = getattr(s, "grating_lines_per_mm", 600.0)
                    break
            beam_w = float(self.spin_epd.value())
            res = calculate_spectrometer_q_factor(wavelength_um=wl, lines_per_mm=g_lines, beam_width_mm=beam_w)
            self.label_q_result.setText(
                f"Spectrometer Resolving Power Q: {res['q_factor']:,.0f}\n"
                f"Spectral Resolution Δλ: {res['spectral_resolution_nm']:.4f} nm\n"
                f"Illuminated Grooves N: {res['illuminated_grooves']:,.0f}"
            )
        else:
            # Resonator cavity mode
            res = calculate_cavity_q_factor(length_mm=total_len, wavelength_um=wl, r1=r, r2=r)
            self.label_q_result.setText(
                f"Cavity Quality Factor Q: {res['q_factor']:.2e}\n"
                f"Finesse F: {res['finesse']:.1f}\n"
                f"Linewidth Δν: {res['linewidth_hz']:.2e} Hz ({res['wavelength_fwhm_pm']:.3f} pm)\n"
                f"Photon Lifetime: {res['photon_lifetime_ns']:.2f} ns"
            )
