"""
Dark and Light Theme Styling System for 2D Optics Simulation Engine.
Provides comprehensive Qt Style Sheets (QSS), dark QPalette configuration,
and color mappings for all UI widgets, tables, toolbars, and matplotlib canvases.
"""

from typing import Optional, Dict, Any

# Curated mapping from light pastel element group tints to sleek dark glass tints for LDE table
DARK_ELEMENT_TINTS: Dict[str, str] = {
    "#E1F5FE": "#0c4a6e",  # Cyan Blue
    "#E8F5E9": "#064e3b",  # Emerald Green
    "#FFF3E0": "#451a03",  # Coral Amber
    "#EDE7F6": "#3b0764",  # Lavender Purple
    "#E0F2F1": "#042f2e",  # Aquamarine Teal
    "#FCE4EC": "#4c0519",  # Rose Pink
    "#FFFDE7": "#422006",  # Golden Sand
    "#E8EAF6": "#1e1b4b",  # Royal Indigo
    "#F5F5F5": "#1e293b",  # Neutral Soft Gray
    "#FFF8E1": "#451a03",  # Mirror Gold
}

# High-contrast vibrant badge text colors for surface component names on dark backgrounds
DARK_BADGE_COLORS: Dict[str, str] = {
    "#01579B": "#38bdf8",  # Vibrant Sky Blue
    "#1B5E20": "#4ade80",  # Vibrant Mint Green
    "#BF360C": "#fb923c",  # Vibrant Amber Orange
    "#311B92": "#c084fc",  # Vibrant Lavender Purple
    "#004D40": "#2dd4bf",  # Vibrant Aquamarine
    "#880E4F": "#f472b6",  # Vibrant Rose Pink
    "#F57F17": "#facc15",  # Vibrant Gold
    "#1A237E": "#818cf8",  # Vibrant Indigo
    "#616161": "#94a3b8",  # Slate Gray
    "#E65100": "#fb923c",  # Vibrant Orange
    "#4A148C": "#c084fc",  # Vibrant Violet
}

DARK_THEME_STYLESHEET = """
/* =========================================================================
   2D Optics Simulation Engine - Professional Dark Theme (Tailwind Slate)
   ========================================================================= */

/* Main Window, Dialogs & General Containers */
QMainWindow, QDialog {
    background-color: #0f172a;
    color: #f1f5f9;
}

QWidget#CentralWidget {
    background-color: #0f172a;
}

/* Menu Bar */
QMenuBar {
    background-color: #0f172a;
    color: #e2e8f0;
    border-bottom: 1px solid #1e293b;
    font-size: 11px;
}
QMenuBar::item {
    background-color: transparent;
    padding: 5px 10px;
    border-radius: 3px;
}
QMenuBar::item:selected, QMenuBar::item:pressed {
    background-color: #1e293b;
    color: #38bdf8;
}

/* Dropdown & Context Menus */
QMenu {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 0px;
    font-size: 11px;
}
QMenu::item {
    padding: 6px 28px 6px 20px;
    border-radius: 3px;
    margin: 1px 4px;
}
QMenu::item:selected {
    background-color: #334155;
    color: #38bdf8;
}
QMenu::item:disabled {
    color: #64748b;
}
QMenu::separator {
    height: 1px;
    background-color: #334155;
    margin: 4px 8px;
}

/* ToolBars */
QToolBar {
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    border-top: none;
    border-left: none;
    border-right: none;
    spacing: 4px;
    padding: 3px 6px;
}
QToolBar::separator {
    width: 1px;
    background-color: #334155;
    margin: 4px 6px;
}
QToolButton {
    background-color: transparent;
    color: #e2e8f0;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
    font-size: 11px;
}
QToolButton:hover {
    background-color: #1e293b;
    border-color: #334155;
    color: #38bdf8;
}
QToolButton:pressed, QToolButton:checked {
    background-color: #0b0f19;
    border-color: #38bdf8;
    color: #38bdf8;
}

/* Status Bar */
QStatusBar {
    background-color: #0b0f19;
    color: #94a3b8;
    border-top: 1px solid #1e293b;
    font-size: 11px;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: #cbd5e1;
}

/* Dock Widgets */
QDockWidget {
    color: #f1f5f9;
    font-weight: 600;
}
QDockWidget::title {
    background-color: #1e293b;
    color: #f1f5f9;
    padding: 6px 10px;
    border-bottom: 1px solid #334155;
    font-size: 11px;
    font-weight: bold;
}
QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
    border-radius: 3px;
}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #334155;
}

/* Tab Widgets & Tab Bars */
QTabWidget::pane {
    border: 1px solid #334155;
    background-color: #0f172a;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #0f172a;
    color: #38bdf8;
    font-weight: bold;
    border-top: 2px solid #38bdf8;
    border-bottom: 1px solid #0f172a;
}
QTabBar::tab:hover:!selected {
    background-color: #334155;
    color: #f1f5f9;
}

/* Table Views, Tree Views & List Views */
QTableView, QTreeView, QListView {
    background-color: #0b0f19;
    color: #f1f5f9;
    gridline-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    selection-background-color: #1e3a8a;
    selection-color: #ffffff;
    alternate-background-color: #0f172a;
}
QTableView::item:hover {
    background-color: #1e293b;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #cbd5e1;
    font-weight: bold;
    font-size: 11px;
    border: 1px solid #334155;
    padding: 5px 6px;
}
QTableCornerButton::section {
    background-color: #1e293b;
    border: 1px solid #334155;
}

/* Text Inputs */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    font-size: 11px;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #38bdf8;
}

/* Number Inputs (SpinBoxes) */
QSpinBox, QDoubleSpinBox {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 11px;
}
QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #38bdf8;
}

/* Dropdown Combos */
QComboBox {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 8px;
    min-height: 20px;
    font-size: 11px;
}
QComboBox:hover {
    border-color: #475569;
}
QComboBox:focus {
    border-color: #38bdf8;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f1f5f9;
    selection-background-color: #334155;
    selection-color: #38bdf8;
    border: 1px solid #334155;
}

/* Standard Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
    color: #38bdf8;
}
QPushButton:pressed {
    background-color: #0f172a;
    border-color: #38bdf8;
}
QPushButton:disabled {
    background-color: #1e293b;
    color: #64748b;
    border-color: #1e293b;
}

/* Checkboxes & Radio Buttons */
QCheckBox, QRadioButton {
    color: #cbd5e1;
    spacing: 6px;
    font-size: 11px;
}
QCheckBox:hover, QRadioButton:hover {
    color: #f1f5f9;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #334155;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 12px;
    font-weight: bold;
    color: #38bdf8;
    font-size: 11px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}

/* ScrollBars */
QScrollBar:vertical {
    background-color: #0b0f19;
    border: none;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #334155;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #475569;
}
QScrollBar:horizontal {
    background-color: #0b0f19;
    border: none;
    height: 10px;
    margin: 0px;
}
QScrollBar::handle:horizontal {
    background-color: #334155;
    min-width: 20px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #475569;
}
QScrollBar::add-line, QScrollBar::sub-line {
    border: none;
    background: none;
}

/* Splitters */
QSplitter::handle {
    background-color: #1e293b;
}
QSplitter::handle:hover {
    background-color: #38bdf8;
}
"""


def create_dark_palette():
    """Builds a matching QPalette for headless/dialog Qt components."""
    try:
        from optics_app.gui.qt_compat import QtGui, QT_AVAILABLE
        if not QT_AVAILABLE or not hasattr(QtGui, "QPalette") or not hasattr(QtGui, "QColor"):
            return None
        QPalette = QtGui.QPalette
        QColor = QtGui.QColor

        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#0f172a"))
        pal.setColor(QPalette.WindowText, QColor("#f1f5f9"))
        pal.setColor(QPalette.Base, QColor("#0b0f19"))
        pal.setColor(QPalette.AlternateBase, QColor("#1e293b"))
        pal.setColor(QPalette.ToolTipBase, QColor("#1e293b"))
        pal.setColor(QPalette.ToolTipText, QColor("#f1f5f9"))
        pal.setColor(QPalette.Text, QColor("#f1f5f9"))
        pal.setColor(QPalette.Button, QColor("#1e293b"))
        pal.setColor(QPalette.ButtonText, QColor("#f1f5f9"))
        pal.setColor(QPalette.BrightText, QColor("#f87171"))
        pal.setColor(QPalette.Highlight, QColor("#2563eb"))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        return pal
    except Exception:
        return None


def create_light_palette():
    """Builds a matching clean light QPalette for headless/dialog Qt components."""
    try:
        from optics_app.gui.qt_compat import QtGui, QT_AVAILABLE
        if not QT_AVAILABLE or not hasattr(QtGui, "QPalette") or not hasattr(QtGui, "QColor"):
            return None
        QPalette = QtGui.QPalette
        QColor = QtGui.QColor

        pal = QPalette()
        pal.setColor(QPalette.Window, QColor("#ffffff"))
        pal.setColor(QPalette.WindowText, QColor("#0f172a"))
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.AlternateBase, QColor("#f8fafc"))
        pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
        pal.setColor(QPalette.ToolTipText, QColor("#0f172a"))
        pal.setColor(QPalette.Text, QColor("#0f172a"))
        pal.setColor(QPalette.Button, QColor("#f1f5f9"))
        pal.setColor(QPalette.ButtonText, QColor("#0f172a"))
        pal.setColor(QPalette.BrightText, QColor("#dc2626"))
        pal.setColor(QPalette.Highlight, QColor("#0284c7"))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        return pal
    except Exception:
        return None


def get_theme_colors(dark_mode: bool) -> Dict[str, str]:
    """Returns curated theme color palette for matplotlib plots, widgets, and canvases."""
    if dark_mode:
        return {
            "bg": "#0f172a",
            "bg_fig": "#0f172a",
            "bg_ax": "#0b0f19",
            "bg_card": "#1e293b",
            "text": "#f1f5f9",
            "fg_text": "#f1f5f9",
            "text_dim": "#94a3b8",
            "fg_subtext": "#94a3b8",
            "grid_col": "#334155",
            "grid": "#334155",
            "spine_col": "#334155",
            "axis_line_col": "#64748b",
            "accent_col": "#38bdf8",
            "success_col": "#4ade80",
            "warning_col": "#fbbf24",
            "danger_col": "#f87171",
            "bbox_face": "#1e293b",
            "bbox_edge": "#334155",
        }
    else:
        return {
            "bg": "#ffffff",
            "bg_fig": "#ffffff",
            "bg_ax": "#ffffff",
            "bg_card": "#f8fafc",
            "text": "#0f172a",
            "fg_text": "#0f172a",
            "text_dim": "#475569",
            "fg_subtext": "#475569",
            "grid_col": "#cbd5e1",
            "grid": "#cbd5e1",
            "spine_col": "#cbd5e1",
            "axis_line_col": "#64748b",
            "accent_col": "#0284c7",
            "success_col": "#16a34a",
            "warning_col": "#d97706",
            "danger_col": "#dc2626",
            "bbox_face": "#f8fafc",
            "bbox_edge": "#cbd5e1",
        }


def get_text_bbox(dark_mode: bool, pad: float = 0.5, alpha: float = 0.92) -> Dict[str, Any]:
    """Returns standard Matplotlib bbox dictionary matching current theme."""
    colors = get_theme_colors(dark_mode)
    return {
        "boxstyle": f"round,pad={pad}",
        "facecolor": colors["bbox_face"],
        "edgecolor": colors["bbox_edge"],
        "alpha": alpha,
        "linewidth": 1.0,
    }


def apply_axes_theme(ax, dark_mode: bool, grid_alpha: Optional[float] = None) -> None:
    """Consistently styles figure axes, labels, ticks, spines, and legend for theme."""
    if ax is None:
        return
    colors = get_theme_colors(dark_mode)
    try:
        ax.set_facecolor(colors["bg_ax"])
    except Exception:
        pass

    try:
        ax.tick_params(colors=colors["fg_subtext"], labelcolor=colors["fg_subtext"])
    except Exception:
        pass

    try:
        for spine in getattr(ax, "spines", {}).values():
            spine.set_color(colors["spine_col"])
    except Exception:
        pass

    try:
        if hasattr(ax, "xaxis") and ax.xaxis and hasattr(ax.xaxis, "label") and ax.xaxis.label:
            ax.xaxis.label.set_color(colors["fg_subtext"])
    except Exception:
        pass

    try:
        if hasattr(ax, "yaxis") and ax.yaxis and hasattr(ax.yaxis, "label") and ax.yaxis.label:
            ax.yaxis.label.set_color(colors["fg_subtext"])
    except Exception:
        pass

    try:
        if hasattr(ax, "title") and ax.title:
            ax.title.set_color(colors["fg_text"])
    except Exception:
        pass

    try:
        g_alpha = grid_alpha if grid_alpha is not None else (0.35 if dark_mode else 0.45)
        ax.grid(True, linestyle="--", alpha=g_alpha, color=colors["grid_col"])
    except Exception:
        pass

    try:
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(colors["bg_card"])
            leg.get_frame().set_edgecolor(colors["spine_col"])
            for text in leg.get_texts():
                text.set_color(colors["fg_text"])
    except Exception:
        pass


def style_colorbar_for_theme(cbar, dark_mode: bool) -> None:
    """Styles a Matplotlib colorbar to match dark or light theme seamlessly."""
    if cbar is None:
        return
    colors = get_theme_colors(dark_mode)
    try:
        if hasattr(cbar, "ax") and cbar.ax:
            cbar.ax.tick_params(colors=colors["fg_subtext"], labelsize=8)
            for spine in getattr(cbar.ax, "spines", {}).values():
                spine.set_color(colors["spine_col"])
            if hasattr(cbar.ax, "yaxis") and hasattr(cbar.ax.yaxis, "label") and cbar.ax.yaxis.label:
                cbar.ax.yaxis.label.set_color(colors["fg_subtext"])
            if hasattr(cbar.ax, "xaxis") and hasattr(cbar.ax.xaxis, "label") and cbar.ax.xaxis.label:
                cbar.ax.xaxis.label.set_color(colors["fg_subtext"])
    except Exception:
        pass
    try:
        if hasattr(cbar, "outline") and cbar.outline:
            cbar.outline.set_edgecolor(colors["spine_col"])
    except Exception:
        pass

