import sys
import os

# Override test headless environment
os.environ["QT_QPA_PLATFORM"] = "windows"

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from v2.gui.qt_compat import QtWidgets
except ImportError:
    from optics_app.gui.qt_compat import QtWidgets

from v2.gui.main_window import MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Optional: apply dark theme or similar
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.resize(1200, 800)
    window.setWindowTitle("2D Optics - V2 Spectrometer Tuner")
    window.show()
    
    sys.exit(app.exec())
