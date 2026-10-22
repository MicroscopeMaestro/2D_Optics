import sys
import os

# The test suite forces the GUI to be invisible ("offscreen").
# We must override it here for the actual desktop app so the user can see it!
os.environ["QT_QPA_PLATFORM"] = "windows"

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optics_app.gui.qt_compat import QtWidgets, QtGui
from optics_app.gui.main_window import MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
    if os.path.exists(icon_path):
        app_icon = QtGui.QIcon(icon_path)
        app.setWindowIcon(app_icon)
    else:
        app_icon = None

    window = MainWindow()
    if app_icon is not None:
        window.setWindowIcon(app_icon)
        
    window.showMaximized()
    sys.exit(app.exec())
