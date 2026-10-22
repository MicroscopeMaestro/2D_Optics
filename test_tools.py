import sys, os
from optics_app.gui.qt_compat import QtWidgets
from optics_app.gui.main_window import MainWindow
from optics_app.ai.tools import interpret_and_execute_command, ACTIVE_OPTICAL_SYSTEM

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
sys_obj = window.current_system

print("ACTIVE BEFORE:", ACTIVE_OPTICAL_SYSTEM)
res = interpret_and_execute_command("flip the first mirror", system=sys_obj)
print("RESULT:", res)
