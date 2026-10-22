import sys, os
sys.path.insert(0, os.path.abspath('v2'))

from optics_app.gui.qt_compat import QtWidgets, QtCore, QT_AVAILABLE
from optics_app.gui.ai_dock import AiChatDock

def test_gui():
    if not QT_AVAILABLE:
        print("QT NOT AVAILABLE")
        return
        
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    
    dock = AiChatDock(main_window=window)
    window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    
    dock.input_box.setText("hi")
    dock.send_message()
    
    if hasattr(dock, 'worker'):
        dock.worker.finished.connect(lambda res: print("Finished signal received:", res))
        dock.worker.error.connect(lambda err: print("Error signal received:", err))
    else:
        print("Worker was not created!")
        
    import threading
    def quit_app():
        print("Quitting app...")
        app.quit()
    threading.Timer(15.0, quit_app).start()
    
    print("Starting event loop...")
    app.exec_()
    print("Chat history:", dock.chat_history.toPlainText())

test_gui()
