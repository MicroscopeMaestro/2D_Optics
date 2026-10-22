import re
with open(r'v2/gui/ai_dock.py', 'r', encoding='utf-8') as f:
    text = f.read()

agent_worker_pattern = r'class AgentWorker.*?self\.error\.emit\(str\(e\)\)'
command_worker_code = '''class CommandWorker(QThread if QT_AVAILABLE else object):
    if QT_AVAILABLE:
        finished = pyqtSignal(object)
        error = pyqtSignal(str)

    def __init__(self, text, system):
        super().__init__()
        self.text = text
        self.system = system

    def run(self):
        try:
            res = interpret_and_execute_command(self.text, system=self.system)
            if QT_AVAILABLE:
                self.finished.emit(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if QT_AVAILABLE:
                self.error.emit(str(e))'''

text = re.sub(agent_worker_pattern, command_worker_code, text, flags=re.DOTALL)

send_message_pattern = r'    def send_message\(self\):.*?Understood\.\}\"\)'
new_send_message = '''    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.chat_history.append(f"<b>You:</b> {text}")
        self.input_box.clear()
        
        if hasattr(self, 'btn_send'):
            self.btn_send.setEnabled(False)
        self.input_box.setEnabled(False)

        # Update active optical system pointer
        system = getattr(self.main_window, "current_system", None) if self.main_window else None
        if system is not None:
            set_active_optical_system(system)

        # Execute in background thread to prevent UI freeze
        self.worker = CommandWorker(text, system)
        self.worker.finished.connect(self.on_command_finished)
        self.worker.error.connect(self.on_command_error)
        self.worker.start()

    def on_command_finished(self, res):
        if hasattr(self, 'btn_send'):
            self.btn_send.setEnabled(True)
        self.input_box.setEnabled(True)
        
        system = getattr(self.worker, 'system', None)
        
        # Write back changes to synchronize wavelengths, fields, apertures
        if self.main_window and system:
            self.main_window.current_system = system
            
        if res.get("executed_results"):
            for step in res["executed_results"]:
                self.chat_history.append(f"<font color='green'><b>Action:</b> {step['tool']}</font>")
                self.chat_history.append(f"<pre>{step['result']}</pre>")
            if hasattr(self.main_window, "update_all_views"):
                self.main_window.update_all_views()
        else:
            self.chat_history.append(f"<b>AI:</b> {res.get('assistant_reply', 'Understood.')}")

    def on_command_error(self, err_msg):
        if hasattr(self, 'btn_send'):
            self.btn_send.setEnabled(True)
        self.input_box.setEnabled(True)
        self.chat_history.append(f"<font color='red'><b>Error:</b> {err_msg}</font>")'''

text = re.sub(send_message_pattern, new_send_message, text, flags=re.DOTALL)

with open(r'v2/gui/ai_dock.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open(r'v2/optics_app/gui/ai_dock.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated both ai_dock.py files to use QThread')
