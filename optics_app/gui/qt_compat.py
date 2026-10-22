"""
Universal Qt Binding & Headless Fallback Abstraction.
Supports PySide6, PyQt6, PySide2, PyQt5, and headless environments.
"""
import sys
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QT_AVAILABLE = False
QT_LIB = None

try:
    from PySide6 import QtWidgets, QtCore, QtGui
    from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot, Qt, QModelIndex, Property as pyqtProperty
    from PySide6.QtCore import Signal, Slot, QTimer
    QT_LIB = "PySide6"
    QT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui
        from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QModelIndex, pyqtProperty, QTimer
        Signal = pyqtSignal
        Slot = pyqtSlot
        QT_LIB = "PyQt6"
        QT_AVAILABLE = True
    except ImportError:
        try:
            from PySide2 import QtWidgets, QtCore, QtGui
            from PySide2.QtCore import Signal as pyqtSignal, Slot as pyqtSlot, Qt, QModelIndex, Property as pyqtProperty
            from PySide2.QtCore import Signal, Slot, QTimer
            QT_LIB = "PySide2"
            QT_AVAILABLE = True
        except ImportError:
            try:
                from PyQt5 import QtWidgets, QtCore, QtGui
                from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QModelIndex, pyqtProperty, QTimer
                Signal = pyqtSignal
                Slot = pyqtSlot
                QT_LIB = "PyQt5"
                QT_AVAILABLE = True
            except ImportError:
                QT_AVAILABLE = False
                QT_LIB = "HeadlessMock"


# ---------------------------------------------------------------------------
# Headless Mock Implementation Classes
# ---------------------------------------------------------------------------
class DummySignal:
    def __init__(self, *args, **kwargs):
        self._slots = []

    def connect(self, slot):
        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args, **kwargs):
        for slot in list(self._slots):
            try:
                slot(*args, **kwargs)
            except Exception:
                pass


class DummyQModelIndex:
    def __init__(self, row=-1, column=-1):
        self._row = row
        self._column = column

    def row(self):
        return self._row

    def column(self):
        return self._column

    def isValid(self):
        return self._row >= 0 and self._column >= 0


class DummyQt:
    class ItemDataRole:
        DisplayRole = 0
        DecorationRole = 1
        EditRole = 2
        ToolTipRole = 3
        StatusTipRole = 4
        WhatsThisRole = 5
        TextAlignmentRole = 6
        BackgroundRole = 8
        ForegroundRole = 9
        UserRole = 32

    DisplayRole = 0
    DecorationRole = 1
    EditRole = 2
    ToolTipRole = 3
    StatusTipRole = 4
    WhatsThisRole = 5
    TextAlignmentRole = 6
    BackgroundRole = 8
    ForegroundRole = 9
    UserRole = 32
    CaseInsensitive = 0

    class Orientation:
        Horizontal = 1
        Vertical = 2
    Horizontal = 1
    Vertical = 2

    class AlignmentFlag:
        AlignLeft = 1
        AlignRight = 2
        AlignCenter = 4
        AlignVCenter = 128
        AlignHCenter = 4
    AlignLeft = 1
    AlignRight = 2
    AlignCenter = 4
    AlignVCenter = 128
    AlignHCenter = 4

    class ItemFlags:
        ItemIsSelectable = 1
        ItemIsEditable = 2
        ItemIsEnabled = 4
    ItemIsSelectable = 1
    ItemIsEditable = 2
    ItemIsEnabled = 4

    class DockWidgetArea:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2
        TopDockWidgetArea = 4
        BottomDockWidgetArea = 8
        AllDockWidgetAreas = 15

    LeftDockWidgetArea = 1
    RightDockWidgetArea = 2
    TopDockWidgetArea = 4
    BottomDockWidgetArea = 8
    AllDockWidgetAreas = 15
    CustomContextMenu = 1
    WindowModal = 2
    ApplicationModal = 1
    NonModal = 0

    class TickPosition:
        NoTicks = 0
        TicksAbove = 1
        TicksLeft = 1
        TicksBelow = 2
        TicksRight = 2
        TicksBothSides = 3
    NoTicks = 0
    TicksAbove = 1
    TicksLeft = 1
    TicksBelow = 2
    TicksRight = 2
    TicksBothSides = 3


class DummyQObject:
    def __init__(self, parent=None):
        self.parent = parent
        self._block_signals = False

    def blockSignals(self, b: bool) -> bool:
        old = self._block_signals
        self._block_signals = bool(b)
        return old

    def setParent(self, parent):
        self.parent = parent


class DummyQAbstractTableModel(DummyQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.system_updated = DummySignal()
        self.dataChanged = DummySignal()

    def beginResetModel(self): pass
    def endResetModel(self): pass
    def beginInsertRows(self, parent, first, last): pass
    def endInsertRows(self): pass
    def beginRemoveRows(self, parent, first, last): pass
    def endRemoveRows(self): pass
    def beginMoveRows(self, s_parent, s_first, s_last, d_parent, d_child): pass
    def endMoveRows(self): pass

    def index(self, row, column, parent=None):
        return DummyQModelIndex(row, column)

    def createIndex(self, row, column):
        return DummyQModelIndex(row, column)


class DummyQStyledItemDelegate(DummyQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
    def createEditor(self, parent, option, index): return None
    def setEditorData(self, editor, index): pass
    def setModelData(self, editor, model, index): pass
    def updateEditorGeometry(self, editor, option, index): pass


class DummyQWidget(DummyQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_obj = None
        self._enabled = True
        self._visible = True
        self._tooltip = ""
        self._object_name = ""

    def setLayout(self, layout):
        self.layout_obj = layout

    def layout(self):
        return self.layout_obj

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def update(self): pass
    def repaint(self): pass
    def deleteLater(self): pass

    def setEnabled(self, b: bool):
        self._enabled = bool(b)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, b: bool):
        self._visible = bool(b)

    def isVisible(self) -> bool:
        return self._visible

    def setToolTip(self, t: str):
        self._tooltip = str(t)

    def toolTip(self) -> str:
        return self._tooltip

    def setObjectName(self, name: str):
        self._object_name = str(name)

    def objectName(self) -> str:
        return self._object_name

    def setWindowTitle(self, title: str):
        self._title = str(title)

    def windowTitle(self) -> str:
        return getattr(self, '_title', "")

    def accept(self):
        pass

    def setFixedWidth(self, w: int): pass
    def setFixedHeight(self, h: int): pass
    def setMinimumWidth(self, w: int): pass
    def setMaximumWidth(self, w: int): pass
    def setMinimumHeight(self, h: int): pass
    def setMaximumHeight(self, h: int): pass
    def setContentsMargins(self, l, t, r, b): pass


class DummyQAction(DummyQObject):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.text = text
        self.triggered = DummySignal()
        self._enabled = True
        self._checkable = False
        self._checked = False

    def setShortcut(self, sc): pass
    def setStatusTip(self, tip): pass
    def setToolTip(self, tip): pass
    def setEnabled(self, enabled): self._enabled = bool(enabled)
    def isEnabled(self): return self._enabled
    def setCheckable(self, c): self._checkable = bool(c)
    def isCheckable(self): return self._checkable
    def setChecked(self, c): self._checked = bool(c)
    def isChecked(self): return self._checked


class DummyQMenuBar(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._menus = []

    def addMenu(self, title):
        menu = DummyQMenu(title, self)
        self._menus.append(menu)
        return menu


class DummyQMenu(DummyQWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self._actions = []

    def addAction(self, text_or_action):
        if isinstance(text_or_action, DummyQAction):
            action = text_or_action
        else:
            action = DummyQAction(str(text_or_action), self)
        self._actions.append(action)
        return action

    def addMenu(self, title):
        menu = DummyQMenu(title, self)
        self._actions.append(menu)
        return menu

    def addSeparator(self): pass
    def exec_(self, pos=None): return None
    def exec(self, pos=None): return None


class DummyQToolBar(DummyQWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._actions = []

    def addAction(self, text_or_action):
        if isinstance(text_or_action, DummyQAction):
            action = text_or_action
        else:
            action = DummyQAction(str(text_or_action), self)
        self._actions.append(action)
        return action

    def addSeparator(self): pass
    def addWidget(self, widget): pass


class DummyQStatusBar(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._msg = ""

    def showMessage(self, msg, timeout=0):
        self._msg = str(msg)

    def currentMessage(self):
        return self._msg


class DummyQMainWindow(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._central_widget = None
        self._status_bar = DummyQStatusBar(self)
        self._menu_bar = DummyQMenuBar(self)

    def setCentralWidget(self, widget):
        self._central_widget = widget

    def centralWidget(self):
        return self._central_widget

    def setWindowTitle(self, title):
        self._title = str(title)

    def windowTitle(self):
        return getattr(self, '_title', "")

    def resize(self, w, h): pass

    def menuBar(self):
        if self._menu_bar is None:
            self._menu_bar = DummyQMenuBar(self)
        return self._menu_bar

    def setStatusBar(self, sb):
        self._status_bar = sb

    def statusBar(self):
        if self._status_bar is None:
            self._status_bar = DummyQStatusBar(self)
        return self._status_bar

    def addToolBar(self, title):
        return DummyQToolBar(title, self)

    def setDockNestingEnabled(self, enabled): pass
    def addDockWidget(self, area, dock): pass
    def splitDockWidget(self, first, second, orientation): pass
    def tabifyDockWidget(self, first, second): pass
    def close(self): pass


class DummyQHeaderView(DummyQWidget):
    def setSectionResizeMode(self, *args): pass
    class ResizeMode:
        Stretch = 1
        Interactive = 0
    Stretch = 1
    Interactive = 0


class DummyQTableView(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._selection_behavior = None
        self._delegates = {}
        self.customContextMenuRequested = DummySignal()

    def setModel(self, model):
        self._model = model

    def model(self):
        return self._model

    def setSelectionBehavior(self, behavior):
        self._selection_behavior = behavior

    def currentIndex(self):
        return DummyQModelIndex(-1, -1)

    def setItemDelegateForColumn(self, col, delegate):
        self._delegates[col] = delegate

    def itemDelegateForColumn(self, col):
        return self._delegates.get(col, None)

    def horizontalHeader(self):
        return DummyQHeaderView(self)

    def setContextMenuPolicy(self, policy): pass
    def viewport(self): return self
    def mapToGlobal(self, pos): return pos
    def indexAt(self, pos): return DummyQModelIndex(0, 0)
    def selectionModel(self): return None


class DummyQTabWidget(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tabs = []
        self.currentChanged = DummySignal()
        self._current_index = 0

    def addTab(self, widget, label):
        self._tabs.append((widget, str(label)))

    def count(self):
        return len(self._tabs)

    def setCurrentIndex(self, index):
        if 0 <= index < len(self._tabs):
            self._current_index = index
            self.currentChanged.emit(index)

    def currentIndex(self):
        return self._current_index


class DummyQSplitter(DummyQWidget):
    def __init__(self, orientation=1, parent=None):
        super().__init__(parent)
        self._widgets = []
        self._orientation = orientation

    def addWidget(self, widget):
        self._widgets.append(widget)

    def setSizes(self, sizes): pass
    def setOrientation(self, orientation): self._orientation = orientation


class DummyQFileDialog:
    @staticmethod
    def getOpenFileName(parent=None, caption="", dir="", filter=""):
        return "", ""
    @staticmethod
    def getSaveFileName(parent=None, caption="", dir="", filter=""):
        return "", ""


class DummyQMessageBox:
    Ok = 1
    Cancel = 2
    Yes = 3
    No = 4
    @staticmethod
    def information(parent=None, title="", text="", *args, **kwargs):
        return 1
    @staticmethod
    def warning(parent=None, title="", text="", *args, **kwargs):
        return 1
    @staticmethod
    def critical(parent=None, title="", text="", *args, **kwargs):
        return 1
    @staticmethod
    def question(parent=None, title="", text="", *args, **kwargs):
        return 1


class DummyQComboBox(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []       # list of (text, userData)
        self._current_index = 0
        self._editable = False
        self._completer = None
        self.currentIndexChanged = DummySignal()
        self.currentTextChanged = DummySignal()

    def addItem(self, text, userData=None):
        self._items.append((str(text), userData))
        if len(self._items) == 1:
            self._current_index = 0

    def addItems(self, items):
        for item in items:
            self._items.append((str(item), None))
        if len(self._items) > 0 and self._current_index < 0:
            self._current_index = 0

    def clear(self):
        self._items.clear()
        self._current_index = -1

    def count(self) -> int:
        return len(self._items)

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return ""

    def setCurrentText(self, text: str):
        text = str(text)
        for i, (item_text, _) in enumerate(self._items):
            if item_text == text:
                self.setCurrentIndex(i)
                return
        if self._editable:
            self.addItem(text)
            self.setCurrentIndex(len(self._items) - 1)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, idx: int):
        if 0 <= idx < len(self._items):
            self._current_index = idx
            if not self._block_signals:
                self.currentIndexChanged.emit(idx)
                self.currentTextChanged.emit(self.currentText())

    def currentData(self, role=32):
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def itemText(self, idx: int) -> str:
        if 0 <= idx < len(self._items):
            return self._items[idx][0]
        return ""

    def itemData(self, idx: int, role=32):
        if 0 <= idx < len(self._items):
            return self._items[idx][1]
        return None

    def findData(self, data, role=32) -> int:
        for i, (_, userData) in enumerate(self._items):
            if userData == data:
                return i
        return -1

    def findText(self, text: str) -> int:
        text = str(text)
        for i, (item_text, _) in enumerate(self._items):
            if item_text == text:
                return i
        return -1

    def setEditable(self, editable: bool):
        self._editable = bool(editable)

    def setCompleter(self, completer):
        self._completer = completer


class DummyQDoubleSpinBox(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._val = 0.0
        self._min = -1e6
        self._max = 1e6
        self._step = 1.0
        self._decimals = 4
        self._prefix = ""
        self._suffix = ""
        self.valueChanged = DummySignal()

    def setRange(self, min_val: float, max_val: float):
        self._min = float(min_val)
        self._max = float(max_val)

    def setMinimum(self, min_val: float):
        self._min = float(min_val)

    def setMaximum(self, max_val: float):
        self._max = float(max_val)

    def setSingleStep(self, step: float):
        self._step = float(step)

    def setDecimals(self, decimals: int):
        self._decimals = int(decimals)

    def setPrefix(self, p: str):
        self._prefix = str(p)

    def setSuffix(self, s: str):
        self._suffix = str(s)

    def value(self) -> float:
        return self._val

    def setValue(self, val: float):
        self._val = float(val)
        if not self._block_signals:
            self.valueChanged.emit(self._val)


class DummyQLineEdit(DummyQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._text = str(text)
        self.returnPressed = DummySignal()
        self.textChanged = DummySignal()

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = str(text)
        if not self._block_signals:
            self.textChanged.emit(self._text)


class DummyQPushButton(DummyQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.text = str(text)
        self.clicked = DummySignal()

    def setText(self, t: str):
        self.text = str(t)

    def text(self) -> str:
        return getattr(self, 'text', "")


class DummyQHBoxLayout:
    def __init__(self, parent=None):
        self._widgets = []
        if parent is not None and hasattr(parent, 'setLayout'):
            parent.setLayout(self)

    def addWidget(self, widget):
        self._widgets.append(widget)

    def addLayout(self, layout):
        self._widgets.append(layout)

    def setContentsMargins(self, l, t, r, b): pass
    def addStretch(self, stretch=0): pass


class DummyQVBoxLayout:
    def __init__(self, parent=None):
        self._widgets = []
        if parent is not None and hasattr(parent, 'setLayout'):
            parent.setLayout(self)

    def addWidget(self, widget):
        self._widgets.append(widget)

    def addLayout(self, layout):
        self._widgets.append(layout)

    def removeWidget(self, widget):
        if widget in self._widgets:
            self._widgets.remove(widget)

    def setContentsMargins(self, l, t, r, b): pass
    def addStretch(self, stretch=0): pass
    def count(self): return len(self._widgets)
    def takeAt(self, idx):
        if 0 <= idx < len(self._widgets):
            w = self._widgets.pop(idx)
            class Item:
                def widget(self): return w if isinstance(w, DummyQWidget) else None
                def layout(self): return w if not isinstance(w, DummyQWidget) else None
            return Item()
        return None


class DummyQGridLayout:
    def __init__(self, parent=None):
        self._items = []
        if parent is not None and hasattr(parent, 'setLayout'):
            parent.setLayout(self)

    def addWidget(self, widget, row=0, col=0, row_span=1, col_span=1):
        self._items.append((widget, row, col, row_span, col_span))

    def addLayout(self, layout, row=0, col=0, row_span=1, col_span=1):
        self._items.append((layout, row, col, row_span, col_span))

    def setContentsMargins(self, l, t, r, b): pass
    def setSpacing(self, s): pass
    def setColumnStretch(self, col, stretch): pass
    def setRowStretch(self, row, stretch): pass


class DummyQLabel(DummyQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._text = str(text)

    def setText(self, text: str):
        self._text = str(text)

    def text(self) -> str:
        return self._text


class DummyQSpinBox(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._val = 0
        self._min = -1000000
        self._max = 1000000
        self._step = 1
        self._prefix = ""
        self._suffix = ""
        self.valueChanged = DummySignal()

    def setRange(self, min_val: int, max_val: int):
        self._min = int(min_val)
        self._max = int(max_val)

    def setMinimum(self, min_val: int):
        self._min = int(min_val)

    def setMaximum(self, max_val: int):
        self._max = int(max_val)

    def setValue(self, val: int):
        self._val = int(val)
        if not self._block_signals:
            self.valueChanged.emit(self._val)

    def value(self) -> int:
        return self._val

    def setSingleStep(self, step: int):
        self._step = int(step)

    def setPrefix(self, p: str):
        self._prefix = str(p)

    def setSuffix(self, s: str):
        self._suffix = str(s)


class DummyQSlider(DummyQWidget):
    def __init__(self, orientation=1, parent=None):
        super().__init__(parent)
        self._orientation = orientation
        self._min = 0
        self._max = 100
        self._val = 0
        self._step = 1
        self._page_step = 10
        self._tick_pos = 0
        self._tick_interval = 0
        self.valueChanged = DummySignal()
        self.sliderMoved = DummySignal()

    def setOrientation(self, o: int):
        self._orientation = int(o)

    def setRange(self, min_val: int, max_val: int):
        self._min = int(min_val)
        self._max = int(max_val)

    def setMinimum(self, min_val: int):
        self._min = int(min_val)

    def setMaximum(self, max_val: int):
        self._max = int(max_val)

    def minimum(self) -> int:
        return self._min

    def maximum(self) -> int:
        return self._max

    def setValue(self, val: int):
        self._val = int(val)
        if not self._block_signals:
            self.valueChanged.emit(self._val)

    def value(self) -> int:
        return self._val

    def setSingleStep(self, step: int):
        self._step = int(step)

    def setPageStep(self, step: int):
        self._page_step = int(step)

    def setTickPosition(self, tp: int):
        self._tick_pos = int(tp)

    def setTickInterval(self, ti: int):
        self._tick_interval = int(ti)


class DummyQDial(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._val = 0
        self._wrapping = False
        self._notches_visible = False
        self.valueChanged = DummySignal()

    def setRange(self, min_val: int, max_val: int):
        self._min = int(min_val)
        self._max = int(max_val)

    def setMinimum(self, min_val: int):
        self._min = int(min_val)

    def setMaximum(self, max_val: int):
        self._max = int(max_val)

    def setValue(self, val: int):
        self._val = int(val)
        if not self._block_signals:
            self.valueChanged.emit(self._val)

    def value(self) -> int:
        return self._val

    def setWrapping(self, w: bool):
        self._wrapping = bool(w)

    def setNotchesVisible(self, nv: bool):
        self._notches_visible = bool(nv)


class DummyQTimer(DummyQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._interval = 0
        self._single_shot = False
        self._active = False
        self.timeout = DummySignal()

    def start(self, msec: int = None):
        if msec is not None:
            self._interval = int(msec)
        self._active = True

    def stop(self):
        self._active = False

    def setInterval(self, msec: int):
        self._interval = int(msec)

    def interval(self) -> int:
        return self._interval

    def setSingleShot(self, b: bool):
        self._single_shot = bool(b)

    def isSingleShot(self) -> bool:
        return self._single_shot

    def isActive(self) -> bool:
        return self._active

    @staticmethod
    def singleShot(msec: int, callable_fn):
        if callable(callable_fn):
            try:
                callable_fn()
            except Exception:
                pass


class DummyQDialog(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = 1
        self.accepted = DummySignal()
        self.rejected = DummySignal()

    def exec(self) -> int:
        return self._result

    def exec_(self) -> int:
        return self._result

    def accept(self):
        self.accepted.emit()

    def reject(self):
        self.rejected.emit()

    def setModal(self, b: bool): pass
    def setWindowTitle(self, t: str): self._title = str(t)


class DummyQDockWidget(DummyQWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = str(title)
        self._widget = None
        self._object_name = ""

    def setWidget(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget

    def setAllowedAreas(self, areas): pass

    def toggleViewAction(self):
        return DummyQAction(self._title, self)


class DummyQCheckBox(DummyQWidget):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._text = str(text)
        self._checked = False
        self.toggled = DummySignal()
        self.stateChanged = DummySignal()

    def setChecked(self, checked: bool):
        self._checked = bool(checked)
        if not self._block_signals:
            self.toggled.emit(self._checked)
            self.stateChanged.emit(2 if self._checked else 0)

    def isChecked(self) -> bool:
        return self._checked

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = str(text)


class DummyQProgressBar(DummyQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._val = 0
        self._min = 0
        self._max = 100

    def setRange(self, min_val: int, max_val: int):
        self._min = int(min_val)
        self._max = int(max_val)

    def setValue(self, val: int):
        self._val = int(val)

    def value(self) -> int:
        return self._val


class DummyQProgressDialog(DummyQWidget):
    def __init__(self, labelText="", cancelButtonText="", minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        self._val = minimum
        self._min = minimum
        self._max = maximum
        self._canceled = False
        self._label_text = str(labelText)
        self._cancel_button_text = str(cancelButtonText)
        self.canceled = DummySignal()

    def setRange(self, min_val: int, max_val: int):
        self._min = int(min_val)
        self._max = int(max_val)

    def setValue(self, val: int):
        self._val = int(val)

    def value(self) -> int:
        return self._val

    def setLabelText(self, text: str):
        self._label_text = str(text)

    def setCancelButtonText(self, text: str):
        self._cancel_button_text = str(text)

    def setWindowTitle(self, title: str):
        self._title = str(title)

    def windowTitle(self) -> str:
        return getattr(self, '_title', "")

    def labelText(self) -> str:
        return self._label_text

    def minimum(self) -> int:
        return self._min

    def maximum(self) -> int:
        return self._max

    def setCancelButton(self, btn): pass
    def setMinimumDuration(self, ms: int): pass
    def setWindowModality(self, modality): pass
    def setAutoReset(self, b: bool): pass
    def setAutoClose(self, b: bool): pass
    def setBar(self, bar): pass
    def setLabel(self, label): pass
    def open(self): pass
    def exec(self) -> int: return 1
    def exec_(self) -> int: return 1
    def accept(self): pass

    def wasCanceled(self) -> bool:
        return self._canceled

    def cancel(self):
        self._canceled = True
        self.canceled.emit()

    def reset(self):
        self._val = self._min
        self._canceled = False

    def close(self): pass


class DummyQGroupBox(DummyQWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = str(title)

    def setTitle(self, title: str):
        self._title = str(title)

    def title(self) -> str:
        return self._title


class DummyQFrame(DummyQWidget):
    Shape = 0
    Shadow = 0
    def setFrameShape(self, s): pass
    def setFrameShadow(self, s): pass


class DummyQScrollArea(DummyQFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._widget = None
    def setWidget(self, w): self._widget = w
    def widget(self): return self._widget
    def setWidgetResizable(self, r): pass


class DummyQThread(DummyQObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.finished = DummySignal()
        self.started = DummySignal()
        self._running = False

    def start(self):
        self._running = True
        self.started.emit()
        self.run()
        self._running = False
        self.finished.emit()

    def wait(self): pass
    def quit(self): self._running = False
    def run(self): pass


class DummyQCompleter:
    def __init__(self, items, parent=None):
        self.items = items
    def setCaseSensitivity(self, cs): pass


class DummyQAbstractItemView:
    SelectRows = 1


class DummyFormLayout(DummyQVBoxLayout):
    def addRow(self, label, field):
        if isinstance(label, str):
            lbl = DummyQLabel(label)
        else:
            lbl = label
        row = DummyQHBoxLayout()
        row.addWidget(lbl)
        row.addWidget(field)
        self.addLayout(row)


class DummyQtWidgetsModule:
    QMainWindow = DummyQMainWindow
    QWidget = DummyQWidget
    QTableView = DummyQTableView
    QTabWidget = DummyQTabWidget
    QSplitter = DummyQSplitter
    QToolBar = DummyQToolBar
    QMenuBar = DummyQMenuBar
    QMenu = DummyQMenu
    QAction = DummyQAction
    QStatusBar = DummyQStatusBar
    QFileDialog = DummyQFileDialog
    QMessageBox = DummyQMessageBox
    QComboBox = DummyQComboBox
    QDoubleSpinBox = DummyQDoubleSpinBox
    QLineEdit = DummyQLineEdit
    QPushButton = DummyQPushButton
    QHBoxLayout = DummyQHBoxLayout
    QVBoxLayout = DummyQVBoxLayout
    QGridLayout = DummyQGridLayout
    QLabel = DummyQLabel
    QSpinBox = DummyQSpinBox
    QSlider = DummyQSlider
    QDial = DummyQDial
    QDockWidget = DummyQDockWidget
    QCheckBox = DummyQCheckBox
    QProgressBar = DummyQProgressBar
    QProgressDialog = DummyQProgressDialog
    QGroupBox = DummyQGroupBox
    QFrame = DummyQFrame
    QScrollArea = DummyQScrollArea
    QDialog = DummyQDialog
    QCompleter = DummyQCompleter
    QHeaderView = DummyQHeaderView
    QAbstractItemView = DummyQAbstractItemView
    QStyledItemDelegate = DummyQStyledItemDelegate
    QFormLayout = DummyFormLayout


class DummyQtCoreModule:
    QAbstractTableModel = DummyQAbstractTableModel
    QModelIndex = DummyQModelIndex
    Qt = DummyQt
    QObject = DummyQObject
    QThread = DummyQThread
    QTimer = DummyQTimer
    pyqtSignal = DummySignal
    Signal = DummySignal
    pyqtSlot = lambda *a, **k: (lambda f: f)
    Slot = lambda *a, **k: (lambda f: f)
    pyqtProperty = lambda *a, **k: (lambda f: f)


class DummyQColor:
    def __init__(self, *args, **kwargs):
        self._name = args[0] if len(args) > 0 and isinstance(args[0], str) else "#000000"
    def name(self):
        return self._name


class DummyQBrush:
    def __init__(self, color=None):
        self._color = color if color is not None else DummyQColor()
    def color(self):
        return self._color


class DummyQtGuiModule:
    QAction = DummyQAction
    QBrush = DummyQBrush
    QColor = DummyQColor


# ---------------------------------------------------------------------------
# Module Exposing (Real Qt or Headless Fallbacks)
# ---------------------------------------------------------------------------
if QT_AVAILABLE:
    QAbstractTableModel = QtCore.QAbstractTableModel
    QModelIndex = QtCore.QModelIndex
    Qt = QtCore.Qt
    QObject = QtCore.QObject
    QThread = QtCore.QThread
    QTimer = QtCore.QTimer

    QBrush = getattr(QtGui, 'QBrush', DummyQBrush)
    QColor = getattr(QtGui, 'QColor', DummyQColor)

    QStyledItemDelegate = QtWidgets.QStyledItemDelegate
    QWidget = QtWidgets.QWidget
    QMainWindow = QtWidgets.QMainWindow
    QTableView = QtWidgets.QTableView
    QTabWidget = QtWidgets.QTabWidget
    QSplitter = QtWidgets.QSplitter
    QToolBar = QtWidgets.QToolBar
    QMenuBar = QtWidgets.QMenuBar
    QMenu = QtWidgets.QMenu
    QAction = QtGui.QAction if hasattr(QtGui, 'QAction') else QtWidgets.QAction
    QStatusBar = QtWidgets.QStatusBar
    QFileDialog = QtWidgets.QFileDialog
    QMessageBox = QtWidgets.QMessageBox
    QDockWidget = QtWidgets.QDockWidget
    QApplication = QtWidgets.QApplication
    QDialog = QtWidgets.QDialog
    QProgressDialog = getattr(QtWidgets, 'QProgressDialog', DummyQProgressDialog)

    # Ensure non-blocking dialogs during tests / offscreen execution
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "pytest" in sys.modules or "unittest" in sys.modules:
        class NonBlockingQMessageBox:
            Ok = 1
            Cancel = 2
            Yes = 3
            No = 4
            @staticmethod
            def information(parent=None, title="", text="", *args, **kwargs):
                return getattr(getattr(QtWidgets, 'QMessageBox', None), 'Ok', 1)
            @staticmethod
            def warning(parent=None, title="", text="", *args, **kwargs):
                return getattr(getattr(QtWidgets, 'QMessageBox', None), 'Ok', 1)
            @staticmethod
            def critical(parent=None, title="", text="", *args, **kwargs):
                return getattr(getattr(QtWidgets, 'QMessageBox', None), 'Ok', 1)
            @staticmethod
            def question(parent=None, title="", text="", *args, **kwargs):
                return getattr(getattr(QtWidgets, 'QMessageBox', None), 'Yes', 1)
        QMessageBox = NonBlockingQMessageBox
        if hasattr(QtWidgets, 'QMessageBox'):
            QtWidgets.QMessageBox = NonBlockingQMessageBox
        QProgressDialog = DummyQProgressDialog
        if hasattr(QtWidgets, 'QProgressDialog'):
            QtWidgets.QProgressDialog = DummyQProgressDialog

    QComboBox = QtWidgets.QComboBox
    QDoubleSpinBox = QtWidgets.QDoubleSpinBox
    QLineEdit = QtWidgets.QLineEdit
    QLabel = QtWidgets.QLabel
    QPushButton = QtWidgets.QPushButton
    QHBoxLayout = QtWidgets.QHBoxLayout
    QVBoxLayout = QtWidgets.QVBoxLayout
    QGridLayout = QtWidgets.QGridLayout
    QCompleter = QtWidgets.QCompleter
    QHeaderView = QtWidgets.QHeaderView
    QAbstractItemView = QtWidgets.QAbstractItemView
    QFormLayout = QtWidgets.QFormLayout
    QSpinBox = QtWidgets.QSpinBox
    QSlider = QtWidgets.QSlider
    QDial = QtWidgets.QDial
    QCheckBox = QtWidgets.QCheckBox
    QProgressBar = QtWidgets.QProgressBar
    QGroupBox = QtWidgets.QGroupBox
    QFrame = QtWidgets.QFrame
    QScrollArea = QtWidgets.QScrollArea
    QTableWidget = QtWidgets.QTableWidget
    QTableWidgetItem = QtWidgets.QTableWidgetItem
    QRadioButton = QtWidgets.QRadioButton
    QButtonGroup = QtWidgets.QButtonGroup
    QMimeData = getattr(QtCore, 'QMimeData', None)

else:
    class DummyQMimeData:
        def __init__(self):
            self._data = {}
            self._text = ""
        def setData(self, mime_type: str, data: bytes):
            self._data[mime_type] = data
        def data(self, mime_type: str) -> bytes:
            return self._data.get(mime_type, b"")
        def hasFormat(self, mime_type: str) -> bool:
            return mime_type in self._data
        def setText(self, text: str):
            self._text = str(text)
        def text(self) -> str:
            return self._text
        def hasText(self) -> bool:
            return bool(self._text)

    QMimeData = DummyQMimeData
    pyqtSignal = DummySignal
    Signal = DummySignal
    pyqtSlot = lambda *a, **k: (lambda f: f)
    Slot = lambda *a, **k: (lambda f: f)
    pyqtProperty = lambda *a, **k: (lambda f: f)

    QModelIndex = DummyQModelIndex
    Qt = DummyQt
    QObject = DummyQObject
    QThread = DummyQThread
    QTimer = DummyQTimer
    QAbstractTableModel = DummyQAbstractTableModel
    QStyledItemDelegate = DummyQStyledItemDelegate
    QWidget = DummyQWidget
    QMainWindow = DummyQMainWindow
    QTableView = DummyQTableView
    QTabWidget = DummyQTabWidget
    QSplitter = DummyQSplitter
    QToolBar = DummyQToolBar
    QMenuBar = DummyQMenuBar
    QMenu = DummyQMenu
    QAction = DummyQAction
    QStatusBar = DummyQStatusBar
    QFileDialog = DummyQFileDialog
    QMessageBox = DummyQMessageBox
    QDockWidget = DummyQDockWidget
    QDialog = DummyQDialog
    QComboBox = DummyQComboBox
    QDoubleSpinBox = DummyQDoubleSpinBox
    QLineEdit = DummyQLineEdit
    QLabel = DummyQLabel
    QPushButton = DummyQPushButton
    QHBoxLayout = DummyQHBoxLayout
    QVBoxLayout = DummyQVBoxLayout
    QGridLayout = DummyQGridLayout
    QSpinBox = DummyQSpinBox
    QSlider = DummyQSlider
    QDial = DummyQDial
    QCheckBox = DummyQCheckBox
    QProgressBar = DummyQProgressBar
    QGroupBox = DummyQGroupBox
    QFrame = DummyQFrame
    QScrollArea = DummyQScrollArea
    QCompleter = DummyQCompleter
    QHeaderView = DummyQHeaderView
    QAbstractItemView = DummyQAbstractItemView
    QFormLayout = DummyFormLayout

    class DummyQTableWidgetItem:
        def __init__(self, text=""):
            self._text = str(text)
            self._data_roles = {}
        def text(self): return self._text
        def setText(self, t): self._text = str(t)
        def setData(self, role, val): self._data_roles[role] = val
        def data(self, role=0): return self._data_roles.get(role)

    class DummyQTableWidget(DummyQWidget):
        def __init__(self, rows=0, cols=0, parent=None):
            super().__init__(parent)
            self._rows = rows
            self._cols = cols
            self._items = {}
            self.cellChanged = DummySignal()
            self._horizontal_header = DummyQHeaderView(self)
        def setRowCount(self, r): self._rows = r
        def rowCount(self): return self._rows
        def setColumnCount(self, c): self._cols = c
        def columnCount(self): return self._cols
        def setHorizontalHeaderLabels(self, labels): self._headers = labels
        def setItem(self, r, c, item): self._items[(r, c)] = item
        def item(self, r, c): return self._items.get((r, c), DummyQTableWidgetItem(""))
        def setCellWidget(self, r, c, w): self._items[(r, c)] = w
        def cellWidget(self, r, c): return self._items.get((r, c))
        def currentRow(self): return 0
        def horizontalHeader(self): return self._horizontal_header

    class DummyQRadioButton(DummyQWidget):
        def __init__(self, text="", parent=None):
            super().__init__(parent)
            self._checked = False
            self.toggled = DummySignal()
        def setChecked(self, b):
            self._checked = bool(b)
            self.toggled.emit(self._checked)
        def isChecked(self): return self._checked

    class DummyQButtonGroup(DummyQObject):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._buttons = []
        def addButton(self, btn): self._buttons.append(btn)

    QTableWidget = DummyQTableWidget
    QTableWidgetItem = DummyQTableWidgetItem
    QRadioButton = DummyQRadioButton
    QButtonGroup = DummyQButtonGroup
    QBrush = DummyQBrush
    QColor = DummyQColor

    QtWidgets = DummyQtWidgetsModule()
    QtCore = DummyQtCoreModule()
    QtGui = DummyQtGuiModule()


__all__ = [
    "QT_AVAILABLE", "QT_LIB", "DummySignal", "DummyQtWidgetsModule", "DummyQtCoreModule", "DummyQtGuiModule",
    "QtWidgets", "QtCore", "QtGui", "pyqtSignal", "pyqtSlot", "Signal", "Slot", "pyqtProperty",
    "QAbstractTableModel", "QModelIndex", "Qt", "QStyledItemDelegate", "QWidget", "QMainWindow",
    "QTableView", "QTabWidget", "QSplitter", "QToolBar", "QMenuBar", "QMenu", "QAction",
    "QStatusBar", "QFileDialog", "QMessageBox", "QDockWidget", "QComboBox", "QDoubleSpinBox",
    "QLineEdit", "QLabel", "QPushButton", "QHBoxLayout", "QVBoxLayout", "QGridLayout", "QCompleter",
    "QHeaderView", "QAbstractItemView", "QFormLayout", "QSpinBox", "QSlider", "QDial", "QTimer",
    "QDialog", "QCheckBox", "QProgressBar", "QProgressDialog", "QGroupBox", "QFrame", "QScrollArea", "QThread", "QObject",
    "QTableWidget", "QTableWidgetItem", "QRadioButton", "QButtonGroup", "QBrush", "QColor", "QMimeData"
]
