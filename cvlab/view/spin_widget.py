from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QLineEdit, QDoubleSpinBox, QSpinBox, QApplication


class NumberLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.pressed_pos = None

    def mousePressEvent(self, event):
        QApplication.instance().setOverrideCursor(Qt.BlankCursor)
        self.pressed_pos = event.globalPos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        QApplication.instance().restoreOverrideCursor()
        self.pressed_pos = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if not self.pressed_pos: return
        new_pos = event.globalPos()
        QCursor.setPos(self.pressed_pos)
        difference = new_pos - self.pressed_pos
        x, y = difference.x(), difference.y()
        if abs(x) >= abs(y) or abs(x)+abs(y)<=2:
            add = difference.x() / 10000.0
        else:
            add = -difference.y() / 500.0
        self.parent().mouseSpin.emit(add, 1.0)


class DoubleSpinBoxEx(QDoubleSpinBox):
    mouseSpin = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.pressed_pos = None
        self.setLineEdit(NumberLineEdit())
        self.mouseSpin.connect(self.change_value)

    @pyqtSlot(float, float)
    def change_value(self, add, mul):
        value = self.value()
        if value < 0: mul = 1/mul
        value = value * mul + add * (self.maximum()-self.minimum())
        self.setValue(value)


class SpinBoxEx(QSpinBox):
    mouseSpin = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.pressed_pos = None
        self.setLineEdit(NumberLineEdit())
        self.mouseSpin.connect(self.change_value)
        self.valueChanged.connect(self.value_changed_slot)
        self.float_value = None

    @pyqtSlot(int)
    def value_changed_slot(self, value):
        self.float_value = value

    @pyqtSlot(float, float)
    def change_value(self, add, mul):
        value = self.float_value
        if value < 0: mul = 1/mul
        value = value * mul + add * (self.maximum()-self.minimum())
        self.setValue(int(round(value)))
        self.float_value = value
