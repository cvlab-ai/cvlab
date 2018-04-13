# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
from builtins import str, range

import itertools

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSlot

from ..diagram.parameters import *
from .highlighter import Highlighter


class GuiButtonParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter):
        super(GuiButtonParameter, self).__init__()
        self.parameter = parameter
        self.button = QtGui.QPushButton(self.parameter.name)
        self.button.setObjectName("ButtonParameterButton")
        self.button.clicked.connect(self.clicked)
        self.addWidget(self.button)

    @pyqtSlot()
    def clicked(self):
        self.parameter.clicked()


class GuiPathParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter):
        super(GuiPathParameter, self).__init__()
        self.parameter = parameter
        self.label = QtGui.QLabel(self.parameter.name)
        self.label.setObjectName("PathParameterName")
        self.path = QtGui.QLineEdit()
        self.path.setObjectName("PathParameterValue")
        self.path.setEnabled(False)
        self.path.setToolTip(self.path.text())
        self.browse = QtGui.QPushButton("...")
        self.browse.setObjectName("PathParameterButton")
        self.browse.clicked.connect(self.choose_path)
        self.addWidget(self.label)
        self.addWidget(self.path)
        self.addWidget(self.browse)
        self.parameter.value_changed.connect(self.on_value_changed)
        self.on_value_changed()

    @pyqtSlot()
    def on_value_changed(self):
        self.set_path_(self.parameter.get())

    def set_path_(self, path):
        if path != "":
            self.path.setText(path)
            self.path.setToolTip(path)
            self.parameter.set(str(path))

    @pyqtSlot()
    def choose_path(self):
        if self.parameter.save_mode:
            path = QtGui.QFileDialog.getSaveFileName(self.browse, "Save file...", self.parameter.get())
        else:
            path = QtGui.QFileDialog.getOpenFileName(self.browse, "Open file...", self.parameter.get())
        self.set_path_(str(path))


class GuiMultiPathParameter(GuiPathParameter):
    def __init__(self, parameter):
        super(GuiMultiPathParameter, self).__init__(parameter)

    @pyqtSlot()
    def on_value_changed(self):
        self.set_paths_(self.parameter.get())

    def set_paths_(self, paths):
        if len(paths) > 0:
            # TODO: put some images list instead of ...
            self.path.setText("...")
            self.path.setToolTip("...")
            self.parameter.set(paths)

    @pyqtSlot()
    def choose_path(self):
        paths = QtGui.QFileDialog.getOpenFileNames(self.browse, "Open multiple files...",
                                                   self.parameter.get()[0])
        self.set_paths_([str(p) for p in paths])



class GuiDirectoryParameter(GuiPathParameter):
    @pyqtSlot()
    def choose_path(self):
        path = QtGui.QFileDialog.getExistingDirectory(self.browse, "Open directory...", self.parameter.get())
        self.set_path_(str(path))



class GuiTextParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter, element):
        super(GuiTextParameter, self).__init__()
        assert isinstance(parameter, TextParameter)
        self.parameter = parameter
        self.element = element
        self.highlighter = None

        self.label = QtGui.QLabel(self.parameter.name)
        self.label.setObjectName("TextParameterName")

        self.button = QtGui.QPushButton("Edit code...")
        self.button.setObjectName("TextParameterButton")
        self.button.clicked.connect(self.edit_code)
        self.addWidget(self.label)
        self.addWidget(self.button)

        self.wnd = QtGui.QDialog(self.element)
        self.wnd.setModal(False)
        self.wnd.setLayout(QtGui.QVBoxLayout())
        self.wnd.setObjectName("CodeDialog")
        self.wnd.setWindowTitle(parameter.window_title)
        self.wnd.finished.connect(self.actualize)
        self.wnd_geometry = None

        if parameter.window_content:
            self.wnd.layout().addWidget(QtGui.QLabel(parameter.window_content))

        self.textedit = QtGui.QPlainTextEdit()
        self.textedit.setLineWrapMode(self.textedit.NoWrap)
        self.textedit.setWordWrapMode(QtGui.QTextOption.NoWrap)
        if parameter.live:
            self.textedit.textChanged.connect(self.actualize)
        self.wnd.layout().addWidget(self.textedit)

        ok_button = QtGui.QPushButton()
        ok_button.setText("OK")
        ok_button.clicked.connect(self.close_but_press)
        self.wnd.layout().addWidget(ok_button)

    @pyqtSlot()
    def edit_code(self):
        self.textedit.setPlainText(self.parameter.get())
        tab_width = QtGui.QFontMetrics(self.textedit.font()).width("    ")
        self.textedit.setTabStopWidth(tab_width)
        self.highlighter = Highlighter(self.textedit.document())

        if self.wnd_geometry:
            self.wnd.setGeometry(self.wnd_geometry)
        self.wnd.show()

    @pyqtSlot()
    def actualize(self):
        self.parameter.set(str(self.textedit.toPlainText()))

    @pyqtSlot()
    def close_but_press(self):
        self.wnd_geometry = self.wnd.geometry()
        self.wnd.accept()


class GuiIntParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter, element):
        super(GuiIntParameter, self).__init__()
        self.parameter = parameter
        self.element = element
        self.label = QtGui.QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.setVisible(False)
        self.slider.setRange(parameter.min, parameter.max)
        self.slider.setSingleStep(parameter.step)
        self.slider.setValue(parameter.get())
        self.slider.valueChanged.connect(self.gui_value_changed)
        self.addWidget(self.slider)
        element.param_sliders.append(self.slider)

        self.spin = QtGui.QSpinBox()
        self.spin.setRange(parameter.min, parameter.max)
        self.spin.setSingleStep(parameter.step)
        self.spin.setValue(parameter.get())
        self.spin.valueChanged.connect(self.gui_value_changed)
        self.addStretch()
        self.addWidget(self.spin)

        self.parameter.value_changed.connect(self.on_value_changed)

    @pyqtSlot(int)
    def gui_value_changed(self, value):
        if self.spin.value() != value or self.slider.value() != value:
            self.parameter.set(int(value))

    @pyqtSlot()
    def on_value_changed(self):
        value = self.parameter.get()
        if self.spin.value() != value:
            self.spin.setValue(value)
        if self.slider.value() != value:
            self.slider.setValue(value)


class GuiFloatParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter, element):
        super(GuiFloatParameter, self).__init__()
        self.parameter = parameter
        self.element = element
        self.changing = False

        self.label = QtGui.QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.setVisible(False)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(1)
        self.slider.setValue(self.value_to_slider(parameter.get()))
        self.slider.valueChanged.connect(self.slider_changed)
        self.addWidget(self.slider)
        element.param_sliders.append(self.slider)

        self.spin = QtGui.QDoubleSpinBox()
        self.spin.setRange(parameter.min, parameter.max)
        self.spin.setSingleStep(parameter.step)
        self.spin.setDecimals(6)
        self.spin.setValue(parameter.get())
        self.spin.valueChanged.connect(self.spin_changed)
        self.spin.stepBy = self.stepBy
        self.addWidget(self.spin)

        self.parameter.value_changed.connect(self.on_value_changed)
        self.ignore_changes = False

    def stepBy(self, steps):
        shift_pressed = (int(QtGui.QApplication.keyboardModifiers()) & QtCore.Qt.ShiftModifier) != 0
        if shift_pressed:
            QtGui.QDoubleSpinBox.stepBy(self.spin, steps * 100)
        else:
            QtGui.QDoubleSpinBox.stepBy(self.spin, steps)

    @pyqtSlot()
    def on_value_changed(self):
        value = self.parameter.get()
        if value != self.slider_to_value(self.slider.value()):
            self.slider.setValue(self.value_to_slider(value))
        if value != self.spin.value():
            self.spin.setValue(value)

    def value_to_slider(self, value):
        v = float(value - self.parameter.min) / (self.parameter.max - self.parameter.min)
        return round(v * 1000)

    def slider_to_value(self, value):
        return value * 0.001 * (self.parameter.max - self.parameter.min) + self.parameter.min

    @pyqtSlot(int)
    def slider_changed(self, value):
        if self.ignore_changes: return
        self.ignore_changes = True
        slider_value = self.slider_to_value(value)
        if slider_value != self.parameter.get():
            self.parameter.set(slider_value)
        self.ignore_changes = False

    @pyqtSlot(float)
    def spin_changed(self, value):
        if self.ignore_changes: return
        self.ignore_changes = True
        if value != self.parameter.get():
            self.parameter.set(value)
        self.ignore_changes = False


class GuiTwoIntsParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter, element):
        super(GuiTwoIntsParameter, self).__init__()

        self.parameter = parameter
        self.element = element

        self.label = QtGui.QLabel(self.parameter.name)
        self.addWidget(self.label)

        min_val = self.parameter.min_val
        max_val = self.parameter.max_val
        self.spin1 = QtGui.QSpinBox()
        self.spin1.setRange(min_val, max_val)
        self.addWidget(self.spin1)
        self.spin2 = QtGui.QSpinBox()
        self.spin2.setRange(min_val, max_val)
        self.addWidget(self.spin2)
        self.on_value_changed()

        #zdarzenie jest generowane od razu, wiec caly obiekt musi juz byc gotowy
        self.spin1.valueChanged.connect(self.spin1_value_changed)
        self.spin2.valueChanged.connect(self.spin2_value_changed)
        self.parameter.value_changed.connect(self.on_value_changed)

    def gui_value(self):
        return int(self.spin1.value()), int(self.spin2.value())

    def is_value_outdated(self):
        return self.parameter.get() != self.gui_value()

    @pyqtSlot()
    def spin1_value_changed(self):
        value = self.parameter.get()
        if value[0] != self.spin1.value():
            new_value = (self.spin1.value(), value[1])
            self.parameter.set(new_value)

    @pyqtSlot()
    def spin2_value_changed(self):
        value = self.parameter.get()
        if value[1] != self.spin2.value():
            new_value = (value[0], self.spin2.value())
            self.parameter.set(new_value)

    @pyqtSlot()
    def on_value_changed(self):
        if self.is_value_outdated():
            self.spin1.setValue(self.parameter.get()[0])
            self.spin2.setValue(self.parameter.get()[1])


class GuiComboboxParameter(QtGui.QHBoxLayout):
    def __init__(self, parameter, element):
        super(GuiComboboxParameter, self).__init__()
        self.parameter = parameter
        self.element = element

        self.label = QtGui.QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.combobox = QtGui.QComboBox()
        for text, value in parameter.values.items():
            self.combobox.addItem(text, value)
        self.combobox.currentIndexChanged.connect(self.combobox_value_changed)
        self.addWidget(self.combobox)

        self.parameter.value_changed.connect(self.on_value_changed)
        self.on_value_changed()

    def gui_value(self):
        value = self.combobox.itemData(self.combobox.currentIndex())
        if hasattr(value, "toPyObject"): value = value.toPyObject()
        return value

    def is_value_outdated(self):
        return self.parameter.get() != self.gui_value()

    @pyqtSlot()
    def combobox_value_changed(self):
        if self.is_value_outdated():
            self.parameter.set(self.gui_value())

    @pyqtSlot()
    def on_value_changed(self):
        if self.is_value_outdated():
            #index = self.combobox.findData(QtCore.QVariant(self.parameter.get()))
            index = self.find_index(self.parameter.get())
            self.combobox.setCurrentIndex(index)

    def find_index(self, data):
        for i in range(self.combobox.count()):
            d = self.combobox.itemData(i)
            if hasattr(d, "toPyObject"): d = d.toPyObject()
            if d == data: return i
        else:
            return -1

#
# class GuiMatrixParameter(QtGui.QHBoxLayout):
#
#     class Button(QtGui.QLabel):
#         pixel_size = (64, 64)
#
#         def __init__(self, parameter, x, y, step):
#             QtGui.QLabel.__init__(self)
#             self.parameter = parameter
#             self.matrix_x = x
#             self.matrix_y = y
#             self.step = step
#             self.setFixedSize(self.pixel_size[0], self.pixel_size[1])
#             self.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
#
#         def mousePressEvent(self, event):
#             if event.button() == QtCore.Qt.LeftButton:
#                 delta = self.step
#             elif event.button() == QtCore.Qt.RightButton:
#                 delta = -self.step
#             else:
#                 event.ignore()
#                 return
#             event.accept()
#             self.add_value(delta)
#
#         def mouseReleaseEvent(self, event):
#             if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
#                 event.accept()
#             else:
#                 event.ignore()
#
#         def wheelEvent(self, event):
#             event.accept()
#             delta = self.step if event.delta() > 0 else -self.step
#             self.add_value(delta)
#
#         def add_value(self, delta):
#             self.element.matrix[self.matrix_y,self.matrix_x] += delta
#             self.setText(str(self.element.matrix[self.matrix_y,self.matrix_x]))
#             self.element.recalculate(False, False, False, force_units_recalc=True)
#
#     def __init__(self, parameter, element):
#         super(GuiMatrixParameter, self).__init__()
#         assert isinstance(parameter, TextParameter)
#         self.parameter = parameter
#         self.element = element
#
#         self.label = QtGui.QLabel(self.parameter.name)
#         self.label.setObjectName("MatrixParameterName")
#
#         self.button = QtGui.QPushButton("Edit matrix...")
#         self.button.setObjectName("MatrixParameterButton")
#         self.button.clicked.connect(self.edit_code)
#         self.addWidget(self.label)
#         self.addWidget(self.button)
#
#         self.wnd = QtGui.QDialog(self.element)
#         self.wnd.setModal(True)
#         self.wnd.setLayout(QtGui.QVBoxLayout())
#         self.wnd.setObjectName("MatrixDialog")
#         self.wnd.setWindowTitle(parameter.window_title)
#         self.wnd.finished.connect(self.actualize)
#         self.wnd_geometry = None
#
#         self.size = (7,7)
#
#         self.edit_widget = QtGui.QWidget()
#         self.edit_widget.setLayout(QtGui.QGridLayout())
#         self.recreate_editor()
#
#         ok_button = QtGui.QPushButton()
#         ok_button.setText("OK")
#         ok_button.clicked.connect(self.close_but_press)
#         self.wnd.layout().addWidget(ok_button)
#
#     def recreate_editor(self):
#         QtGui.QWidget().setLayout(self.edit_widget.layout())
#         layout = QtGui.QGridLayout()
#         self.edit_widget.setLayout(layout)
#         w, h = self.last_parameters["size"]
#         dtype = self.last_parameters["type"]
#         step = self.last_parameters["step"]
#         self.matrix = np.zeros((h, w), dtype=dtype)
#         layout.setSpacing(0)
#         for x, y in itertools.product(xrange(w), xrange(h)):
#             pixel = self.Button(self, x, y, step)
#             pixel.setText("0")
#             layout.addWidget(pixel, y, x)
#
#     @pyqtSlot()
#     def edit_code(self):
#         self.textedit.setPlainText(self.parameter.get())
#         tab_width = QtGui.QFontMetrics(self.textedit.font()).width("    ")
#         self.textedit.setTabStopWidth(tab_width)
#         if self.wnd_geometry:
#             self.wnd.setGeometry(self.wnd_geometry)
#         self.wnd.show()
#
#     @pyqtSlot()
#     def actualize(self):
#         self.parameter.set(unicode(self.textedit.toPlainText()))
#
#     @pyqtSlot()
#     def close_but_press(self):
#         self.wnd_geometry = self.wnd.geometry()
#         self.wnd.accept()
#
#
