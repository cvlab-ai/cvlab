from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cvlab.view.spin_widget import *
from ..diagram.parameters import *
from .highlighter import Highlighter


class GuiBaseParameter(QHBoxLayout):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(2)


class GuiButtonParameter(GuiBaseParameter):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.button = QPushButton(self.parameter.name)
        self.button.setObjectName("ButtonParameterButton")
        self.button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.button.clicked.connect(self.clicked)
        self.addWidget(self.button)

    @pyqtSlot()
    def clicked(self):
        self.parameter.clicked()


class GuiPathParameter(GuiBaseParameter):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.label = QLabel(self.parameter.name)
        self.label.setObjectName("PathParameterName")
        self.path = QLineEdit()
        self.path.setObjectName("PathParameterValue")
        self.path.setEnabled(False)
        self.path.setToolTip(self.path.text())
        self.browse = QPushButton("...")
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
            path, _ = QFileDialog.getSaveFileName(self.browse, "Save file...", self.parameter.get())
        else:
            path, _ = QFileDialog.getOpenFileName(self.browse, "Open file...", self.parameter.get())
        self.set_path_(str(path))


class GuiMultiPathParameter(GuiPathParameter):
    def __init__(self, parameter):
        super().__init__(parameter)

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
        paths, _ = QFileDialog.getOpenFileNames(self.browse, "Open multiple files...", self.parameter.get()[0])
        self.set_paths_([str(p) for p in paths])


class GuiDirectoryParameter(GuiPathParameter):
    @pyqtSlot()
    def choose_path(self):
        path = QFileDialog.getExistingDirectory(self.browse, "Open directory...", self.parameter.get())
        self.set_path_(str(path))


class GuiTextParameter(GuiBaseParameter):
    def __init__(self, parameter, element):
        super().__init__(parameter)
        assert isinstance(parameter, TextParameter)
        self.element = element
        self.highlighter = None

        self.label = QLabel(self.parameter.name)
        self.label.setObjectName("TextParameterName")

        self.button = QPushButton("Edit code...")
        self.button.setObjectName("TextParameterButton")
        self.button.clicked.connect(self.edit_code)
        self.addWidget(self.label)
        self.addWidget(self.button)

        self.wnd = QDialog(self.element)
        self.wnd.setModal(False)
        self.wnd.setLayout(QVBoxLayout())
        self.wnd.setObjectName("CodeDialog")
        self.wnd.setWindowTitle(parameter.window_title)
        desktop = QApplication.instance().desktop()
        self.wnd.resize(desktop.screenGeometry(desktop.screenNumber(self.element)).width() // 2,
                        desktop.screenGeometry(desktop.screenNumber(self.element)).height() // 2)
        self.wnd.finished.connect(self.actualize)
        self.wnd_geometry = None

        if parameter.window_content:
            self.wnd.layout().addWidget(QLabel(parameter.window_content))

        self.textedit = QPlainTextEdit()
        self.textedit.setLineWrapMode(self.textedit.NoWrap)
        self.textedit.setWordWrapMode(QTextOption.NoWrap)
        if parameter.live:
            self.textedit.textChanged.connect(self.actualize)
        self.wnd.layout().addWidget(self.textedit)

        ok_button = QPushButton()
        ok_button.setText("OK")
        ok_button.clicked.connect(self.close_but_press)
        self.wnd.layout().addWidget(ok_button)

    @pyqtSlot()
    def edit_code(self):
        self.textedit.setPlainText(self.parameter.get())
        tab_width = QFontMetrics(self.textedit.font()).width("    ")
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


class GuiIntParameter(GuiBaseParameter):
    def __init__(self, parameter, element):
        super().__init__(parameter)
        self.element = element
        self.label = QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setVisible(False)
        self.slider.setRange(parameter.min, parameter.max)
        self.slider.setSingleStep(parameter.step)
        self.slider.setValue(parameter.get())
        self.slider.valueChanged.connect(self.gui_value_changed)
        self.addWidget(self.slider)
        element.param_sliders.append(self.slider)

        self.spin = SpinBoxEx()
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


class GuiFloatParameter(GuiBaseParameter):
    def __init__(self, parameter, element):
        super().__init__(parameter)
        self.element = element
        self.changing = False

        self.label = QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setVisible(False)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(1)
        self.slider.setValue(self.value_to_slider(parameter.get()))
        self.slider.valueChanged.connect(self.slider_changed)
        self.addWidget(self.slider)
        element.param_sliders.append(self.slider)

        self.spin = DoubleSpinBoxEx()
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
        shift_pressed = (int(QApplication.keyboardModifiers()) & QtCore.Qt.ShiftModifier) != 0
        if shift_pressed:
            QDoubleSpinBox.stepBy(self.spin, steps * 100)
        else:
            QDoubleSpinBox.stepBy(self.spin, steps)

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


class GuiMultiNumberParameter(GuiBaseParameter):
    def __init__(self, parameter, element, count, type):
        super().__init__(parameter)

        self.element = element
        self.count = count
        self.type = type

        self.label = QLabel(self.parameter.name)
        self.addWidget(self.label)

        min_ = self.parameter.min
        max_ = self.parameter.max

        self.spins = []
        for i in range(count):
            spin = SpinBoxEx()
            spin.setRange(min_, max_)
            self.addWidget(spin)
            self.spins.append(spin)

        self.on_value_changed()

        # event is generated instantly, so the whole object must be ready
        for spin in self.spins:
            spin.valueChanged.connect(self.spin_value_changed)

        self.parameter.value_changed.connect(self.on_value_changed)

    def gui_value(self):
        return tuple(self.type(spin.value()) for spin in self.spins)

    @pyqtSlot()
    def spin_value_changed(self):
        parameter_value = self.parameter.get()
        spin_value = self.gui_value()
        if parameter_value != spin_value:
            self.parameter.set(spin_value)

    @pyqtSlot()
    def on_value_changed(self):
        value = self.parameter.get()
        gui_value = self.gui_value()
        if value != gui_value:
            for spin, value in zip(self.spins, value):
                spin.setValue(value)


class GuiComboboxParameter(GuiBaseParameter):
    def __init__(self, parameter, element):
        super().__init__(parameter)
        self.element = element

        self.label = QLabel(self.parameter.name)
        self.addWidget(self.label)

        self.combobox = QComboBox()
        self.combobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combobox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        self.combobox.setContentsMargins(0,0,0,0)
        self.combobox.setMinimumContentsLength(1)
        self.combobox.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

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
# class GuiMatrixParameter(GuiBaseParameter):
#
#     class Button(QLabel):
#         pixel_size = (64, 64)
#
#         def __init__(self, parameter, x, y, step):
#             QLabel.__init__(self)
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
#         self.label = QLabel(self.parameter.name)
#         self.label.setObjectName("MatrixParameterName")
#
#         self.button = QPushButton("Edit matrix...")
#         self.button.setObjectName("MatrixParameterButton")
#         self.button.clicked.connect(self.edit_code)
#         self.addWidget(self.label)
#         self.addWidget(self.button)
#
#         self.wnd = QDialog(self.element)
#         self.wnd.setModal(True)
#         self.wnd.setLayout(QVBoxLayout())
#         self.wnd.setObjectName("MatrixDialog")
#         self.wnd.setWindowTitle(parameter.window_title)
#         self.wnd.finished.connect(self.actualize)
#         self.wnd_geometry = None
#
#         self.size = (7,7)
#
#         self.edit_widget = QWidget()
#         self.edit_widget.setLayout(QGridLayout())
#         self.recreate_editor()
#
#         ok_button = QPushButton()
#         ok_button.setText("OK")
#         ok_button.clicked.connect(self.close_but_press)
#         self.wnd.layout().addWidget(ok_button)
#
#     def recreate_editor(self):
#         QWidget().setLayout(self.edit_widget.layout())
#         layout = QGridLayout()
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
#         tab_width = QFontMetrics(self.textedit.font()).width("    ")
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
