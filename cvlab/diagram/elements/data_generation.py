import itertools

from PyQt5.QtCore import Qt

from .base import *


class MatrixGenerator(InputElement):
    name = "Matrix generator"
    comment = "Generates selected matrix"

    def get_attributes(self):
        return [], [Output("output")], [
            ComboboxParameter("type", {"Rectangle": 1, "Circle": 2, "Rectangle [mono]": 3, "Circle [mono]": 4}),
            IntParameter("size", min_=1)]

    def process_inputs(self, inputs, outputs, parameters):
        size = parameters["size"]
        output = None

        if parameters["type"] == 1: #rectangle
            output = np.empty((size, size, 3), np.uint8)
            output.fill(255)

        elif parameters["type"] == 2: #circle
            output = np.zeros((size, size, 3), np.uint8)
            cv.circle(output, (size // 2, size // 2), size // 2, (255, 255, 255), -1)

        elif parameters["type"] == 3: #rectangle
            output = np.empty((size, size), np.uint8)
            output.fill(255)

        elif parameters["type"] == 4: #circle
            output = np.zeros((size, size), np.uint8)
            cv.circle(output, (size // 2, size // 2), size // 2, 255, -1)

        if output is None: raise ProcessingError("Wrong matrix type")

        outputs["output"] = Data(output)


class MatrixEditor(InputElement):
    name = "Matrix editor"
    comment = "Allows to create a matrix pixel by pixel"

    def __init__(self):
        super(MatrixEditor, self).__init__()
        self.last_parameters = {}
        self.edit_widget = QWidget()
        self.matrix = None
        self.layout().addWidget(self.edit_widget)
        self.edit_widget.setLayout(QGridLayout())

    def get_attributes(self):
        return [], [Output("output")], [
            ComboboxParameter("type", {"8-bit unsigned":np.uint8, "8-bit signed":np.int8, "32-bit float": np.float32}),
            SizeParameter("size"),
            FloatParameter("step", value=1)]

    def process_inputs(self, inputs, outputs, parameters):
        if parameters != self.last_parameters:
            self.last_parameters = parameters.copy()
            self.recreate_editor()
        outputs["output"] = Data(self.matrix+0)

    class Button(QLabel):
        pixel_size = (64, 64)

        def __init__(self, element, x, y, step):
            QLabel.__init__(self)
            self.element = element
            self.matrix_x = x
            self.matrix_y = y
            self.step = step
            self.setFixedSize(self.pixel_size[0], self.pixel_size[1])
            self.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                delta = self.step
            elif event.button() == Qt.RightButton:
                delta = -self.step
            else:
                event.ignore()
                return
            event.accept()
            self.add_value(delta)

        def mouseReleaseEvent(self, event):
            if event.button() in (Qt.LeftButton, Qt.RightButton):
                event.accept()
            else:
                event.ignore()

        def wheelEvent(self, event):
            event.accept()
            delta = self.step if event.angleDelta().y() > 0 else -self.step
            self.add_value(delta)

        def add_value(self, delta):
            self.element.matrix[self.matrix_y,self.matrix_x] += delta
            self.setText(str(self.element.matrix[self.matrix_y,self.matrix_x]))
            self.element.recalculate(False, False, False, force_units_recalc=True)


    def recreate_editor(self):
        QWidget().setLayout(self.edit_widget.layout())
        layout = QGridLayout()
        self.edit_widget.setLayout(layout)
        w, h = self.last_parameters["size"]
        dtype = self.last_parameters["type"]
        step = self.last_parameters["step"]
        self.matrix = np.zeros((h, w), dtype=dtype)
        layout.setSpacing(0)
        for x, y in itertools.product(range(w), range(h)):
            pixel = self.Button(self, x, y, step)
            pixel.setText("0")
            layout.addWidget(pixel, y, x)


class MatrixEditor2(InputGuiElement, ThreadedElement):
    name = "Matrix editor 2"
    comment = "Allows to create a matrix pixel by pixel"

    def get_attributes(self):
        return [], [Output("output")], [MatrixParameter("matrix")]

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = Data(parameters["matrix"])


class OpenCVGetStructuringElement(InputElement):
    name = "Structuring Element"
    comment = "Returns structuring element"

    def get_attributes(self):
        return [], \
               [Output("output")], \
               [
                    ComboboxParameter("Element type", values=OrderedDict([
                        ("MORPH_RECT", cv.MORPH_RECT),
                        ("MORPH_ELLIPSE", cv.MORPH_ELLIPSE),
                        ("MORPH_CROSS", cv.MORPH_CROSS)
                    ])),
                    IntParameter("x element size", "Width of structuring element", 5, min_=1, max_=255, step=1),
                    IntParameter("y element size", "Height of structuring element", 5, min_=1, max_=255, step=1),
               ]

    def process_inputs(self, inputs, outputs, parameters):
        element_type = parameters["Element type"]
        x_element_size = parameters["x element size"]
        y_element_size = parameters["y element size"]

        element = cv.getStructuringElement(element_type, (x_element_size, y_element_size))
        outputs["output"] = Data(element)


register_elements_auto(__name__, locals(), "Data generation", 7)
