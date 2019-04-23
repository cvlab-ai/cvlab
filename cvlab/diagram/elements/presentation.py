from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

from .base import *


class ImagePreview(NormalElement):
    name = "Image preview"
    comment = "Shows an image using selected presentation method"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [ComboboxParameter("type", [
                ("No change", 0),
                ("Truncate to 0-255", 10),
                ("Truncate to 0-1", 20),
                ("Scale contrast to 0-1", 21),
                ("Mean -> 0.5 max/min -> 1/0", 22),
                ("Divide by 255.0", 23),
                ("0 -> 0.5, max/min -> 1/0", 30),
               ])]

    def process_inputs(self, inputs, outputs, parameters):
        i = inputs["input"].value
        t = parameters["type"]
        o = None
        if t == 0:
            o = i
        elif t == 10:
            o = i.clip(0, 255).astype(np.uint8)
        elif t == 20:
            o = i.astype(np.float32).clip(0.0, 1.0)
        elif t == 21:
            o = i.astype(np.float32)
            min_, max_, _, _ = cv.minMaxLoc(o.flatten())
            if min_ == max_:
                o = np.zeros(o.shape) + 0.5
            else:
                o = (o-min_)/(max_-min_)+min_
        elif t == 22:
            o = i.astype(np.float32)
            min_, max_, _, _ = cv.minMaxLoc(o.flatten())
            if min_ == max_:
                o = np.zeros(o.shape) + 0.5
            else:
                if len(o.shape) > 2:
                    mean = cv.mean(cv.mean(o)[:3])[0]
                else:
                    mean = cv.mean(o)[0]
                scale = 0.5/max(max_-mean, mean-min_)
                o = (o-mean)*scale+0.5
        elif t == 23:
            o = i / 255.
        elif t == 30:
            o = i.astype(np.float32)
            min_, max_, _, _ = cv.minMaxLoc(o.flatten())
            if min_ == max_:
                o = np.zeros(o.shape) + 0.5
            else:
                scale = 0.5/max(max_,-min_)
                o = o*scale+0.5
        outputs["output"] = Data(o)


class ImagePreview3D(ImagePreview):
    name = "Image preview 3D"
    comment = "Shows selected slice from 3D image"

    def get_attributes(self):
        attributes = super(ImagePreview3D, self).get_attributes()
        attributes[2].append(ComboboxParameter("axis", [("Z",0),("Y",1),("X",2)]))
        attributes[2].append(FloatParameter("slice", min_=0, max_=1, step=0.01))
        return attributes
    
    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value
        axis = parameters["axis"]
        slice = int(round(parameters["slice"] * (image.shape[axis]-1)))

        if axis == 0: slice = image[slice, ...]
        elif axis == 1: slice = image[:, slice, ...]
        elif axis == 2: slice = image[:, :, slice, ...]
        else: raise Exception("Unknown 'axis' parameter")

        inputs["input"].value = slice
        super(ImagePreview3D, self).process_inputs(inputs, outputs, parameters)


class Plot3d(NormalElement):
    name = "Plot 3D (wireframe)"
    comment = "Simple plot of 3D wireframe"

    def __init__(self):
        super(Plot3d, self).__init__()
        self.figure = Figure(figsize=(4, 4), dpi=90, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = self.figure.add_subplot(111, projection='3d')
        # self.axes.hold(False)
        self.plot_widget = FigureCanvasQTAgg(self.figure)
        self.axes.mouse_init()
        self.layout().addWidget(self.plot_widget)

    def get_attributes(self):
        return [Input("input")], [], []

    def process_inputs(self, inputs, outputs, parameters):
        i = inputs["input"].value
        X = i[0]
        Y = i[1]
        Z = i[2]

        self.axes.cla()
        self.axes.plot_wireframe(X, Y, Z)
        self.plot_widget.draw()
        self.plot_widget.updateGeometry()


register_elements_auto(__name__, locals(), "Visualization", 3)
