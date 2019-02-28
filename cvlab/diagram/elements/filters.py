from .base import *


class OpenCVMorphologyEx(NormalElement):
    name = "Morphological transform"
    comment = "Advanced morphological transform"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    ComboboxParameter("operation", values=OrderedDict([
                        ("MORPH_OPEN", cv.MORPH_OPEN),
                        ("MORPH_CLOSE", cv.MORPH_CLOSE),
                        ("MORPH_GRADIENT", cv.MORPH_GRADIENT),
                        ("MORPH_TOPHAT ", cv.MORPH_TOPHAT),
                        ("MORPH_BLACKHAT", cv.MORPH_BLACKHAT)
                    ])),
                    ComboboxParameter("element type", values=OrderedDict([
                        ("MORPH_RECT", cv.MORPH_RECT),
                        ("MORPH_ELLIPSE", cv.MORPH_ELLIPSE),
                        ("MORPH_CROSS", cv.MORPH_CROSS)
                    ])),
                    IntParameter("element size", "Size of structuring element", 5, min_=1, max_=255, step=2),
                    IntParameter("iterations", "Number of iterations", 1, min_=0, max_=255)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        operation = parameters["operation"]
        element_type = parameters["element type"]
        element_size = parameters["element size"]
        iterations = parameters["iterations"]

        element = cv.getStructuringElement(element_type, (element_size, element_size))
        image = cv.morphologyEx(image, operation, element, iterations=iterations,)
        outputs["output"] = Data(image)


class OpenCVDilate(NormalElement):
    name = "Dilate"
    comment = "Dilation morphological transform"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    ComboboxParameter("element type", values=OrderedDict([
                        ("MORPH_RECT", cv.MORPH_RECT),
                        ("MORPH_ELLIPSE", cv.MORPH_ELLIPSE),
                        ("MORPH_CROSS", cv.MORPH_CROSS)
                    ])),
                    IntParameter("element size", "Size of structuring element", 5, min_=1, max_=255, step=2),
                    IntParameter("iterations", "Number of iterations", 1, min_=0, max_=255)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        element_type = parameters["element type"]
        element_size = parameters["element size"]
        iterations = parameters["iterations"]

        element = cv.getStructuringElement(element_type, (element_size, element_size))
        image = cv.dilate(image, element, iterations=iterations)
        outputs["output"] = Data(image)


class OpenCVErode(NormalElement):
    name = "Erode"
    comment = "Erosion morphological transform"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    ComboboxParameter("element type", values=OrderedDict([
                        ("MORPH_RECT", cv.MORPH_RECT),
                        ("MORPH_ELLIPSE", cv.MORPH_ELLIPSE),
                        ("MORPH_CROSS", cv.MORPH_CROSS)
                    ])),
                    IntParameter("element size", "Size of structuring element", 5, min_=1, max_=255, step=2),
                    IntParameter("iterations", "Number of iterations", 1, min_=0, max_=255)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        element_type = parameters["element type"]
        element_size = parameters["element size"]
        iterations = parameters["iterations"]

        element = cv.getStructuringElement(element_type, (element_size, element_size))
        image = cv.erode(image, element, iterations=iterations)
        outputs["output"] = Data(image)


register_elements_auto(__name__, locals(), "Filters", 5)
