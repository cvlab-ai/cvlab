from .base import *


class OpenCVCanny(NormalElement):
    name = "Canny transform"
    comment = "Canny edge detector"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("thr1", "Threshold 1", min_=0, max_=255),
                IntParameter("thr2", "Threshold 2", min_=0, max_=255)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value

        color = False
        if len(image.shape) >= 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            color = True

        output = cv.Canny(image, parameters["thr1"], parameters["thr2"])

        if color:
            output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

        outputs["output"] = Data(output)


register_elements_auto(__name__, locals(), "Features 2D", 5)
