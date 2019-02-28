#
# Creating your own Element for CV Lab
#

# import whatever libraries you want

import cv2 as cv

# also import this
from cvlab.diagram.elements.base import *


# create a class which inherits NormalElement
class MySampleElement(NormalElement):
    name = "My sample element"
    comment = "My first self-created element, hurray! :)"
    package = "My private elements"

    def get_attributes(self):
        return [Input("input", name="Input")], \
               [Output("output-1", name="First output"), Output("output-2", name="Second output")], \
               [IntParameter("param", "Some integer parameter", min_=1, max_=255)]

    def process_inputs(self, inputs, outputs, parameters):
        # read the parameters and inputs
        image = inputs["input"].value
        param = parameters["param"]

        o1 = cv.blur(image, (param, param))
        self.may_interrupt()  # allow interruptions here
        o2 = image + param

        # set the outputs
        outputs["output-1"] = Data(o1)
        outputs["output-2"] = Data(o2)


# register all elements defined above in the palette
register_elements_auto(__name__, locals(), "My private elements", 1)

