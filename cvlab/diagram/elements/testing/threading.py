# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from builtins import range

from cvlab.diagram.elements.base import *


class OpenCVBlurSingle(FunctionGuiElement, ThreadedElement):
    name = "Blur transform [not threaded]"
    comment = "Simple blurring of the image [not threaded]"

    def get_attributes(self):
        return [Input("input")], [Output("output")], [IntParameter("ratio", "Blur ratio")]

    def process_inputs(self, inputs, outputs, parameters):
        ratio = parameters["ratio"]
        outputs["output"] = Data(cv.blur(inputs["input"].value, (ratio, ratio)))


class SlowThreadedElement(FunctionGuiElement, ThreadedElement):
    name = "Slow threaded element"
    comment = ""

    def get_attributes(self):
        return [Input("input")], [Output("output")], [IntParameter("time", "Time")]

    def process_inputs(self, inputs, outputs, parameters):
        time = parameters["time"]
        for i in range(time):
            outputs["output"] = Data(copy(inputs["input"].value))


class SlowElement(FunctionGuiElement, ThreadedElement):
    name = "Slow element"
    comment = ""

    def get_attributes(self):
        return [Input("input")], [Output("output")], [IntParameter("ratio", "Blur ratio")]

    def process_inputs(self, inputs, outputs, parameters):
        time = parameters["time"]
        for i in range(time):
            outputs["output"] = Data(copy(inputs["input"].value))

