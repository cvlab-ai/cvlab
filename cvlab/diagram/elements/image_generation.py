import math

from .base import *


class InPaint(NormalElement):
    name = "Inpaint"
    comment = "Inpaint"
    package = "Photo"

    def get_attributes(self):
        return [Input("input"), Input("mask")], \
               [Output("output")], \
               [IntParameter("radius", value=3, min_=1, max_=100), IntParameter("flags")]

    def process_channels(self, inputs, outputs, parameters):
        output = cv.inpaint(inputs["input"].value, inputs["mask"].value, parameters["radius"], parameters["flags"])
        outputs["output"] = Data(output)


class GetGaborKernel(InputElement):
    name = "getGaborKernel"
    comment = "getGaborKernel"
    package = "Data generation"

    def get_attributes(self):
        return [], [Output("output")], \
               [
                    IntParameter("ksize", "ksize", 1, min_=1, max_=255, step=2),
                    FloatParameter("sigma", "sigma", 0.0, min_=0.0, max_=100.0),
                    FloatParameter("theta", "theta", 0.0, min_=0.0, max_=math.pi, step=0.01),
                    FloatParameter("lambda", "lambda", 0.0, min_=0.0, max_=100.0),
                    FloatParameter("gamma", "gamma", math.pi * 0.5, min_=0.0, max_=100.0, step=0.1),
                    FloatParameter("psi", "psi", 0.0, min_=0.0, max_=100.0),
                    ComboboxParameter("ktype", values={
                        "CV_64F": cv.CV_64F,
                        "CV_32F": cv.CV_32F
                    })
               ]

    def process_inputs(self, inputs, outputs, parameters):
        ksize = parameters["ksize"]
        sigma = parameters["sigma"]
        theta = parameters["theta"]
        lambda_ = parameters["lambda"]
        gamma = parameters["gamma"]
        psi = parameters["psi"]
        ktype = parameters["ktype"]

        image = cv.getGaborKernel(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=lambda_, gamma=gamma, psi=psi, ktype=ktype)
        outputs["output"] = Data(image)


register_elements_auto(__name__, locals(), "Image generation", 7)
