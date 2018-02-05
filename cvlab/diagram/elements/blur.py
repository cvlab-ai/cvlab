# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from builtins import range

from .base import *


class OpenCVBlur(NormalElement):
    name = "Blur transform"
    comment = "Simple blurring of the image"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("ratio", "Blur ratio", min_=1, max_=255)]

    def process_inputs(self, inputs, outputs, parameters):
        ratio = parameters["ratio"]
        outputs["output"] = Data(cv.blur(inputs["input"].value, (ratio, ratio)))



class GaussianBlur3D(NormalElement):
    name = 'Gaussian blur 3D'
    comment = 'Gaussian blur for 3D data'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('kernel', None, 9, 1, 999, 2),
                FloatParameter('sigmaX', 'sigmaX', min_=0),
                FloatParameter('sigmaY', 'sigmaY', min_=0),
                FloatParameter('sigmaZ', 'sigmaZ', min_=0),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['src'].value.copy()
        kernel = parameters['kernel']
        sigmaX = parameters['sigmaX']
        sigmaY = parameters['sigmaY']
        sigmaZ = parameters['sigmaZ']
        borderType = parameters['borderType']

        for z in image:
            self.may_interrupt()
            cv.GaussianBlur(z, (kernel, kernel), sigmaX, z, sigmaY, borderType)

        for x, y in itertools.product(range(image.shape[1]), range(image.shape[2])):
            self.may_interrupt()
            z = image[:, y, x]
            image[:, y, x] = cv.GaussianBlur(z, (1, kernel), sigmaZ, None, sigmaZ, borderType).reshape(-1)

        outputs['dst'] = Data(image)


register_elements_auto(__name__, locals(), "Blur", 5)

