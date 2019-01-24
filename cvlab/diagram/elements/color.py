from .base import *


class ColorNormalizer(NormalElement):
    name = "Color normalizer"
    comment = "Normalizes each color by given mean and standard deviation"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [FloatParameter("mean", value=128, min_=-255, max_=255),
                FloatParameter("std dev", value=64, min_=-255, max_=255)]

    def process_channels(self, inputs, outputs, parameters):
        image = inputs["input"].value
        outmean = parameters["mean"]
        outstddev = parameters["std dev"]

        mean, stddev = cv.meanStdDev(image)
        if stddev == 0: stddev = 1

        output = (image.astype(np.float32) - mean) * (outstddev/stddev) + outmean

        outputs["output"] = Data(np.clip(output, 0, 255).astype(image.dtype))


class OpenCVInRange(NormalElement):
    name = "InRange"
    comment = "Inrange"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    IntParameter("min val", "Minimum value", 0, min_=0, max_=255),
                    IntParameter("max val", "Maximum value", 255, min_=0, max_=255)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        min = parameters["min val"]
        max = parameters["max val"]

        image = cv.inRange(image, min, max)
        outputs["output"] = Data(image)


class OpenCVInRange3D(NormalElement):
    name = "InRange3D"
    comment = "Inrange"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    IntParameter("1 min val", "1:Minimum value", 0, min_=0, max_=255),
                    IntParameter("1 max val", "1:Maximum value", 255, min_=0, max_=255),
                    IntParameter("2 min val", "2:Minimum value", 0, min_=0, max_=255),
                    IntParameter("2 max val", "2:Maximum value", 255, min_=0, max_=255),
                    IntParameter("3 min val", "3:Minimum value", 0, min_=0, max_=255),
                    IntParameter("3 max val", "3:Maximum value", 255, min_=0, max_=255)
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        min1 = parameters["1 min val"]
        max1 = parameters["1 max val"]
        min2 = parameters["2 min val"]
        max2 = parameters["2 max val"]
        min3 = parameters["3 min val"]
        max3 = parameters["3 max val"]

        image = cv.inRange(image, (min1, min2, min3), (max1, max2, max3))
        outputs["output"] = Data(image)


class ContrastChange(NormalElement):
    name = "Contrast change"
    comment = "Changes contrast of the image"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [FloatParameter("factor")]

    def process_channels(self, inputs, outputs, parameters):
        image = inputs["input"].value

        avg = cv.mean(image)
        output = cv.add(cv.subtract(np.float64(image), avg) * parameters["factor"], avg)
        output = cv.add(np.zeros(output.shape, image.dtype), output, dtype=cvtypes[image.dtype.name])

        outputs["output"] = Data(output)


register_elements_auto(__name__, locals(), "Color", 4)
