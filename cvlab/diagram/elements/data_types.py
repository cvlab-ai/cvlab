from .base import *


class ColorConverter(NormalElement):
    name = "Color converter"
    comment = "Converts image to another color space"
    package = "Color"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [ComboboxParameter("code", [
                   ('No change', None),
                   ('BGR -> Gray', cv.COLOR_BGR2GRAY),
                   ('RGB -> Gray', cv.COLOR_RGB2GRAY),
                   ('Gray -> BGR', cv.COLOR_GRAY2BGR),
                   ('Gray -> RGB', cv.COLOR_GRAY2RGB),
                   ('BGR - > HSV', cv.COLOR_BGR2HSV),
                   ('RGB - > HSV', cv.COLOR_RGB2HSV),
                   ('HSV - > BGR', cv.COLOR_HSV2BGR),
                   ('HSV - > RGB', cv.COLOR_HSV2RGB)
               ])]

    def process_inputs(self, inputs, outputs, parameters):
        if parameters["code"] is None:
            outputs["output"] = Data(inputs["input"].value.copy())
        else:
            outputs["output"] = Data(cv.cvtColor(inputs["input"].value, parameters["code"]))


class TypeConverter(NormalElement):
    name = "Type converter"
    comment = "Converts contents of the image to another internal data type"
    package = "Type conversion"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [ComboboxParameter("type", {
                   '8-bit unsigned': np.uint8,
                   '32-bit float': np.float32
               })]

    def process_inputs(self, inputs, outputs, parameters):
        type = parameters["type"]
        input_ = inputs["input"].value
        if type == input_.dtype:
            output = input_
        elif input_.dtype == np.uint8 and type == np.float32:
            output = np.float32(input_) * 1./255
        elif input_.dtype == np.float32 and type == np.uint8:
            output = np.uint8(input_ * 255.)
        else:
            raise ProcessingError("Wrong convertion type")
        outputs["output"] = Data(output)


class ChannelSplitter(NormalElement):
    name = "Channel splitter"
    comment = "Splits the image into channels"
    package = "Channels"

    def get_attributes(self):
        return [Input("input")], \
               [Output("1"), Output("2"), Output("3"), Output("4")], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        channels = cv.split(inputs["input"].value)
        for i, image in enumerate(channels):
            outputs[str(i + 1)] = Data(image)


class ChannelMerger(NormalElement):
    name = "Channel merger"
    comment = "Merges the channels into an image"
    package = "Channels"

    def get_attributes(self):
        return [Input("1"), Input("2", optional=True), Input("3", optional=True), Input("4", optional=True)], \
               [Output("output")], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        r = []
        od = OrderedDict(sorted(inputs.items()))
        for i, image in enumerate(od.values()):
            if image.type() != Data.NONE:
                r.append(image.value)
        img = cv.merge(r)
        outputs["output"] = Data(img)


class ColorspaceExtractor(NormalElement):
    name = "Color space extractor"
    comment = "Splits the image into RGB and HSV"
    package = "Color"

    def get_attributes(self):
        return [Input("input")], \
               [Output("R"), Output("G"), Output("B"), Output("H"), Output("S"), Output("V")], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        bgr = cv.split(inputs["input"].value)
        hsv = cv.split(cv.cvtColor(inputs["input"].value, cv.COLOR_BGR2HSV_FULL))
        outputs["B"] = Data(bgr[0])
        outputs["G"] = Data(bgr[1])
        outputs["R"] = Data(bgr[2])
        outputs["H"] = Data(hsv[0])
        outputs["S"] = Data(hsv[1])
        outputs["V"] = Data(hsv[2])

register_elements_auto(__name__, locals(), "Data types")
