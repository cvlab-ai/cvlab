from .base import *


class Resizer(NormalElement):
    name = "Resize"
    comment = "Resizes its inputs"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("width", value=120), IntParameter("height", value=120)]

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = Data(cv.resize(inputs["input"].value, (parameters["width"], parameters["height"])))


class AutoResizer(NormalElement):
    name = "Automatic resizer"
    comment = "Resizes its inputs to match first input's size"

    def get_attributes(self):
        return [Input("main"), Input("others", multiple=True)], \
               [Output("main"), Output("others", desequencing=True)], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        size = inputs["main"].value.shape
        size = (size[1], size[0])
        outputs["main"] = Data(cv.resize(inputs["main"].value, size))
        outputs["others"] = Data(cv.resize(inputs["others"].value, size))


class Cropper(NormalElement):
    name = "Cropper"
    comment = "Crops the image"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("width", min_=1, max_=10000, value=128),
                IntParameter("height", min_=1, max_=10000, value=128),
                FloatParameter("x", "x", 0.5, 0, 1), FloatParameter("y", "y", 0.5, 0, 1)]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs["input"].value
        cw = int(img.shape[1] * parameters["x"])
        w2 = parameters["width"] // 2
        ch = int(img.shape[0] * parameters["y"])
        h2 = parameters["height"] // 2
        crop = img[max(0, ch - h2):ch + h2, max(0, cw - w2):cw + w2]
        outputs["output"] = Data(crop)



class AutoCrop(NormalElement):
    name = "Auto cropper"
    comment = "Automatically removes borders from image"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [FloatParameter("thr", "Threshold", 0), ComboboxParameter("thr_type", [("lighter", "lighter"), ("darker", "darker")], "Crop values")]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value
        threshold = parameters["thr"]
        threshold_type = parameters["thr_type"]

        if threshold_type == "lighter":
            if len(image.shape) == 3:
                i = image.min(axis=2)
            else: i = image
            mask = i >= threshold
        elif threshold_type == "darker":
            if len(image.shape) == 3:
                i = image.max(axis=2)
            else: i = image
            mask = i <= threshold
        else:
            raise Exception("Wrong threshold type")

        horizontal = np.min(mask, 0).tolist()
        vertical = np.min(mask, 1).tolist()

        assert isinstance(horizontal, list)
        assert isinstance(vertical, list)

        left = horizontal.index(0)
        right_margin = horizontal[::-1].index(0)
        right = len(horizontal) - right_margin

        top = vertical.index(0)
        bottom_margin = vertical[::-1].index(0)
        bottom = len(vertical) - bottom_margin

        cropped = image[top:bottom, left:right, ...] + 0

        outputs["output"] = ImageData(cropped)


register_elements_auto(__name__, locals(), "Transforms", 5)

