from .base import *


class Rotate(NormalElement):
    name = "Rotate"
    comment = "Rotates using multiple 90"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [
                    IntParameter("angle", "Angle", 0, min_=0, max_=270, step=90),
               ]

    def process_inputs(self, inputs, outputs, parameters):
        image = np.copy(inputs["input"].value)
        angle = parameters["angle"]
        angle_rad = np.radians(angle)
        h, w = image.shape[:2]

        trans1 = np.array([[1, 0, -w / 2],
                           [0, 1, -h / 2],
                           [0, 0, 1]],
                          dtype=np.float32)
        if angle % 180 == 90:
            h, w = w, h
        trans2 = np.array([[1, 0, w / 2],
                           [0, 1, h / 2],
                           [0, 0, 1]],
                          dtype=np.float32)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 1]],
                          dtype=np.float32)

        mat = np.dot(np.dot(trans2, rot), trans1)

        out = cv.warpAffine(image, mat[:2], (w, h))

        outputs["output"] = Data(out)


register_elements_auto(__name__, locals(), "Transforms", 5)