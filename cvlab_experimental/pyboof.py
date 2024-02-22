import pyboof as pb

from cvlab.diagram.elements.base import *


# create a class which inherits NormalElement
class BarcodeReader(NormalElement):
    name = "Barcode reader"
    comment = "Detects barcodes, QR codes, Aztec codes and more"
    package = "PyBoof library"

    types = {
        "Aztec": "aztec",
        "QR": "qrcode",
        "Micro QR": "microqr",
    }

    def get_attributes(self):
        return [Input("input", name="Input")], \
               [Output("output", name="Detections"), Output("failures", name="Failed detections")], \
               [ComboboxParameter("type", name="Barcode type", values=self.types)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value

        if image.dtype == np.float32:
            image = (image*255).clip(0,255).astype(np.uint8)

        factory = pb.FactoryFiducial(image.dtype)
        detector = getattr(factory, parameters['type'])()

        image = pb.ndarray_to_boof(image)

        detector.detect(image)

        codes = [
            [
                d.message,
                [[v.y, v.x] for v in d.bounds.vertexes]
            ]
            for d in detector.detections
        ]
        outputs["output"] = Data(codes)

        codes = [
            [
                d.message,
                [[v.y, v.x] for v in d.bounds.vertexes]
            ]
            for d in detector.failures
        ]
        outputs["failures"] = Data(codes)


# register all elements defined above in the palette
register_elements_auto(__name__, locals(), "PyBoof", 1)
