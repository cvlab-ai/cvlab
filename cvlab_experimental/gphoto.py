from cvlab.diagram.elements.base import *


class DSLR(ProcessElement):
    name = "DSLR Camera"
    comment = "DSLR Camera photo capture"

    lock = Lock()
    command = "cd /tmp; killall gvfsd-gphoto2; killall gvfs-gphoto2-volume-monitor; gphoto2 --capture-image-and-download --force-overwrite --filename={outputs}"

    def get_attributes(self):
        return [], \
               [Output("output")], \
               [ButtonParameter("capture", self.capture)]

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": ImageData()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_inputs(self, inputs, outputs, parameters):
        with self.lock:
            image = self.run_command([], 1)
        outputs["output"] = ImageData(image)

    def capture(self):
        self.recalculate(True, True, True)

register_elements_auto(__name__, locals(), "Camera", 10)

