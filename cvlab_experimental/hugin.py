from cvlab.diagram.elements.base import *


class Enfuse(ProcessElement):
    name = "Enfuse"
    comment = "Enfuse program"

    command = "enfuse --exposure-weight={exposure} --saturation-weight={saturation} --contrast-weight={contrast} {hardmask} -o {outputs} {inputs}"

    def get_attributes(self):
        return [Input("inputs")], \
               [Output("output")], \
               []

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": ImageData()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_inputs(self, inputs, outputs, parameters):
        images = inputs['inputs'].desequence_all()
        if any(image is None for image in images): raise Exception("Image is empty")
        enfused = self.run_command(images, 1, exposure=0, saturation=0, contrast=1, hardmask="--hard-mask")
        outputs["output"] = ImageData(enfused)


class AlignStack(ProcessElement):
    name = "Align image stack"
    comment = "align_image_stack from Hugin"

    command = "align_image_stack  -midz -c 256 -g 1 --corr=0.3 -f 20 -a {output_dir}/ {inputs}"

    def get_attributes(self):
        return [Input("inputs")], \
               [Output("outputs")], \
               []

    def get_processing_units(self, inputs, parameters):
        outputs = {"outputs": inputs["inputs"].create_placeholder()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_inputs(self, inputs, outputs, parameters):
        images = inputs['inputs'].desequence_all()
        if any(image is None for image in images): raise Exception("Image is empty")
        aligned = self.run_command(images, 0)
        outputs["outputs"] = Sequence([ImageData(a) for a in aligned])


register_elements_auto(__name__, locals(), "Hugin", 10)