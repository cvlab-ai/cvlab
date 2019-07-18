import os
from subprocess import check_call
from tempfile import mkstemp, mkdtemp

import cv2 as cv

from ...core.threaded_element import ThreadedElement
from ...view.elements import *
from ... import CVLAB_DIR
from ..data import *
from ..connectors import *
from ..errors import *

from . import register_elements_auto, register_elements


cvtypes = {
    "uint8": cv.CV_8U,
    "uint16": cv.CV_16U,
    "int8": cv.CV_8S,
    "int16": cv.CV_16S,
    "float32": cv.CV_32F,
    "float64": cv.CV_64F
}


class NormalElement(FunctionGuiElement, ThreadedElement):
    pass


class InputElement(InputGuiElement, ThreadedElement):
    def get_source(self):
        name = self.__class__.__name__.lower()
        code = None
        outputs = list(self.outputs.keys())
        return name, code, outputs


class SequenceToDataElement(NormalElement):
    """processes its inputs as-they-are, returns only simple Data type"""

    def get_processing_units(self, inputs, parameters):
        outputs = {name: Data() for name in self.outputs}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs



class MultiInputOneOutputElement(NormalElement):
    def get_attributes(self):
        return [Input("inputs", multiple=True)], [Output("output")], []

    def get_processing_units(self, inputs, parameters):
        input_list = inputs["inputs"].value
        if len(input_list) == 1 and input_list[0].type() == Data.SEQUENCE:
            ins = {i: d for i, d in enumerate(input_list[0].value)}
        else:
            ins = {i: d for i, d in enumerate(input_list)}
        return NormalElement.get_default_processing_units(self, ins, parameters, ["output"])


class SequenceToSequenceElement(NormalElement):
    num_outputs = 8
    def get_attributes(self):
        return [Input("inputs", multiple=True)], [Output("output")], []

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": Sequence([ImageData() for _ in range(self.num_outputs)])}
        return [ProcessingUnit(self, inputs, parameters, outputs)], outputs


class ProcessElement(NormalElement):
    command = "command_name {args} {inputs} {outputs}"

    def run_command(self, images, output_count, **args):
        in_files = [mkstemp(".bmp", "cvlab_in_")[1] for _ in images]

        if output_count:
            out_files = [mkstemp(".bmp", "cvlab_out_")[1] for _ in range(output_count)]
            out_dir = None
        else:
            out_files = []
            out_dir = mkdtemp(prefix="cvlab_out_")

        try:
            for in_file, image in zip(in_files, images):
                cv.imwrite(in_file, image)

            command = self.command.format(inputs=" ".join(in_files), outputs=" ".join(out_files), output_dir=out_dir, **args)
            command = command.replace("\\", "/")
            print("Executing:", command)
            check_call(command, shell=True)

            results = []

            if out_files:
                results += [cv.imread(out_file) for out_file in out_files]

            if out_dir:
                outputs = os.listdir(out_dir)
                results += [cv.imread(out_dir + "/" + o) for o in outputs]

            if len(results) == 1: return results[0]
            else: return tuple(results)

        finally:
            try:
                for f in in_files + out_files:
                    os.remove(f)
            except Exception as e:
                print("Exception during ProcessElement cleanup:", e)


