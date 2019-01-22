from copy import *

from .base import *


class PlusOperator(MultiInputOneOutputElement):
    name = "Plus operator"
    comment = "Adds all input images"

    def process_inputs(self, inputs, outputs, parameters):
        output = Data()
        for input in inputs.values():
            if output.type() == output.NONE:
                output.value = copy(input.value)
            else:
                cv.add(output.value, input.value, output.value)
            self.may_interrupt()
        outputs["output"] = output


class MinusOperator(NormalElement):
    name = "Minus operator"
    comment = "Subtracts second from first image"

    def get_attributes(self):
        return [Input("from"), Input("what")], [Output("output")], []

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = Data(cv.subtract(inputs["from"].value, inputs["what"].value))


class AbsDiffOperator(NormalElement):
    name = "Difference operator"
    comment = "Calculates absolute difference between images"

    def get_attributes(self):
        return [Input("1", "Input 1"), Input("2", "Input 2")], [Output("output")], []

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = Data(cv.absdiff(inputs["1"].value, inputs["2"].value))


class AverageOperator(MultiInputOneOutputElement):
    name = "Average operator"
    comment = "Averages all input images"

    def process_inputs(self, inputs, outputs, parameters):
        temp = None
        for input_ in inputs.values():
            if temp is None:
                temp = np.float32(input_.value)
            else:
                temp += np.float32(input_.value)
        outputs["output"] = Data(np.uint8(temp // len(inputs)))


class MaxOperator(MultiInputOneOutputElement):
    name = "Maximum operator"
    comment = "Output image contains maximum pixel values"

    def process_inputs(self, inputs, outputs, parameters):
        temp = None
        for input_ in inputs.values():
            if temp is None:
                temp = copy(input_.value)
            else:
                temp = np.maximum(temp, input_.value)
        outputs["output"] = Data(temp)


class MinOperator(MultiInputOneOutputElement):
    name = "Minimum operator"
    comment = "Output image contains minimum pixel values"

    def process_inputs(self, inputs, outputs, parameters):
        temp = None
        for input_ in inputs.values():
            if temp is None:
                temp = copy(input_.value)
            else:
                temp = np.minimum(temp, input_.value)
        outputs["output"] = Data(temp)


class InvertOperator(NormalElement):
    name = "Invertion operator"
    comment = "Inverts pixel values"

    def get_attributes(self):
        return [Input("input")], [Output("output")], []

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = Data(-inputs["input"].value - 1)


class ScalarMultiplyOperator(NormalElement):
    name = "Scalar multiply"
    comment = "Multiplies matrice by scalar"

    def get_attributes(self):
        return [Input("input")], [Output("output")], [FloatParameter("factor", min_=0, max_=10)]

    def process_inputs(self, inputs, outputs, parameters):
        input_ = inputs["input"].value
        factor = parameters["factor"]
        if len(input_.shape) == 2:
            output = cv.multiply(input_, factor)
        else:
            output = cv.multiply(input_, (factor, factor, factor, factor))
        outputs["output"] = Data(output)


class ScalarAddOperator(NormalElement):
    name = "Scalar add"
    comment = "Adds constant value to each matrix element"

    def get_attributes(self):
        return [Input("input")], [Output("output")], [FloatParameter("value", min_=-255, max_=255)]

    def process_inputs(self, inputs, outputs, parameters):
        input_ = inputs["input"].value
        value = parameters["value"]
        if len(input_.shape) == 2:
            output = cv.add(input_, value)
        else:
            output = cv.add(input_, (value, value, value, value))
        outputs["output"] = Data(output)


register_elements_auto(__name__, locals(), "Basic operators", 4)
