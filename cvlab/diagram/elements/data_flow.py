from .base import *


class Forwarder(NormalElement):
    name = "Forwarder"
    comment = "Forwards its input without processing"

    def __init__(self):
        super(Forwarder, self).__init__()

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               []

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": inputs["input"]}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_units(self):
        pass

    def get_source(self):
        name = self.__class__.__name__.lower()
        src = """\
def {name}(inputs, outputs, parameters):
    outputs['output'] = inputs['input']
""".format(**locals())
        return name, src, []



class Sequencer(NormalElement):
    name = "Sequencer"
    comment = "Makes a single sequenced data from its inputs"

    def __init__(self):
        super(Sequencer, self).__init__()

    def get_attributes(self):
        return [Input("inputs", multiple=True)], \
               [Output("output", "Sequence")], \
            []

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": Sequence(inputs["inputs"].value)}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_units(self):
        pass


class Desequencer(NormalElement):
    name = "Desequencer"
    comment = "Gets single sequenced data and separates it"

    def get_attributes(self):
        return [Input("input")], [Output("outputs", desequencing=True)], []

    def get_processing_units(self, inputs, parameters):
        outputs = {"outputs": inputs["input"]}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_units(self):
        pass





class Zip(NormalElement):
    name = "Zip"
    comment = "Zips its inputs"

    def get_attributes(self):
        return [Input("inputs", multiple=True)], [Output("outputs")], []

    def get_processing_units(self, inputs, parameters):
        input_list = inputs['inputs'].value

        if len(input_list) == 1 and input_list[0].type() == Data.SEQUENCE:
            input_list = input_list[0].value

        output_list = list(zip(*input_list))
        output_list = [Sequence(list(values)) for values in output_list]

        outputs = {"outputs": Sequence(output_list)}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]

        return units, outputs

    def process_units(self):
        pass



class SequenceSelector(NormalElement):
    name = "Sequence selector"
    comment = "Selects a simple image from a sequence"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("number", value=0, min_=0, max_=100)]

    def get_processing_units(self, inputs, parameters):
        sequence = inputs["input"]
        if sequence.type() != Data.SEQUENCE or not sequence.value:
            raise TypeError("Input data must be a non-empty sequence")
        data = sequence.value[0]
        if not all(d.is_compatible(data) for d in sequence.value):
            raise TypeError("All data values in sequence must be compatible")
        outputs = {"output": data.create_placeholder()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_inputs(self, inputs, outputs, parameters):
        outputs["output"] = inputs["input"].value[parameters["number"]]


class SequenceDeleter(NormalElement):
    name = "Sequence deleter"
    comment = "Deletes a simple image from a sequence"

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("number", value=0, min_=0, max_=100)]

    def get_processing_units(self, inputs, parameters):
        sequence = inputs["input"]
        if sequence.type() != Data.SEQUENCE or not sequence.value or len(sequence.value) <= 1:
            raise TypeError("Input data must be a non-empty sequence")
        if not all(d.is_compatible(sequence.value[0]) for d in sequence.value):
            raise TypeError("All data values in sequence must be compatible")
        data = Sequence(sequence.value[:-1])
        outputs = {"output": data.create_placeholder()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_inputs(self, inputs, outputs, parameters):
        deleted = parameters["number"]
        new_seq = inputs["input"].value[:]
        del new_seq[deleted]
        outputs["output"] = Sequence(new_seq)



class ConcatenateOperator(MultiInputOneOutputElement):
    name = "Concatenate operator"
    comment = "Concatenates input arrays"
    package = "Matrix miscellaneous"

    def process_inputs(self, inputs, outputs, parameters):
        output = Data()
        output.value = np.concatenate([d.value for d in inputs.values()])
        outputs["output"] = output


register_elements_auto(__name__, locals(), "Data flow", 1)