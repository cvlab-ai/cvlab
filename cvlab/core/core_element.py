import inspect
import re

from .hooks import *
from .exceptions import *


TEST_QT = False


class CoreElement(Element):
    """Base class for all diagram elements"""

    def __init__(self):
        super(CoreElement, self).__init__()
        for o in self.outputs.values():
            o.hook = OutputHook(o) if not TEST_QT else OutputQtHook(o)
        for i in self.inputs.values():
            i.hook = InputHook(i) if not TEST_QT else InputQtHook(i)
        self.state = self.STATE_READY
        self.data = None
        self.units = []
        self.parameters_changed = True
        self.structure_changed = True
        self.actual_processing_unit = None
        self.delayed_recalculate = False
        self.is_recalculating = False
        self.prepare_empty_data()

    def recalculate(self, refresh_parameters, refresh_structure, force_break, force_units_recalc=False):
        while True:
            try:
                self.parameters_changed |= refresh_parameters
                self.structure_changed |= refresh_structure
                if self.is_recalculating:
                    return
                else:
                    self.delayed_recalculate = False
                self.is_recalculating = True
                self.set_state(self.STATE_BUSY)
                for unit in self.units:
                    unit.calculated = False
                self.process()
                self.set_state(self.STATE_READY)
            except (InterruptException, ProcessingBreak):
                pass
            except Exception as e:
                self.set_state(self.STATE_ERROR, e)  # todo: core2: uncomment this!!!
            self.actual_processing_unit = None
            if not self.delayed_recalculate:
                break
        self.is_recalculating = False

    def may_interrupt(self):
        """Informs the program that element may be interrupted here"""
        return self.delayed_recalculate

    def process_channels(self, inputs, outputs, parameters):
        """Does processing of input channels and returns the outputs"""
        print("process channels")

    def process_inputs(self, inputs, outputs, parameters):
        """Does processing of inputs and returns the outputs"""

        ins = defaultdict(dict)  # [channel number][name] -> input channel image
        for iname, idata in inputs.items():
            for chnum, ich in enumerate(cv.split(idata.value)):
                ins[chnum][iname] = Data(ich)
        outs = defaultdict(dict)  # [output name][channel number] -> output channel image
        for chnum, ich in ins.items():
            o = {}
            self.may_interrupt()
            self.process_channels(ich, o, parameters)
            self.may_interrupt()
            for outname, outch in o.items():
                outs[outname][chnum] = outch
        for outname, channels in outs.items():
            if not all(channels.values()):
                outputs[outname] = Data()
            else:
                outputs[outname] = Data(cv.merge([i.value for i in channels.values()]))

    def get_processing_units(self, inputs, parameters):
        """Returns processing units and output placeholders for the element"""
        return self.get_default_processing_units(inputs, parameters, self.outputs.keys())

    def get_default_processing_units(self, inputs, parameters, output_ids):
        sequences = 0

        for input_ in inputs.values():
            if input_.type() == Data.SEQUENCE:
                sequences = max(sequences, len(input_.value))
            else:
                sequences = max(sequences, 0)

        if sequences == 0:
            outputs = {name: Data() for name in output_ids}
            return [ProcessingUnit(self, inputs, parameters, outputs)], outputs

        units = []
        outputs = {name: Sequence() for name in output_ids}

        for seq_number in range(0, sequences):
            seq_inputs = {input_name: input_.sequence_get_value(seq_number) for input_name, input_ in inputs.items()}
            seq_units, seq_outputs = self.get_default_processing_units(seq_inputs, parameters, output_ids)
            units += seq_units
            for output_name, output_data in seq_outputs.items():
                outputs[output_name].value.append(output_data)

        return units, outputs

    def clear_outputs(self):
        for o in self.data.outputs.values():
            o.clear()

    def prepare_parameters(self):
        self.parameters_changed = False
        for name, parameter in self.parameters.items():
            self.data.parameters[name] = parameter.get()
        self.clear_outputs()
        for unit in self.units:
            unit.calculated = False

    def prepare_empty_data(self):
        self.data = DataSet()
        for unit in self.units:
            unit.disconnect_observables()
        self.units = []

    def prepare_structure(self):
        self.structure_changed = False
        self.prepare_empty_data()
        self.prepare_parameters()
        for name, input_ in self.inputs.items():
            self.data.inputs[name] = input_.get()
        self.units, self.data.outputs = self.get_processing_units(self.data.inputs, self.data.parameters)
        for name, data in self.data.outputs.items():
            self.outputs[name].put(data)
        for unit in self.units:
            unit.connect_observables()

    def prepare_data(self):
        if self.structure_changed:
            self.prepare_structure()
        elif self.parameters_changed:
            self.prepare_parameters()

    def process_units(self):
        for unit in self.units:
            assert isinstance(unit, ProcessingUnit)
            if unit.calculated:
                continue
            if not unit.ready_to_execute():
                continue
            self.actual_processing_unit = unit
            self.may_interrupt()
            inputs = {n: d.copy() for n, d in unit.inputs.items()}
            self.may_interrupt()
            outputs = {}
            self.process_inputs(inputs, outputs, unit.parameters)
            self.may_interrupt()

            # TODO: This is a workaround. Elements should never remove objects from 'outputs'
            # We should really modify Elements to no stop doing that
            # Also, this is probably wrong if there is more than one ProcessingUnit!
            for output_name in outputs:
                unit.outputs[output_name].assign(outputs[output_name])

            unit.calculated = True
            self.actual_processing_unit = None
            self.may_interrupt()

    def process(self):
        self.prepare_data()
        self.process_units()
        self.may_interrupt()

    def delete(self):
        self.prepare_empty_data()  # disconnects data connections
        for o in self.outputs.values():
            o.disconnect_all()
        for i in self.inputs.values():
            i.disconnect_all()
            i.hook.delete()

    def get_source(self):
        """Returns FUNCTION_NAME and SOURCE of the element in format (without leading whitespace!)
            def FUNCTION_NAME(inputs, outputs, parameters):
                # do processing: inputs, parameters -> outputs
        """
        name = self.__class__.__name__.lower()
        if self.process_inputs != CoreElement.process_inputs:
            src = inspect.getsource(self.process_inputs).replace("def process_inputs(self,", "def " + name + "(")
        else:
            src = inspect.getsource(self.process_channels).replace("def process_channels(self,", "def " + name + "(")
        src = src.replace("self.may_interrupt()", "")
        margin = re.search(r"^(\s+)def ", src)
        if margin: src = re.sub("^" + margin.group(1), "", src, flags=re.MULTILINE)
        src = src.replace("\t", "    ")
        return name, src, []

