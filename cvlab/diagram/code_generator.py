from ..core.core_element import CoreElement
from .connectors import Input, Output

data_header = """\

from collections import defaultdict
from copy import copy
from threading import Lock, RLock


class Data:
    NONE = 0
    SEQUENCE = 1
    IMAGE = 2

    def __init__(self, value=None, _type=IMAGE):
        self._type = _type
        if _type == Data.SEQUENCE and value is None:
            self._value = []
        else:
            self._value = value
        self.lock = RLock()

    def clear(self):
        with self.lock:
            if self._type == Data.SEQUENCE:
                for d in self._value:
                    d.clear()
            else:
                self.value = None

    def copy(self):
        with self.lock:
            if self._type == self.NONE:
                return EmptyData()
            if self._type == self.SEQUENCE:
                return Sequence([d.copy() for d in self._value])
            elif self._type == self.IMAGE:
                if self._value is None or (hasattr(self._value, "size") and not len(self._value)):
                    pass
                    # print("Copying null data!")
                return ImageData(self._value)
            else:
                raise TypeError("Wrong Data.type")

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        with self.lock:
            if self._value is new_value:
                return
            self._value = new_value

    def assign(self, other):
        assert isinstance(other, Data)
        assert self.is_compatible(other)  # todo: assert czy raise?
        if self._type == Data.SEQUENCE:
            assert len(self._value) == len(other._value)  # todo: jak wyĹĽej
            for mine, her in zip(self._value, other._value):
                mine.assign(her)
        else:
            self.value = other.value

    def is_compatible(self, other):
        assert isinstance(other, Data)
        if self._type != other._type: return False
        if self._type == Data.SEQUENCE:
            if len(self._value) != len(other._value): return False
            return all(mine.is_compatible(her) for mine, her in zip(self._value, other._value))
        return True

    def type(self):
        with self.lock:
            if self._value is None or (self._type == Data.IMAGE and hasattr(self.value, "size") and not self._value.size):
                return Data.NONE
            else:
                return self._type

    def sequence_get_value(self, sequence_number):
        with self.lock:
            if self.type() != Data.SEQUENCE:
                return self
            else:
                return self._value[min(sequence_number, len(self._value) - 1)]

    __getitem__ = sequence_get_value

    def __iter__(self):
        with self.lock:
            if self.type() == Data.NONE:
                return iter([])
            elif self.type() == Data.SEQUENCE:
                return iter(self._value)
            else:
                return iter([self])

    def desequence_all(self):
        with self.lock:
            if self._type == Data.NONE: return [None]
            if self._type == Data.IMAGE: return [self._value]
            if self._type == Data.SEQUENCE:
                t = []
                for d in self._value:
                    t += d.desequence_all()
                return t
            raise TypeError("Wrong data type - cannot desequence")

    def is_complete(self):
        with self.lock:
            t = self.type()
            if t == Data.NONE:
                return False
            elif t == Data.IMAGE:
                return True
            else:
                return all(d.is_complete() for d in self._value)

    def create_placeholder(self):
        if self._type == Data.SEQUENCE:
            return Data([d.create_placeholder() for d in self._value], Data.SEQUENCE)
        else:
            return Data()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        id_ = "0x{:08X}".format(id(self))
        if self._type == Data.NONE: return '<Data [empty] at {}>'.format(id_)
        if self.type() == Data.IMAGE:
            try:
                shape = "?"
                shape = len(self._value)
                shape = self._value.shape
            except Exception:
                pass
            return '<Data [image {}] at {}>'.format(shape, id_)
        if self.type() == Data.NONE: return "<Data [empty image] at {}>".format(id_)
        if self.type() == Data.SEQUENCE:
            s = ""
            images_count = 0
            none_count = 0
            for d in self._value:
                if d.type() == Data.SEQUENCE:
                    s += str(d) + ", "
                if d.type() == Data.IMAGE:
                    images_count += 1
                if d.type() == Data.NONE:
                    none_count += 1
            if images_count:
                s += str(images_count) + " images, "
            if none_count:
                s += str(none_count) + " nones, "
            if len(s) > 0:
                s = s.rstrip(', ')
            s = '<Sequence ' + '[' + s + '] at {}>'.format(id_)
            return s
        raise TypeError("Wrong data type - cannot desequence")

    def __nonzero__(self):
        return self.type() != Data.NONE

    ready = __nonzero__

    def __eq__(self, other):
        if not isinstance(other, Data): return False
        with self.lock, other.lock:
            if self.type() != other.type(): return False
            if self.type() == Data.SEQUENCE:
                if len(self._value) != len(other._value): return False
                assert all(isinstance(d, Data) for d in self._value)
                assert all(isinstance(d, Data) for d in other._value)
                return all(a == b for a, b in zip(self._value, other._value))
            else:
                return self._value is other._value
                # todo: if we want logical equality rather than reference equality, we shall use this:
                # elif self.type() == Data.IMAGE:
                # #assert isinstance(self._value, np.ndarray)
                # return self._value is other._value and np.array_equal(self._value, other._value)
                # else:
                #     raise ValueError("Wrong data type")


class EmptyOptionalData(Data):
    def ready(self):
        return True


def Sequence(values=None):
    return Data(values, Data.SEQUENCE)


def EmptyData():
    return Data(None, Data.NONE)


def ImageData(value=None):
    return Data(value, Data.IMAGE)

"""


file_template = """\
#!/usr/bin/env python
#
# Code generated with CV Lab
# https://github.com/cvlab-ai/cvlab
#


### imports ###

{imports}

### helpers ###

{helpers}

### functions ###

{functions}

### process function ###

{process_function}

### script ###

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("Usage: generated.py {inputs}")
        exit(1)
    args = []
    for arg in sys.argv[1:]:
        args.append(Data(cv.imread(arg)))
    outputs = process(*args)
    for name, output in outputs.items():
        cv.imwrite("output-{{}}.png".format(name), output.value)
        
"""


class CodeGenerator:
    def __init__(self, element):
        self.base_element = element
        self.functions = {}  # code -> name
        self.elem_names = {}  # element -> name
        self.imports = {"import numpy as np", "import cv2 as cv", "import cv2"}
        self.inputs = set()
        self.output = ""
        self.elem_cnt = 0
        self.code = ""

    def generate(self):
        self.process(self.base_element)

        imports = "\n".join(self.imports)

        helpers = data_header

        functions = "\n\n".join(self.functions)

        inputs = ", ".join(self.inputs)
        code = self.code.replace("\n", "\n    ")
        output = self.output
        process_function = "def process({inputs}):\n    {code}\n    return {output}\n".format(**locals())

        return file_template.format(**locals())

    def process(self, element):
        assert isinstance(element, CoreElement)

        # fixme: this may not work for elements with inputs set as multiple

        self.elem_cnt += 1
        element_name = element.__class__.__name__.lower() + str(self.elem_cnt)
        func_name, func_code, input_names = element.get_source()
        self.inputs.update(["{element_name}_{input_name}".format(element_name=element_name, input_name=input_name) for input_name in input_names])
        if func_code:
            self.functions[func_code] = func_name
        if element is self.base_element:
            self.output = element_name
        inputs = {}  # input name -> value
        for input in element.inputs.values():
            assert isinstance(input, Input)
            for output in input.connected_from:
                assert isinstance(output, Output)
                in_elem = output.parent
                if in_elem not in self.elem_names:
                    self.elem_names[in_elem] = self.process(in_elem)
                value = '{}["{}"]'.format(self.elem_names[in_elem], output.name)
                if input.multiple:
                    multi_number = len(inputs)
                    inputs[input.name + str(multi_number)] = value
                else:
                    inputs[input.name] = value
        if func_code:
            inputs = "{" + ",".join(['"{}":{}'.format(input_name, value) for input_name, value in inputs.items()]) + "}"
            params = "{" + ",".join(['"{}":{}'.format(param_name, repr(value.get())) for param_name, value in element.parameters.items()]) + "}"
            self.code += """\
{element_name} = {{}}
{func_name}({inputs}, {element_name}, {params})
""".format(**locals())
        else:
            inputs = "{" + ",".join(['"{output_name}":{element_name}_{output_name}'.format(output_name=output_name, element_name=element_name) for output_name in input_names]) + "}"
            self.code += "{element_name} = {inputs}\n".format(**locals())

        return element_name


def generate(element):
    return CodeGenerator(element).generate()

