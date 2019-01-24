from collections import defaultdict
from threading import Lock, RLock

from .errors import ProcessingError


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
        self.observers = defaultdict(int)
        self.observers_lock = Lock()
        self.lock = RLock()

    def add_observer(self, observer, recursive):
        with self.observers_lock:
            self.observers[observer] += 1
        if recursive and self._type == Data.SEQUENCE:
            with self.lock:
                for v in self._value:
                    v.add_observer(observer, True)

    def remove_observer(self, observer, recursive):
        with self.observers_lock:
            self.observers[observer] -= 1
            if self.observers[observer] <= 0:
                del self.observers[observer]
        if recursive and self._type == Data.SEQUENCE:
            with self.lock:
                for v in self._value:
                    v.remove_observer(observer, True)

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
        with self.observers_lock:
            for o in self.observers:
                o(self)

    def assign(self, other):
        assert isinstance(other, Data)
        if not self.is_compatible(other):
            raise ProcessingError("Data.assign: Data not compatible")
        if self._type == Data.SEQUENCE:
            if len(self._value) != len(other._value):
                raise ProcessingError("Data.assign: Sequence not compatible")
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
        """Returns a one-dimensional array with all sequence values"""
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

    def __bool__(self):
        return self.type() != Data.NONE

    ready = __bool__

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



class DataSet:
    def __init__(self, inputs=None, parameters=None, outputs=None):
        self.inputs = inputs if inputs is not None else {}
        self.parameters = parameters if parameters is not None else {}
        self.outputs = outputs if outputs is not None else {}

    def reset_outputs(self):
        for d in self.outputs.values():
            assert isinstance(d, Data)
            d.clear()

    def __eq__(self, other):
        return self.parameters == other.parameters and self.inputs == other.inputs

    def __str__(self):
        return "DataSet(" + repr(self.inputs) + ", " + repr(self.parameters) + ", " + repr(self.outputs) + ")"

    __repr__ = __str__


class ProcessingUnit(DataSet):
    def __init__(self, element, inputs=None, parameters=None, outputs=None):
        super(ProcessingUnit, self).__init__(inputs, parameters, outputs)
        self.element = element
        self.calculated = False

    def is_being_processed(self):
        return self.element.actual_processing_unit is self

    def ready_to_execute(self):
        return all(d.ready() for d in self.inputs.values())

    def data_changed(self, data):
        if self.calculated:
            self.calculated = False
            self.reset_outputs()
        self.element.recalculate(False, False, self.is_being_processed())

    def connect_observables(self):
        for input in self.inputs.values():
            input.add_observer(self.data_changed, True)

    def disconnect_observables(self):
        for input in self.inputs.values():
            input.remove_observer(self.data_changed, True)

    def __str__(self):
        return "ProcessingUnit(" + repr(self.inputs) + ", " + repr(self.parameters) + ", " + repr(
            self.outputs) + ", calculated=" + repr(self.calculated) + ", ready_to_execute=" + repr(
            self.ready_to_execute()) + ")"

    __repr__ = __str__

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

