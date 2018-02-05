from __future__ import unicode_literals

import threading
from collections import OrderedDict

import numpy as np
from PyQt4.QtCore import pyqtSignal, QObject

from . import id_manager


class Parameter(QObject):
    value_changed = pyqtSignal()

    def __init__(self, id, name=None, value=None, *args, **kwargs):
        super(Parameter, self).__init__()
        self.id = id
        self.object_id = id_manager.next_id(self)
        if name is None: name = id
        self.name = name
        self.value = value
        self.lock = threading.RLock()
        self.children = set()
        self.value_changed.emit()

    def connect_child(self, child):
        assert isinstance(child, Parameter)
        if child not in self.children:
            self.children.add(child)
            child.set(self.value)
            child.connect_child(self)

    def disconnect_child(self, child):
        if child in self.children:
            self.children.remove(child)
            child.disconnect_child(self)

    def disconnect_all_children(self):
        for child in frozenset(self.children):
            self.disconnect_child(child)

    def set(self, value):
        with self.lock:
            if self.value == value: return
            self.value = value
        self.value_changed.emit()
        for child in self.children:
            child.set(value)

    def get(self):
        with self.lock:
            return self.value

    def to_json(self):
        return self.value

    def from_json(self, data):
        self.set(data)


class PathParameter(Parameter):
    def __init__(self, id, name=None, value="", save_mode=False, *args, **kwargs):
        super(PathParameter, self).__init__(id, name, value, *args, **kwargs)
        self.save_mode = save_mode


class SavePathParameter(PathParameter):
    def __init__(self, *args, **kwargs):
        super(SavePathParameter, self).__init__(save_mode=True, *args, **kwargs)


class MultiPathParameter(Parameter):
    pass


class DirectoryParameter(Parameter):
    pass


class TextParameter(Parameter):
    def __init__(self, id, name=None, value="", window_title="Text parameter editor", window_content="", live=True):
        super(TextParameter, self).__init__(id, name, value)
        self.window_title = window_title
        self.window_content = window_content
        self.live = live


class NumberParameter(Parameter):
    def __init__(self, id, name=None, value=None, min_=-100000, max_=100000, step=1):
        if value is None:
            if min_ >= 0:
                value = min_
            else:
                value = min(max_, 0)
        super(NumberParameter, self).__init__(id, name, value)
        self.min = min_
        self.max = max_
        self.step = step


class IntParameter(NumberParameter):
    def set(self, value):
        NumberParameter.set(self, int(value))


class FloatParameter(NumberParameter):
    def __init__(self, id, name=None, value=None, min_=-1000, max_=1000, step=0.1):
        super(FloatParameter, self).__init__(id, name, value=value, min_=min_, max_=max_, step=step)

    def set(self, value):
        NumberParameter.set(self, float(value))


class SizeParameter(Parameter):
    def __init__(self, id, name=None, value=(1, 1)):
        super(SizeParameter, self).__init__(id, name, value)
        self.min_val = 0
        big_int = pow(10, 9)
        self.max_val = big_int

    def set(self, value):
        Parameter.set(self, tuple(value))


class PointParameter(Parameter):
    def __init__(self, id, name=None, value=(0, 0)):
        super(PointParameter, self).__init__(id, name, value)
        self.min_val = -1
        big_int = pow(10, 9)
        self.max_val = big_int

    def set(self, value):
        Parameter.set(self, tuple(value))


class ComboboxParameter(Parameter):
    def __init__(self, id, values, name=None, default_value_idx=None):
        super(ComboboxParameter, self).__init__(id, name)
        self.values = OrderedDict(values)
        if default_value_idx is not None:
            self.set(list(self.values.values())[default_value_idx])
        else:
            self.value = list(self.values.values())[0]

    def to_json(self):
        return list(self.values.values()).index(self.value)

    def from_json(self, data):
        self.set(list(self.values.values())[data])


class ButtonParameter(Parameter):
    def __init__(self, id, callback, name=None):
        super(ButtonParameter, self).__init__(id, name)
        self.callback = callback

    def to_json(self):
        return ""

    def from_json(self, data):
        pass

    def clicked(self):
        self.callback()


class MatrixParameter(Parameter):
    def to_json(self):
        return self.value.tolist()

    def from_json(self, data):
        self.set(np.array(data))
