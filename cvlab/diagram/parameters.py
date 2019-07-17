import os
import threading
from collections import OrderedDict

import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject

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
    base_path = None

    def __init__(self, id, name=None, value="", save_mode=False, *args, **kwargs):
        super(PathParameter, self).__init__(id, name, value, *args, **kwargs)
        self.save_mode = save_mode

    def to_json(self):
        return self.to_json_relative(self.value)

    def from_json(self, data):
        self.set(self.from_json_relative(data))

    @classmethod
    def to_json_relative(cls, path):
        abs_path = os.path.abspath(path).replace("\\","/")
        base_path = os.path.abspath(cls.base_path).replace("\\","/")
        up_base_path = os.path.dirname(base_path)
        if abs_path.startswith(base_path):  # relative path
            relative_path = abs_path[len(base_path):]
            if relative_path.startswith("/"): relative_path = relative_path[1:]
            return relative_path
        elif abs_path.startswith(up_base_path):  # relative to parent directory
            relative_path = abs_path[len(up_base_path):]
            if relative_path.startswith("/"): relative_path = relative_path[1:]
            relative_path = "../" + relative_path
            return relative_path
        else:  # absolute path
            return abs_path

    @classmethod
    def from_json_relative(cls, path):
        path = path.replace("\\","/")
        if os.path.isabs(path):  # absolute path
            return path
        else:  # relative path
            p = os.path.abspath(cls.base_path + "/" + path).replace("\\","/")
            return p


class SavePathParameter(PathParameter):
    def __init__(self, *args, **kwargs):
        super(SavePathParameter, self).__init__(save_mode=True, *args, **kwargs)


class MultiPathParameter(Parameter):
    def to_json(self):
        return [PathParameter.to_json_relative(path) for path in self.value]

    def from_json(self, data):
        self.set([PathParameter.from_json_relative(path) for path in data])


class DirectoryParameter(PathParameter):
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
        self.min = 0
        self.max = 10**9

    def set(self, value):
        Parameter.set(self, tuple(value))


class PointParameter(Parameter):
    def __init__(self, id, name=None, value=(0, 0)):
        super(PointParameter, self).__init__(id, name, value)
        self.min = -1
        self.max = 10**9

    def set(self, value):
        Parameter.set(self, tuple(value))


class ScalarParameter(Parameter):
    def __init__(self, id, name=None, value=(0, 0, 0, 0), min_=-1000, max_=1000):
        super(ScalarParameter, self).__init__(id, name, value)
        self.min = min_
        self.max = max_

    def set(self, value):
        value = list(value) + [0,0,0,0]
        value = tuple(value[:4])
        Parameter.set(self, value)


class ComboboxParameter(Parameter):
    def __init__(self, id, values, name=None, default_value_idx=None):
        super(ComboboxParameter, self).__init__(id, name)
        self.values = OrderedDict(values)
        if default_value_idx is not None:
            self.set(list(self.values.values())[default_value_idx])
        else:
            self.value = list(self.values.values())[0]

    def to_json(self):
        # return list(self.values.values()).index(self.value)  # cv-lab v1.0
        index = list(self.values.values()).index(self.value)
        name = list(self.values)[index]
        return {"name": name, "value": self.value}

    def from_json(self, data):
        if isinstance(data, int):  # cv-lab v1.0
            if 0 <= data < len(self.values):
                value = list(self.values.values())[data]
            else:
                print("Warning: Cannot decode parameter '{}' value '{}'".format(self.name, data))
                value = data
        else: # cv-lab v1.1+
            value = data["value"]

        self.set(value)


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
