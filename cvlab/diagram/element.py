import threading
from collections import OrderedDict

import cv2 as cv

from . import id_manager
from .connectors import Output, Input
from .parameters import Parameter


class Element:
    name = "Unnamed element"
    comment = ""
    icon = None

    """
    Interface for all logic and GUI diagram objects.
    Methods must be thread-safe and non-blocking.
    Use of parameters must be thread-safe.
    """

    STATE_UNSET = 0
    STATE_BUSY = 1
    STATE_READY = 2
    STATE_ERROR = 3

    def __init__(self):
        super(Element, self).__init__()

        self.state = self.STATE_UNSET
        self.message = ""
        self.lock = threading.RLock()
        self.diagram = None
        self.object_id = id_manager.next_id(self)
        self.unique_id = str(id_manager.unique_id())

        inputs, outputs, parameters = self.get_attributes()

        self.inputs = OrderedDict()
        for i in inputs:
            assert isinstance(i, Input)
            i.parent = self
            self.inputs[i.id] = i

        self.outputs = OrderedDict()
        for i in outputs:
            assert isinstance(i, Output)
            i.parent = self
            self.outputs[i.id] = i

        self.parameters = OrderedDict()
        for i in parameters:
            assert isinstance(i, Parameter)
            #i.parent_element = self
            i.value_changed.connect(self.parameter_changed)
            self.parameters[i.id] = i

    def get_attributes(self):
        """
        Returns three lists: inputs, outputs and parameters
        """
        pass

    def parameter_changed(self):
        self.recalculate(True, False, True)

    #logic methods

    def recalculate(self, refresh_parameters, refresh_structure, force_break):
        """
        Informs that something has changed, and the element must re-do the calculations.
        If force_break=True, then it also forces element to break and cancel current calculations.
        If recal_structure=True, the it also recreates the structure of processing (necessary when Sequences are changed od elements are [dis]connected)
        """
        print("recalculate.")

    def delete(self):
        pass

    def may_interrupt(self):
        return False

    def set_state(self, state, info=""):
        if isinstance(info, cv.error):
            info = str(info)
            if "error: " in info:
                info = "OpenCV: " + info[info.index("error: ") + 7:]
        elif isinstance(info, Exception):
            info = info.__class__.__name__ + ": " + ", ".join(map(str, info.args))
        else:
            info = str(info)

        info = info.strip()

        with self.lock:
            if self.state == state and self.message == info: return
            self.state = state

            if state == self.STATE_READY:
                info = "Ready. " + info
            elif state == self.STATE_BUSY:
                info = "Waiting... " + info
            elif state == self.STATE_ERROR:
                info = "ERROR! " + info

            self.message = info

        self.notify_state_changed()

    def to_json(self):
        return {
            "_type": "element",
            "class": self.__class__.__name__,
            "module": self.__module__,
            "parameters": self.parameters,
            "unique_id": self.unique_id
        }

    def from_json(self, data):
        if "unique_id" in data:
            self.unique_id = data["unique_id"]
        for param, value in data["parameters"].items():
            if param in self.parameters:
                self.parameters[param].from_json(value)
        self.recalculate(True, True, True)


    #gui methods

    def notify_state_changed(self):
        """
        Informs the GUI that the processing is finished.
        """
        print("redraw.")
