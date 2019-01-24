from collections import deque
from datetime import datetime

from .base import *


class DelayLine(NormalElement):
    name = "Delay line"
    comment = "Delays its input (useful for video processing)"
    num_outputs = 5

    def __init__(self):
        super(DelayLine, self).__init__()
        self.memory = deque(maxlen=5)
        for i in range(self.num_outputs):
            self.memory.append(EmptyData())

    def get_attributes(self):
        return [Input("input")], \
               [Output("o" + str(i + 1)) for i in range(self.num_outputs)], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        for i in range(self.num_outputs):
            outputs["o" + str(i + 1)] = self.memory[self.num_outputs - i - 1]
        self.memory.append(inputs['input'].copy())



class Snapshot(NormalElement):
    name = "Snapshot"
    comment = "Saves actual input (optionally makes a sequence from them)"

    def __init__(self):
        super(Snapshot, self).__init__()
        self.do_snap = False

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [ButtonParameter("snap", self.snap)]

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": inputs["input"].create_placeholder()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_units(self):
        if not self.do_snap:
            return
        self.do_snap = False
        unit = self.units[0]
        unit.outputs["output"].assign(unit.inputs["input"])

    def snap(self):
        self.do_snap = True
        self.recalculate(False, False, True)



class Accumulator(NormalElement):
    name = "Accumulator"
    comment = "Accumulates its input at given speed"

    def __init__(self):
        super(Accumulator, self).__init__()
        self.memory = None

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [ComboboxParameter("function",{"Average":"avg","Minimum":"min","Maximum":"max"}),
                FloatParameter("speed", value=0.1, min_=0, max_=1, step=0.01),
                ButtonParameter("reset", self.reset_memory)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs["input"].value
        speed = parameters["speed"]
        func = parameters["function"]
        memory = self.memory

        if memory is None or memory.shape != image.shape or memory.dtype != image.dtype:
            memory = image
        else:
            if func == "avg":
                memory = cv.addWeighted(memory, 1-speed, image, speed, 0)
            elif func == "min":
                memory = np.minimum(memory, image)
            elif func == "max":
                memory = np.maximum(memory, image)

        self.may_interrupt()
        self.memory = memory
        outputs["output"] = Data(memory)

    def reset_memory(self):
        self.memory = None




class FpsCounter(NormalElement):
    name = "FPS counter"
    comment = "Counts input frames per second"

    DEQUE_SIZE = 5

    def __init__(self):
        super(FpsCounter, self).__init__()
        self.last_time = datetime.now()
        self.memory = deque(maxlen=5)
        self.bg = False

    def get_attributes(self):
        return [Input("input")], [Output("output")], [IntParameter("memory", value=5, min_=1, max_=50)]

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": Data()}
        return [ProcessingUnit(self, inputs, parameters, outputs)], outputs

    def process(self):
        self.prepare_data()
        if not self.inputs["input"].get(): return
        self.bg = not self.bg
        act_time = datetime.now()
        fps = 1. / (act_time - self.last_time).total_seconds()
        img = np.zeros((30, 50), np.uint8)
        memsize = self.parameters["memory"].get()
        if memsize > 1:
            if self.memory.maxlen != memsize:
                self.memory = deque(self.memory, memsize)
            self.memory.append(fps)
            fps = sum(self.memory) / len(self.memory)
        txt = str(int(round(fps)))
        if self.bg:
            txt += "."
        cv.putText(img, txt, (10, 20), 1, 1, 255, 2)
        if not self.outputs["output"].get():
            self.outputs["output"].put(Data(img))
        else:
            self.outputs["output"].get().value = img
        self.last_time = act_time

register_elements_auto(__name__, locals(), "Video", 4)
