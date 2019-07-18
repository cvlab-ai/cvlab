import time
import math
from threading import Event
from datetime import datetime, timedelta

from .base import *


class Camera(InputElement):
    name = "Camera input"
    comment = "Reads real-time video from physical camera device"
    device_lock = Lock()
    repeat_after_end = False

    def __init__(self):
        super(Camera, self).__init__()
        self.capture = None
        self.actual_parameters = {"device": None, "width": 0, "height": 0, "fps": 0}
        self.last_frame_time = datetime.now()
        self.play = Event()  # todo: we should load this state from some parameter
        self.play.set()
        self.recalculate(True, True, True)

    def get_attributes(self):
        return [], \
            [Output("output")], \
            [IntParameter("device", value=0, min_=0, max_=10),
             IntParameter("width", value=0, min_=0, max_=4096),
             IntParameter("height", value=0, min_=0, max_=4096),
             FloatParameter("fps", value=15, min_=0.1, max_=120),
             ButtonParameter("pause", self.playpause, "Play / Pause")]

    def playpause(self):
        if self.play.is_set():
            self.play.clear()
        else:
            self.play.set()

    def delete(self):
        self.play.set()
        ThreadedElement.delete(self)
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

    def process(self):
        parameters = {}
        for name, parameter in self.parameters.items():
            parameters[name] = parameter.get()

        if self.actual_parameters != parameters:
            self.last_frame_time = datetime(2000, 1, 1)

        if self.actual_parameters["device"] != parameters["device"]:
            with Camera.device_lock:
                if self.capture is not None and self.capture.isOpened():
                    self.capture.release()
                self.capture = cv.VideoCapture(parameters["device"])
            self.may_interrupt()
            self.actual_parameters["device"] = parameters["device"]

        if not self.capture or not self.capture.isOpened():
            raise ProcessingError("Can't open the camera device")

        if self.actual_parameters["width"] != parameters["width"]:
            self.capture.set(3, parameters["width"])
            self.actual_parameters["width"] = parameters["width"]
            self.may_interrupt()

        if self.actual_parameters["height"] != parameters["height"]:
            self.capture.set(4, parameters["height"])
            self.actual_parameters["height"] = parameters["height"]
            self.may_interrupt()

        if self.actual_parameters["fps"] != parameters["fps"]:
            self.capture.set(5, int(math.ceil(parameters["fps"])))
            self.actual_parameters["fps"] = parameters["fps"]
            self.may_interrupt()

        if not self.outputs["output"].get():
            data = Data()
            self.outputs["output"].put(data)
        else:
            data = self.outputs["output"].get()

        while True:
            self.may_interrupt()
            now = datetime.now()
            if now - self.last_frame_time < timedelta(seconds=1.0/parameters["fps"]):
                seconds_to_wait = 1.0/parameters["fps"] - (now-self.last_frame_time).total_seconds()
                breaks = int(round(seconds_to_wait*10+1))
                for _ in range(breaks):
                    time.sleep(seconds_to_wait/breaks)
                    self.may_interrupt()
            self.last_frame_time = datetime.now()
            self.may_interrupt()
            self.set_state(Element.STATE_BUSY)
            retval, image = self.capture.read()
            self.may_interrupt()
            if image is not None and len(image) > 0:
                if parameters["width"] and parameters["height"] and (image.shape[0] != parameters["height"] or image.shape[1] != parameters["width"]):
                    image = cv.resize(image, (parameters["width"], parameters["height"]))
                data.value = image
                self.set_state(Element.STATE_READY)
                self.notify_state_changed()
            elif self.repeat_after_end:
                # if reading from file, then repeat
                # todo: what if this is a camera device? if so, this means that camera was disconnected so we shall restart it
                self.capture.set(1, 0)  # poczatek wideo
            else:
                print("VideoCapture returned null image")
            self.may_interrupt()
            self.play.wait()


class VideoFrameFilter(NormalElement):
    name = "Video frame filter"
    comment = "Filters only selected frames from input stream"

    def __init__(self):
        super(VideoFrameFilter, self).__init__()
        self.dropped = 9999

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [IntParameter("drop", value=1, min_=0, max_=120, name="Drop frames")]

    def get_processing_units(self, inputs, parameters):
        outputs = {"output": inputs["input"].create_placeholder()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def process_units(self):
        unit = self.units[0]
        if not unit.ready_to_execute():
            return
        self.dropped += 1
        if self.dropped > unit.parameters["drop"]:
            data = unit.inputs["input"].copy()
            if not data.is_complete():
                return
            self.dropped = 0
            self.may_interrupt()
            unit.outputs["output"].value = data.value




class VideoRecorder(FunctionGuiElement, ThreadedElement):
    name = "Video recorder"
    comment = "Saves its input as a video file"

    def __init__(self):
        super(VideoRecorder, self).__init__()
        self.path = ""
        self.video = None
        self.video_lock = RLock()
        self.video_size = (0, 0)

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [SavePathParameter("path"), SizeParameter("size", value=(640, 480)),
                ButtonParameter("record", self.record), ButtonParameter("stop", self.stop)]

    def process_inputs(self, inputs, outputs, parameters):
        with self.video_lock:
            if self.video and self.video.isOpened():
                frame = inputs["input"].value
                frame = cv.resize(frame, self.video_size, interpolation=cv.INTER_LINEAR)
                self.video.write(frame)

    def record(self):
        self.path = self.parameters["path"].get()
        self.video_size = self.parameters["size"].get()
        with self.video_lock:
            if self.video:
                self.stop()
            self.video = cv.VideoWriter(self.path, -1, 15, self.video_size, True)

    def stop(self):
        with self.video_lock:
            if self.video:
                self.video.release()
            self.video = None



class VideoLoader(Camera):
    name = "Video loader"
    comment = "Loads a video from disk"
    repeat_after_end = True

    def get_attributes(self):
        params = super(VideoLoader, self).get_attributes()
        assert params[2][0].name == "device"
        params[2][0] = PathParameter("device", "Path", value=CVLAB_DIR+"/images/fractal.avi")
        return params


register_elements_auto(__name__, locals(), "Video IO", 4)

