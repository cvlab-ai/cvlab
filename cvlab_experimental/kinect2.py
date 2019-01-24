import os
if os.name != 'nt': raise ImportError("Kinect module only work on Windows")

from datetime import datetime, timedelta
from threading import Event

import math
import time

from cvlab.diagram.elements.base import *
from cvlab.diagram.elements.video_io import Camera

class KinectCamera(NormalElement):
    name = "Kinect (OpenNI)"
    comment = "Reads real-time video and depth from physical kinect device"
    device_lock = Camera.device_lock

    def __init__(self):
        super(KinectCamera, self).__init__()
        self.capture = None
        self.actual_parameters = {"width": 0, "height": 0, "fps": 0}
        self.last_frame_time = datetime.now()
        self.play = Event()  # todo: we shall read playing state from some parameter after load
        self.play.set()
        self.recalculate()

    def get_attributes(self):
        return [], \
            [Output("image"), Output("depth")], \
            [IntParameter("width", value=0, min_=0, max_=4096),
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

        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        self.may_interrupt()
        with self.device_lock:
            self.capture = cv.VideoCapture(cv.cv.CV_CAP_OPENNI)
            self.capture.set(cv.cv.CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, cv.cv.CV_CAP_OPENNI_VGA_30HZ)

        if not self.capture or not self.capture.isOpened():
            raise ProcessingError("Can't open the camera device")

        # todo: uncomment this and implement correctly!
        # if self.actual_parameters["width"] != parameters["width"]:
        #     self.capture.set(3, parameters["width"])
        #     self.actual_parameters["width"] = parameters["width"]
        #     self.may_interrupt()
        #
        # if self.actual_parameters["height"] != parameters["height"]:
        #     self.capture.set(4, parameters["height"])
        #     self.actual_parameters["height"] = parameters["height"]
        #     self.may_interrupt()

        if self.actual_parameters["fps"] != parameters["fps"]:
            self.capture.set(5, int(math.ceil(parameters["fps"])))
            self.actual_parameters["fps"] = parameters["fps"]
            self.may_interrupt()

        while True:
            self.may_interrupt()
            now = datetime.now()
            if now - self.last_frame_time < timedelta(seconds=1.0/parameters["fps"]):
                seconds_to_wait = 1.0/parameters["fps"] - (now-self.last_frame_time).total_seconds()
                breaks = int(round(seconds_to_wait*10+1))
                for _ in range(breaks):
                    time.sleep(seconds_to_wait/breaks)
                    self.may_interrupt()
            self.may_interrupt()
            self.set_state(Element.STATE_BUSY)
            _, image = self.capture.retrieve(0, cv.cv.CV_CAP_OPENNI_BGR_IMAGE)
            _, depth = self.capture.retrieve(0, cv.cv.CV_CAP_OPENNI_DEPTH_MAP)
            self.last_frame_time = datetime.now()
            self.may_interrupt()
            if image is not None and len(image) > 0:
                if parameters["width"] and parameters["height"] and (image.shape[0] != parameters["height"] or image.shape[1] != parameters["width"]):
                    image = cv.resize(image, (parameters["width"], parameters["height"]))
                self.outputs["image"].put(Data(image))

                if parameters["width"] and parameters["height"] and (depth.shape[0] != parameters["height"] or depth.shape[1] != parameters["width"]):
                    depth = cv.resize(depth, (parameters["width"], parameters["height"]))
                self.outputs["depth"].put(Data(depth))

                self.set_state(Element.STATE_READY)
                self.notify_state_changed()
            else:
                return 
            self.may_interrupt()
            self.play.wait()


register_elements_auto(__name__, locals(), "Kinect", 10)


