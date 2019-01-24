import os
if os.name != 'nt': raise ImportError("Kinect module only work on Windows")

from datetime import datetime, timedelta, time
from threading import Event

from cvlab.diagram.elements.base import *

from .thirdparty.pykinect import nui


class Kinect(NormalElement):
    name = "Kinect (pykinect)"
    comment = "Reads data from Kinect device"
    device_lock = Lock()

    def __init__(self):
        super(Kinect, self).__init__()
        self.last_frame_time = datetime.now()
        self.play = Event()  # todo: ladowanie tego powinno byc z parametru - zapamietywane przy zapisie/odczycie
        self.play.set()
        self.last_parameters = None
        self.kinect = None
        self.color_frame = None
        self.depth_frame = None
        self.color_timestamp = None
        self.depth_timestamp = None
        self.color_image = None
        self.depth_image = None
        self.kinect_color_lock = threading.Lock()
        self.kinect_depth_lock = threading.Lock()
        self.recalculate(True, True, True)

    def get_attributes(self):
        return [], \
               [Output("color"), Output("depth")], \
               [IntParameter("device", value=0, min_=0, max_=10),
                ComboboxParameter("color res", [("640x480", 0), ("1280x960", 1)]),
                ComboboxParameter("depth res", [("640x480", 0), ("320x240", 1)]),
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
        if self.kinect is not None:
            self.kinect.close()

    def process(self):
        print("process")
        if not self.check_parameters_up_to_date():
            if self.kinect is not None:
                self.kinect.close()
                self.kinect = None
            try:
                self.kinect = nui.Runtime(index=self.parameters["device"].get())
            except:
                raise Exception("Device not connected or Microsoft Kinect SDK missing")
            self.kinect.video_frame_ready += self.video_frame_ready
            self.kinect.depth_frame_ready += self.depth_frame_ready

            color_resolution = self.parameters["color res"].get()
            if color_resolution == 0:
                res = nui.ImageResolution.Resolution640x480
            else:
                res = nui.ImageResolution.Resolution1280x1024
            self.kinect.video_stream.open(nui.ImageStreamType.Video, 2, res, nui.ImageType.Color)

            depth_resolution = self.parameters["depth res"].get()
            if depth_resolution == 0:
                res = nui.ImageResolution.Resolution640x480
            else:
                res = nui.ImageResolution.Resolution320x240
            self.kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, res, nui.ImageType.Depth)

        if self.kinect is not None:
            while True:
                self.produce_output()

    def check_parameters_up_to_date(self):
        parameters = self.get_the_parameters()
        result = self.last_parameters == parameters
        self.last_parameters = parameters
        return result

    def get_the_parameters(self):
        parameters = {}
        for name in ["device", "color res", "depth res"]:
            parameters[name] = self.parameters[name].get()
        return parameters

    def video_frame_ready(self, frame):
        with self.kinect_color_lock:
            self.color_frame = frame
            self.color_image = self.get_color_image()
            self.color_timestamp = frame.timestamp

    def depth_frame_ready(self, frame):
        with self.kinect_depth_lock:
            self.depth_frame = frame
            self.depth_image = self.get_depth_image()
            self.depth_timestamp = frame.timestamp

    def produce_output(self):
        print("produce")
        self.may_interrupt()
        self.sleep_if_needed()
        self.may_interrupt()

        update = False
        color = None
        depth = None
        with self.kinect_color_lock:
            with self.kinect_depth_lock:
                if self.color_image is not None and self.depth_image is not None and self.timestamps_ok():
                    # copy with eliminating mirroring
                    color = self.color_image[:, ::-1, :]
                    depth = self.depth_image[:, ::-1]
                    # get rid of player indexes
                    depth = np.right_shift(depth, 3)
                    update = True

        self.may_interrupt()
        if update:
            self.may_interrupt()
            self.set_state(Element.STATE_BUSY)
            if color is not None and len(color) > 0 and depth is not None and len(depth) > 0:
                self.outputs["color"].put(Data(color))
                self.outputs["depth"].put(Data(depth))
                self.last_frame_time = datetime.now()
                self.set_state(Element.STATE_READY)
                self.notify_state_changed()
        self.may_interrupt()
        self.play.wait()

    def timestamps_ok(self):
        return abs(self.color_timestamp - self.depth_timestamp) <= 10

    def sleep_if_needed(self):
        now = datetime.now()
        if now - self.last_frame_time < timedelta(seconds=1.0 / self.parameters["fps"].value):
            seconds_to_wait = 1.0 / self.parameters["fps"].value - (now - self.last_frame_time).total_seconds()
            breaks = int(round(seconds_to_wait * 10 + 1))
            for _ in range(breaks):
                time.sleep(seconds_to_wait / breaks)
                self.may_interrupt()

    def get_color_image(self):
        if self.color_frame is not None:
            image = self.prepare_color_image()
            self.color_frame.image.copy_bits(image.ctypes.data)
            return image[:, :, :3]

    def get_depth_image(self):
        if self.depth_frame is not None:
            depth = self.prepare_depth_image()
            self.depth_frame.image.copy_bits(depth.ctypes.data)
            return depth

    def prepare_color_image(self):
        if self.color_frame.resolution == nui.ImageResolution.Resolution1280x1024:
            return np.zeros((1024, 960, 4), np.uint8)
        else:
            return np.zeros((480, 640, 4), np.uint8)

    def prepare_depth_image(self):
        if self.depth_frame.resolution == nui.ImageResolution.Resolution640x480:
            return np.zeros((480, 640, 1), np.uint16)
        else:
            return np.zeros((240, 320, 1), np.uint16)


register_elements_auto(__name__, locals(), "Kinect", 10)

