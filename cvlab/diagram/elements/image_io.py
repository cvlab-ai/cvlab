from .base import *


class ImageLoader(InputElement):
    name = "Image loader"
    comment = "Loads the image from disk"

    def get_attributes(self):
        return [], [Output("output")], [PathParameter("path", value=CVLAB_DIR+"/images/lena.jpg")]

    def process_inputs(self, inputs, outputs, parameters):
        d = cv.imread(parameters["path"])
        if d is not None:
            self.may_interrupt()
            outputs["output"] = Data(d)


class ImageSequenceLoader(InputElement):
    name = "Image Sequence loader"
    comment = "Loads a sequence of images from disk"

    def get_attributes(self):
        return [], [Output("output", "Sequence")], [MultiPathParameter("paths", value=[CVLAB_DIR+"/images/lena.jpg"])]

    def process(self):
        paths = self.parameters["paths"].get()
        images = len(paths)
        sequence = Sequence([Data() for _ in range(images)])
        self.outputs["output"].put(sequence)
        for path, data in zip(paths, sequence):
            self.may_interrupt()
            image = cv.imread(path)
            if image is not None:
                data.value = image


class ImageLoader3D(InputElement):
    name = "Image loader 3D"
    comment = "Loads multiple images as 3D image"

    def get_attributes(self):
        return [], [Output("output")], [MultiPathParameter("paths", value=[CVLAB_DIR+"/images/lena.jpg"]*10)]

    def process_inputs(self, inputs, outputs, parameters):
        paths = parameters["paths"]
        image = []

        for path in sorted(paths):
            slice = cv.imread(path)
            self.may_interrupt()
            image.append(slice)
            if slice.shape != image[0].shape:
                raise Exception("Inconsisten slice dimensions")

        image = np.array(image)
        outputs["output"] = Data(image)


class RecurrentSequenceLoader(InputElement):
    name = "Image Recurrent Sequence loader"
    comment = "Loads a recurrent sequence of images from disk (reflecting directory structure)"

    max_level = 5

    def __init__(self):
        super(RecurrentSequenceLoader, self).__init__()
        self.level = 0

    def get_attributes(self):
        return [], [Output("output", "Sequences")], [DirectoryParameter("directory", value="images")]

    def read_directory(self, directory):
        self.level += 1
        sequence = Sequence()
        for entry in os.listdir(directory):
            try:
                if not entry or entry[0] == '.': continue
                path = directory + "/" + entry
                if os.path.isdir(path):
                    if self.level >= self.max_level: continue
                    sequence.value.append(self.read_directory(path))
                else:
                    image = cv.imread(path)
                    if image is not None and image.shape:
                        data = ImageData(image)
                        sequence.value.append(data)
            except Exception:
                pass
        self.level -= 1
        return sequence

    def process(self):
        self.level = 0
        directory = self.parameters["directory"].get()
        data = self.read_directory(directory)
        self.outputs["output"].put(data)


class ImageSaver(NormalElement):
    name = "Image saver"
    comment = "Saves actual image (optionally makes a sequence from them)"

    def get_attributes(self):
        return [Input("input")], [], [SavePathParameter("path", value="")]

    def process_inputs(self, inputs, outputs, parameters):
        cv.imwrite(parameters["path"], inputs["input"].value)


class ArrayLoader(InputElement):
    name = "Array loader"
    comment = "Loads numpy array from disk"

    def get_attributes(self):
        return [], [Output("output")], [PathParameter("path", value=CVLAB_DIR+"/images/default.npy")]

    def process_inputs(self, inputs, outputs, parameters):
        d = np.load(parameters["path"])
        if d is not None:
            self.may_interrupt()
            outputs["output"] = Data(d)


class ArraySaver(NormalElement):
    name = "Array saver"
    comment = "Saves numpy array to disk"

    def get_attributes(self):
        return [Input("input")], [], [SavePathParameter("path", value="")]

    def process_inputs(self, inputs, outputs, parameters):
        np.save(parameters["path"], inputs["input"].value)


register_elements_auto(__name__, locals(), "Image IO", 2)
