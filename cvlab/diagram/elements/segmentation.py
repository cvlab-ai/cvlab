from .base import *


class GrabCut(NormalElement):
    name = "GrabCut"
    comment = "Foreground Extraction using GrabCut Algorithm"

    def get_attributes(self):
        return [Input("image"), Input("classes")], \
               [Output("classes")], \
               [ComboboxParameter('convert_input', [('None', ''), ('4-value', '4'), ('255-value', '255')], "Convert input", 1),
                ComboboxParameter('convert_output', [('None', ''), ('2-value', '2'), ('4-value', '4'), ('255 2-value', '255-2'), ('255 4-value', '255-4')], "Convert output", 1),
                IntParameter('iterations', "Iterations", 1, 1, 100)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        classes = inputs['classes'].value+0

        if parameters['convert_input']:
            if parameters['convert_input'] == '255':
                classes //= 51
            tmp = np.zeros_like(classes)
            tmp[classes == 0] = cv.GC_BGD
            tmp[classes == 1] = cv.GC_PR_BGD
            tmp[classes == 2] = cv.GC_PR_FGD
            tmp[classes >= 3] = cv.GC_FGD
            classes = tmp

        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        cv.grabCut(image, classes, None, bgdmodel, fgdmodel, parameters['iterations'], cv.GC_INIT_WITH_MASK)

        if parameters['convert_output'] == '2':
            tmp = np.zeros_like(classes)
            tmp[classes == cv.GC_BGD] = 0
            tmp[classes == cv.GC_PR_BGD] = 0
            tmp[classes == cv.GC_PR_FGD] = 1
            tmp[classes == cv.GC_FGD] = 1
            classes = tmp
        elif parameters['convert_output'] == '4':
            tmp = np.zeros_like(classes)
            tmp[classes == cv.GC_BGD] = 0
            tmp[classes == cv.GC_PR_BGD] = 1
            tmp[classes == cv.GC_PR_FGD] = 2
            tmp[classes == cv.GC_FGD] = 3
            classes = tmp
        elif parameters['convert_output'] == '255-2':
            tmp = np.zeros_like(classes)
            tmp[classes == cv.GC_BGD] = 0
            tmp[classes == cv.GC_PR_BGD] = 0
            tmp[classes == cv.GC_PR_FGD] = 255
            tmp[classes == cv.GC_FGD] = 255
            classes = tmp
        elif parameters['convert_output'] == '255-4':
            tmp = np.zeros_like(classes)
            tmp[classes == cv.GC_BGD] = 0
            tmp[classes == cv.GC_PR_BGD] = 85
            tmp[classes == cv.GC_PR_FGD] = 170
            tmp[classes == cv.GC_FGD] = 255
            classes = tmp

        outputs["classes"] = Data(classes)

register_elements_auto(__name__, locals(), "Segmentation", 6)
