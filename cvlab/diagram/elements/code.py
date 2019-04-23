from .base import *


class CodeElement(NormalElement):
    name = "Code element"
    comment = "Runs user-given Python code"

    def __init__(self):
        super(CodeElement, self).__init__()
        self.compiled_code_str = ""
        self.compiled_code = None
        self.memory = {}

    def get_attributes(self):
        return [Input("input")], \
               [Output("output")], \
               [TextParameter("code", value=u"import cv2 as cv\nimport numpy as np\n#your code here\nreturn None",
                              window_content="def fun(image=numpy.array, parameters={}, intpoint=func, memory={}):",
                              window_title="Code editor"),
                ComboboxParameter("split_channels", OrderedDict([("Channels", True),("Image", False)]), "What to process", 1)]

    def compile(self, code):
        code = str(code)
        self.compiled_code_str = code
        self.compiled_code = None
        code = code.replace("\r\n", "\n")
        code = u"def fun(image, parameters, intpoint, memory):\n\t" + code.replace("\n", "\n\t") + u"\n\treturn None"
        c = compile(code, "<string>", 'exec')
        loc = {}
        exec(c, loc)
        self.compiled_code = loc["fun"]

    def process_channels(self, inputs, outputs, parameters):
        outputs["output"] = Data(self.compiled_code(inputs["input"].value, parameters, self.may_interrupt, self.memory))

    def process_inputs(self, inputs, outputs, parameters):
        if self.compiled_code_str != parameters["code"]:
            self.compile(parameters["code"])
        if not self.compiled_code: return
        if parameters["split_channels"]:
            NormalElement.process_inputs(self, inputs, outputs, parameters)
        else:
            self.process_channels(inputs, outputs, parameters)

    def get_source(self):
        # fixme: this won't work with 'process channels' set
        name = self.__class__.__name__.lower() + str(abs(hash(self)))[-8:]
        fun_code = self.parameters["code"].get().replace("intpoint()", "").replace("\r\n","\n").replace("\n","\n\t").replace("\t","    ")
        source = """
# inner code for {name}
def {name}_fun(image, parameters, memory):
    {fun_code}

# memory for {name}
{name}_memory = {{}}

# general code for {name}
def {name}(inputs, outputs, parameters):
    image = inputs["input"].value
    parameters = {{}}
    global {name}_memory
    result = {name}_fun(image, parameters, {name}_memory)
    outputs["output"] = Data(result)
""".format(**locals())
        return name, source, []




class CodeElementEx(CodeElement):
    name = "Code element (extended version)"

    def get_attributes(self):
        return [Input("in1", optional=True), Input("in2", optional=True), Input("in3", optional=True),
                Input("in4", optional=True)], \
               [Output("o1"), Output("o2"), Output("o3"), Output("o4")], \
               [TextParameter("code",
                              value=u"import cv2 as cv\nimport numpy as np\n#your code here\nreturn None, None, None, "
                                    u"None",
                              window_content="def fun(in1, in2, in3, in4, parameters={}, intpoint=func, memory={}):",
                              window_title="Code editor"),
                ComboboxParameter("split_channels", OrderedDict([("Channels", True),("Image", False)]), "What to process", 1)]

    def compile(self, code):
        code = str(code)
        self.compiled_code_str = code
        self.compiled_code = None
        code = code.replace("\r\n", "\n")
        code = u"def fun(in1, in2, in3, in4, parameters, intpoint, memory):\n\t" + code.replace("\n",
                                                                                                "\n\t") + \
               u"\n\treturn None, None, None, None"
        c = compile(code, "<string>", 'exec')
        loc = {}
        exec(c, loc)
        self.compiled_code = loc["fun"]

    def process_channels(self, inputs, outputs, parameters):
        ins = [None] * 4
        for i in range(4):
            n = "in" + str(i + 1)
            if n in inputs and inputs[n]:
                ins[i] = inputs[n].value
        o = self.compiled_code(ins[0], ins[1], ins[2], ins[3], parameters, self.may_interrupt, self.memory)
        for i, v in enumerate(o):
            n = "o" + str(i + 1)
            outputs[n] = Data(v)

    def get_source(self):
        # fixme: this won't work with 'process channels' set
        name = self.__class__.__name__.lower() + "_" + str(abs(hash(self)))[-8:]
        fun_code = self.parameters["code"].get().replace("intpoint()", "").replace("\r\n","\n").replace("\n","\n\t").replace("\t","    ")
        source = """
# inner code for {name}
def {name}_fun(in1, in2, in3, in4, parameters, memory):
    {fun_code}

# memory for {name}
{name}_memory = {{}}

# general code for {name}
def {name}(inputs, outputs, parameters):
    ins = [None] * 4
    for i in range(4):
        n = "in" + str(i + 1)
        if n in inputs and inputs[n]:
            ins[i] = inputs[n].value
    o = {name}_fun(ins[0], ins[1], ins[2], ins[3], parameters, {name}_memory)
    for i, v in enumerate(o):
        n = "o" + str(i + 1)
        outputs[n] = Data(v)
""".format(**locals())
        return name, source, []


class CodeElementSequence(CodeElementEx, SequenceToSequenceElement):
    name = "Code element (sequence output)"
    num_outputs = 8

    def get_attributes(self):
        return [Input("inputs", multiple=True)], \
               [Output("output")], \
               [TextParameter("code",
                              value=u"import cv2 as cv\nimport numpy as np\n\n#your code here\n\nreturn []",
                              window_content="def fun(inputs, parameters={}, intpoint=func, memory={}):",
                              window_title="Code editor"),
                IntParameter("outputs", min_=1, max_=100, value=8)]

    def get_processing_units(self, inputs, parameters):
        self.num_outputs = parameters["outputs"]
        return SequenceToSequenceElement.get_processing_units(self, inputs, parameters)

    def process_inputs(self, inputs, outputs, parameters):
        if len(inputs["inputs"].value) == 0: raise Exception("Connect some inputs")

        if self.num_outputs != parameters['outputs']:
            self.recalculate(True, True, True, True)
            self.may_interrupt()

        if self.compiled_code_str != parameters["code"]:
            self.compile(parameters["code"])
        if not self.compiled_code: return

        inputs = inputs["inputs"].desequence_all()
        o = self.compiled_code(inputs, parameters, self.may_interrupt, self.memory)
        o = list(o)[:self.num_outputs]
        o += [None] * (self.num_outputs-len(o))
        outputs["output"] = Sequence([ImageData(d) for d in o])

    def compile(self, code):
        code = str(code)
        self.compiled_code_str = code
        self.compiled_code = None
        code = code.replace("\r\n", "\n")
        code = u"def fun(inputs, parameters, intpoint, memory):\n\t" + code.replace("\n", "\n\t") + \
               u"\n\treturn [None, None, None, None]"
        c = compile(code, "<string>", 'exec')
        loc = {}
        exec(c, loc)
        self.compiled_code = loc["fun"]



register_elements_auto(__name__, locals(), "Code", 10)

