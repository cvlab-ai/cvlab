import inspect
import re
from collections import OrderedDict

import cv2


UNKNOWN = ""
MATRIX = "matrix"
BOOL = "bool"
INT = "int"
FLOAT = "float"
STRING = "str"
SCALAR = "scalar"
SIZE = "size"
POINT = "point"
ENUM = "enum"


class Argument:
    def __init__(self, name, function, optional=None, description=""):
        self.name = name
        self.function = function
        self.optional = optional
        self.description = description
        self.type = UNKNOWN
        self.odd = None
        self.min = None
        self.max = None
        self.enum = {}
        self.multi_enum = None

    def parse_doc(self):
        name = self.name.lower()
        if name in ("image", "src", "dst", "img", "markers", "points", "m", "mask", "mtx", "edges", "roots", "window" \
            "centers", "bestlabels", "inpaintmask") \
            or "matr" in name or "coeffs" in name or "src" in name or "points" in name or "image" in name:
            self.type = MATRIX
        elif name in ("retval",):
            self.type = INT
        elif name in ("flags",):
            self.type = ENUM
            self.enum = self.auto_constants()
            self.multi_enum = True
        elif name in ("interpolation","bordermode","bordertype","ddepth","dtype","code") or (name == "type" and self.function == "threshold"):
            self.type = ENUM
            self.enum = self.auto_constants()
            self.multi_enum = False
        elif name in ("size","dsize") or "size" in name:
            self.type = SIZE
        elif name in ("point","center","anchor","pt1","pt2","pt3","pt4","org"):
            self.type = POINT
        elif name in ("count","dx","dy","thickness") or "radius" in name or "iteration" in name or "iters" in name:
            self.type = INT
            self.min = 0
        elif name in ("maxval","minval","angle","scale") or "thresh" in name \
            or "sigma" in name or "alpha" in name or "gamma" in name or "beta" in name or "scale" in name:
            self.type = FLOAT
        elif name in ("bordervalue","color"):
            self.type = SCALAR
        elif name in ("l2gradient",):
            self.type = BOOL
        elif "name" in name or name in ("text","string"):
            self.type = STRING

    def all_constants(self, prefix, consts):
        for name, value in vars(cv2).items():
            if name.startswith(prefix):
                consts[name] = value

    def detect_constants(self):
        consts = {}
        for name in re.findall(r"[A-Z0-9]+_[A-Z0-9_]+", self.description):
            if hasattr(cv2, name):
                consts[name] = getattr(cv2, name)
        return consts

    def auto_constants(self):
        types = {
            "ddepth": ["CV_8", "CV_16", "CV_32", "CV_64"],
            "dtype": ["CV_8", "CV_16", "CV_32", "CV_64"],
            "bordertype": ["BORDER_"],
            "bordermode": ["BORDER_"],
            "houghmode": ["HOUGH_"],
            "thresholdtype": ["THRESH_"],
            "colormaptype": ["COLORMAP_"],
            "linetype": ["LINE_"],
            "interpolationflag": ["INTER_"],
            "code": ["COLOR_"],
        }

        consts = self.detect_constants()

        if self.name in ("ddepth","dtype"):
            consts["NONE"] = 0

        if self.name in types:
            for prefix in types[self.name]:
                self.all_constants(prefix, consts)

        for name in consts.copy():
            prefix = re.match(r"([A-Z0-9]_).*", name)
            if not prefix: continue
            prefix = prefix.groups(1)
            self.all_constants(prefix, consts)

        for tag in re.findall(r"#\w+", self.description):
            tag = tag[1:]
            for typename, prefixes in types.items():
                if tag.lower().startswith(typename):
                    for prefix in prefixes:
                        self.all_constants(prefix, consts)

        for k in consts:
            if "cv2" in k:
                print("DUPA")

        return consts

    def source_input(self):
        optional = ", optional=True" if self.optional else ""
        return "Input('{self.name}', '{self.name}'{optional})".format(**locals())

    def source_output(self):
        return "Output('{self.name}', '{self.name}')".format(**locals())

    def source_param(self):
        if self.type == INT:
            return "IntParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == FLOAT:
            return "FloatParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == SIZE:
            return "SizeParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == POINT:
            return "PointParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == STRING:
            return "TextParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == SCALAR:
            return "ScalarParameter('{self.name}', '{self.name}')".format(**locals())
        if self.type == ENUM:
            if not self.enum: raise Exception("No values given for enum")
            values = ["('" + name + "'," + repr(value) + ")" for name, value in self.enum.items()]
            values = ",".join(values)
            return "ComboboxParameter('{self.name}', [{values}])".format(**locals())
        raise Exception("Cannot get parameter source for argument " + self.name)


    def __repr__(self):
        s = "Argument(" + self.name
        if self.type: s += ", " + self.type
        if self.optional is not None: s += ", opt:" + str(self.optional)
        if self.min is not None: s += ", min:" + str(self.min)
        if self.max is not None: s += ", max:" + str(self.max)
        if self.odd is not None: s += ", odd:" + str(self.odd)
        if self.enum: s += ", enum:" + str(self.enum)
        if self.multi_enum is not None: s += ", multi:" + str(self.multi_enum)
        if self.description: s += ", '" + self.description + "'"
        s += ")"
        return s


class Function:
    class_template = """\
class {class_name}(NormalElement):
    name = '{element_name}'
    comment = '''{element_comment}'''

    def get_attributes(self):
        return [{inputs_def}], \\
               [{outputs_def}], \\
               [{params_def}]

    def process_inputs(self, inputs, outputs, parameters):
        {code}
"""
    params_indent = "                "
    code_indent = "        "

    def __init__(self, obj):
        self.obj = obj
        self.doc = self.obj.__doc__
        self.name = obj.__name__
        self.args = OrderedDict()  # name -> Argument
        self.ret = OrderedDict()  # name -> Argument

    def parse_doc(self):
        doc = self.doc.splitlines()

        line = doc[0]
        print("DEBUG", line)
        name, args, opt, ret = re.match(r"(\w+)\((.*?)(\[.*\])?\) -> (.*)", line).groups()
        args = re.findall(r"\w+", args or "")
        opt = re.findall(r"\w+", opt or "")
        ret = re.findall(r"\w+", ret or "")

        # if not self.args: raise Exception("No arguments")
        # if not self.ret: raise Exception("No return value")

        descriptions = {}
        param = None
        desc = None
        for line in doc:
            if not line: continue
            if line[0] == '.': line = line[1:]
            line = line.strip()
            if not line: continue
            if line[0] == "@":
                if param:
                    descriptions[param] = desc
                param = desc = ""
            if line.startswith("@param"):
                param, desc = re.match(r"@param (\w+) (.+)", line).groups()
            elif param:
                desc += "\n" + line
        if param:
            descriptions[param] = desc

        for a in args: self.args[a] = Argument(a, self.name, optional=False, description=descriptions.get(a,""))
        for a in opt: self.args[a] = Argument(a, self.name, optional=True, description=descriptions.get(a,""))
        for a in ret:
            if a != "None":
                self.ret[a] = Argument(a, self.name, description=descriptions.get(a,""))

        for a in self.args.values():
            a.parse_doc()
        for a in self.ret.values():
            a.parse_doc()

    def source(self):
        element_name = self.name[0].upper() + self.name[1:]
        element_comment = self.doc.replace("\n.   ","\\n")
        class_name = "OpenCVAuto2_" + element_name

        args = []
        for arg in self.args.values():
            if arg.optional and arg.name in self.ret:
                print("DEBUG Ignoring optional argument because it is also returned:", arg)
                continue
            if not arg.type:
                if arg.optional:
                    print("WARN Ignoring optional argument:", arg)
                    continue
                else:
                    raise Exception("Unknown type of mandatory argument:", arg)
            else:
                args.append(arg)

        inputs = [arg for arg in args if arg.type == MATRIX]
        params = [arg for arg in args if arg.type != MATRIX]
        outputs = [arg for arg in self.ret.values() if arg.name != "retval"]

        if not inputs and not params:
            raise Exception("No inputs nor params recognized")

        if not outputs:
            raise Exception("Function does not return anything")

        inputs_def = ""
        for arg in inputs:
            if inputs_def: inputs_def += ",\n" + self.params_indent
            inputs_def += arg.source_input()
            
        outputs_def = ""
        for arg in outputs:
            if outputs_def: outputs_def += ",\n" + self.params_indent
            outputs_def += arg.source_output()

        params_def = ""
        for arg in params:
            if params_def: params_def += ",\n" + self.params_indent
            params_def += arg.source_param()

        # code
        code = ""

        for arg in inputs:
            if code: code += "\n" + self.code_indent
            code += "{arg.name} = inputs['{arg.name}'].value".format(**locals())

        for arg in params:
            if code: code += "\n" + self.code_indent
            code += "{arg.name} = parameters['{arg.name}']".format(**locals())

        ret_names = ", ".join(arg.name for arg in self.ret.values())
        func_args = ", ".join(["{arg.name}={arg.name}".format(**locals()) for arg in inputs] + ["{arg.name}={arg.name}".format(**locals()) for arg in params])

        code += "\n" + self.code_indent
        code += "{ret_names} = cv2.{self.name}({func_args})".format(**locals())

        for arg in outputs:
            if code: code += "\n" + self.code_indent
            code += "outputs['{arg.name}'] = Data({arg.name})".format(**locals())

        return self.class_template.format(**locals())


def process(name, obj):
    function = Function(obj)
    function.parse_doc()
    source = function.source()
    print(" 100% ")
    print(source)
    return source


def process_cv2():
    source = """\
# Source generated with cvlab/tools/generate_opencv.py
# See: https://github.com/cvlab-ai/cvlab
   
import cv2
from cvlab.diagram.elements.base import *

"""
    for name, obj in vars(cv2).items():
        try:
            if not inspect.isbuiltin(obj) and not inspect.isfunction(obj): continue
            print("INFO", name)
            function_source = process(name, obj)
            source += "### " + name + " ###\n\n"
            source += function_source
            source += "\n"
        except Exception as e:
            print("ERROR", e)
        print()

    source += "\n\n"
    source += 'register_elements_auto(__name__, locals(), "OpenCV autogenerated 2", 15)'
    source += "\n\n"

    with open("opencv_auto2.py", 'w') as f:
        f.write(source)

    print("DONE. Source saved to file: opencv_auto2.py")


if __name__ == '__main__':
    process_cv2()
