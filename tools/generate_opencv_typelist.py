import json
import os
import sys


class TypeDef:
    def __init__(self, name="", type="", args=None):
        self.name = name if name not in ("from","import","def") else name + "_"
        self.type = type
        self.args = args or []

    @classmethod
    def from_declaration(cls, declaration):
        name = declaration[0]
        assert isinstance(name, str)

        if name.startswith("class"):
            return TypeDef(name=name[9:], type="class")
        if name.startswith("struct"):
            return TypeDef(name=name[10:], type="struct")
        if name.startswith("const"):
            return TypeDef(name=name[9:], type="const")
        if not name.startswith("cv."):
            raise Exception("Cannot read type definition from:", declaration)

        name = name[3:]
        if len(name.split(".")) > 1:
            return TypeDef(name=name, type="class_func")

        from gen2 import FuncVariant
        info = FuncVariant("", name, declaration, False)
        type = TypeDef(name=name, type="func")
        print("INFO Function: cv2." + name)
        for arg in info.args:
            arg = TypeDef.from_arg(arg)
            type.args.append(arg)

        return type

    @classmethod
    def from_arg(cls, arg):
        from gen2 import ArgInfo
        assert isinstance(arg, ArgInfo)
        type = TypeDef(arg.name)
        if arg.tp in ("Mat","Array","vector_Mat","vector_int","vector_float","vector_Point","vector_Rect","vector_uchar"): type.type = "mat"
        elif arg.tp in ("Size",): type.type = "size"
        elif arg.tp in ("int","long","long int","size_t"): type.type = "int"
        elif arg.tp in ("double","float","long double"): type.type = "float"
        elif arg.tp in ("Point","Point2i"): type.type = "point"
        elif arg.tp in ("Point2f","Point2d"): type.type = "point_float"
        elif arg.tp in ("Scalar",): type.type = "scalar"
        elif arg.tp in ("bool",): type.type = "bool"
        elif arg.tp in ("String","string"): type.type = "str"
        elif arg.tp in ("Rect",): type.type = "rect"
        elif arg.tp in ("RotatedRect",): type.type = "rect_rotated"
        elif arg.tp in ("TermCriteria",): type.type = "term_criteria"

        if type.type:
            print("DEBUG Argument type:", arg.name, "[", arg.tp, "] ->", type.type)
        else:
            print("WARNING Cannot detect type from argument:", arg.name, arg.tp)

        return type


def get_typelist():
    import hdr_parser
    headers = hdr_parser.opencv_hdr_list
    parser = hdr_parser.CppHeaderParser(False)
    types = {}
    for header in headers:
        for declaration in parser.parse(header):
            function = TypeDef.from_declaration(declaration)
            if not function: continue
            for arg in function.args:
                name = "{function.name}:{arg.name}".format(**locals()).lower()
                type = arg.type
                if type:
                    if types.get(name, type) != type:
                        print("WARNING Inconsistent types: " + name)
                    types[name] = type
    return types



def generate_typelist(opencv_source_dir, destination_dir):
    working_dir = os.getcwd()
    base_dir = opencv_source_dir + "/modules/python/src2"
    os.chdir(base_dir)
    sys.path.append(base_dir)
    types = get_typelist()
    with open(destination_dir + "/typelist.json", "w") as f:
        json.dump(types, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    opencv_source_dir = sys.argv[1]
    destination_dir = os.path.dirname(__file__)
    generate_typelist(opencv_source_dir, destination_dir)