import json
import os
import re
import sys
from glob import glob


class TypeDef:
    def __init__(self, name="", type="", args=None, namespace=""):
        self.name = name if name not in ("from","import","def") else name + "_"
        self.type = type
        self.args = args or []
        self.namespace = namespace

    @classmethod
    def from_declaration(cls, declaration):
        name = declaration[0]
        assert isinstance(name, str)

        namespace = ".".join(name.split()[-1].split(".")[1:-1])

        if name.startswith("class"):
            return TypeDef(name=name[9:], type="class", namespace=namespace)
        if name.startswith("struct"):
            return TypeDef(name=name[10:], type="struct", namespace=namespace)
        if name.startswith("const"):
            return TypeDef(name=name[9:], type="const", namespace=namespace)
        if not name.startswith("cv."):
            raise Exception("Cannot read type definition from:", declaration)

        name = name[3:]
        if len(name.split(".")) > 1:
            return TypeDef(name=name, type="class_func", namespace=namespace)

        from gen2 import FuncVariant
        info = FuncVariant("", name, declaration, False)
        type = TypeDef(name=name, type="func", namespace=namespace)
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


def get_typelist(headers):

    group_map = {

    }

    import hdr_parser
    headers = list(headers) + hdr_parser.opencv_hdr_list
    headers = map(os.path.abspath, headers)
    headers = set(headers)

    parser = hdr_parser.CppHeaderParser(False)

    types = {}
    namespaces = {}
    groups = {}

    for header in sorted(headers):
        header = header.replace("\\","/")

        print("INFO Processing header file:", header)
        if not header or not os.path.exists(header): continue

        group = "modules/" + re.match(r".*/modules/(.+)", header).group(1)
        # group = group_map.get(group, None)

        for declaration in parser.parse(header):
            function = TypeDef.from_declaration(declaration)
            if not function: continue

            if function.type == "func":
                namespaces[function.name] = function.namespace
                groups[function.name] = group

            for arg in function.args:
                name = "{function.name}:{arg.name}".format(**locals()).lower()
                type = arg.type
                if type:
                    if types.get(name, type) != type:
                        print("ERROR Inconsistent types: " + name)
                    types[name] = type

    return types, namespaces, groups


def doxygen_groups(build_dir):
    import bs4
    groups = {}
    dir = build_dir + "/doc/doxygen/html"
    for file in sorted(glob(dir + "/*/*/group*.html")):
        try:
            html = open(file).read()
            html = bs4.BeautifulSoup(html,features="lxml")

            title = html.select_one(".title")
            group = title.contents[0].strip()

            # Extract parent groups
            # group = []
            # for n in title.select("a.el"):
            #     group.append(n.text.strip())
            # group.append(title.contents[0].strip())
            # group = " : ".join(group)

            print("DEBUG Group for file:", file, "->", group)

            functions = html.select_one("a[name=func-members]").parent.parent.parent.parent.select("tr .memItemRight")
            for function in functions:
                fname = function.select_one("a.el").text.strip()
                fname = fname.replace("cv::","")
                fname = fname.replace("::",".")
                fname = re.sub(r"<[^>]*>","",fname)
                if "operator" in fname: continue
                fname = fname.lower()

                print("DEBUG Function:", fname)

                groups[fname] = group

        except Exception as e:
            print("ERROR", e)

    return groups


def generate_typelist(source_dir, build_dir, destination_dir):
    headers = open(build_dir + "/modules/python_bindings_generator/headers.txt").readlines()
    headers = [h.strip() for h in headers if h.strip()]

    working_dir = os.getcwd()
    base_dir = source_dir + "/modules/python/src2"
    os.chdir(base_dir)
    sys.path.append(base_dir)

    groups = doxygen_groups(build_dir)
    types, namespaces, _ = get_typelist(headers)

    with open(destination_dir + "/typelist.json", "w") as f:
        json.dump(types, f, indent=2, sort_keys=True)

    with open(destination_dir + "/typelist_groups.json", "w") as f:
        json.dump(groups, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    opencv_source_dir = sys.argv[1]
    opencv_build_dir = sys.argv[2]
    destination_dir = os.path.dirname(__file__)
    generate_typelist(opencv_source_dir, opencv_build_dir, destination_dir)