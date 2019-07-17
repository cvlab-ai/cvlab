import sys
import importlib
import os
from collections import defaultdict

import re

ignored_modules = ["sample", "testing"]


registered_elements = defaultdict(list)  # package -> [elements]
sort_keys = defaultdict(lambda: 999)
all_elements = {}
plugin_callbacks = []


def element_name(name):
    name = re.match(r"(cvlab[^.]*\.)?(diagram\.)?(elements\.)?(experimental\.|testing\.|custom\.)?(.+)", name).group(5)
    return name


def register_elements(package, elements, sort_key=999):
    sort_keys[package] = min(sort_key, sort_keys[package])
    for element in elements:
        element_package = getattr(element, "package", None) or package
        module = element.__module__
        classname = element.__name__
        name = module + "." + classname
        registered_elements[element_package].append(element)
        all_elements[element_name(name)] = element
        sort_keys[element_package] = min(sort_keys[element_package], sort_key + 100)


def register_elements_auto(module_name, module_locals, package, sort_key=999):
    elements = [cls for cls in module_locals.values() if isinstance(cls, type) and cls.__module__ == module_name and hasattr(cls, "name")]
    return register_elements(package, elements, sort_key)


def get_sorted_elements():
    return sorted(registered_elements.items(), key=lambda kv: sort_keys[kv[0]])


def get_element_fallback(name):
    class_name = name.split(".")[-1]
    for element in all_elements.values():
        if element.__name__ == class_name:
            print("WARN: Loading fallback element. Requested name: {name}. Returned class: {element}".format(**locals()))
            return element
    raise Exception("Cannot find element " + name)


def get_element(name):
    name = element_name(name)
    element = all_elements.get(name, None)
    if not element: element = get_element_fallback(name)
    return element


def available_modules(path):
    modules = []
    dir = os.path.realpath(path)
    if not os.path.isdir(dir): dir = os.path.dirname(dir)
    for entry in os.listdir(dir):
        if entry == '__init__.py':
            continue
        if entry[-3:] == ".py":
            entry = entry[:-3]
        elif not os.path.isdir(dir + "/" + entry):
            continue
        if os.path.isdir(dir + "/" + entry) and not os.path.exists(dir + "/" + entry + "/__init__.py"):
            continue
        modules.append(entry)
    return modules


def load_modules(modules, package):
    for module in modules:
        if module in ignored_modules:
            print("Ignoring module:", module)
            continue
        try:
            print("Loading module:", module)
            importlib.import_module("." + module, package)
        except BaseException as e:
            print("ERROR during loading module {module}: {e}".format(**locals()))
            # if __debug__:
            #     traceback.print_exc()


def load_plugins():
    for path in sys.path:
        if not os.path.isdir(path): continue
        for module in os.listdir(path):
            if not module.startswith("cvlab_"): continue
            module_path = path + "/" + module
            if os.path.isdir(module_path) and not os.path.isfile(module_path + "/__init__.py"): continue
            if module in sys.modules: continue
            try:
                print("Loading plugin:", module)
                importlib.import_module(module)
            except Exception as e:
                print("Error while loading module", module, ":", e)


def load_auto(path):
    modules = available_modules(path)
    package = path.replace("\\","/")
    package = re.match(r".*(cvlab(_\w+)?/.+)", package).group(1)
    package = re.sub(r"/__init__\.py.*", "", package)
    package = package.replace("/",".")
    package = package.replace("cvlab.cvlab","cvlab")
    package = package.replace("cvlab.cvlab","cvlab")
    load_modules(modules, package)


def add_plugin_callback(callback):
    plugin_callbacks.append(callback)


load_auto(__file__)
load_plugins()
