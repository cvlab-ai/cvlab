# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from six import itervalues, iteritems

import importlib
import os
import traceback
from collections import defaultdict

import re

ignored_modules = ["sample", "testing"]


registered_elements = defaultdict(list)  # package -> [elements]
sort_keys = defaultdict(lambda: 999)
all_elements = {}


def element_name(name):
    name = re.match(r"(cvlab\.)?(diagram\.)?(elements\.)?(experimental\.|testing\.|custom\.)?(.+)", name).group(5)
    return name


def register_elements(package, elements, sort_key=999):
    registered_elements[package] += elements
    sort_keys[package] = min(sort_key, sort_keys[package])
    for element in elements:
        module = element.__module__
        classname = element.__name__
        name = module + "." + classname
        all_elements[element_name(name)] = element


def register_elements_auto(module_name, module_locals, package, sort_key=999):
    elements = [cls for cls in itervalues(module_locals) if isinstance(cls, type) and cls.__module__ == module_name and hasattr(cls, "name")]
    return register_elements(package, elements, sort_key)


def get_sorted_elements():
    return sorted(iteritems(registered_elements), key=lambda kv: sort_keys[kv[0]])


def get_element_fallback(name):
    class_name = name.split(".")[-1]
    for element in itervalues(all_elements):
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


def load_auto(path):
    modules = available_modules(path)
    package = os.path.dirname(path).replace(os.path.dirname(__file__), "").replace("\\","/").replace("/",".")
    package = "cvlab.diagram.elements" + package
    load_modules(modules, package)


load_auto(__file__)
