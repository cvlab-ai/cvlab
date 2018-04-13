
from __future__ import unicode_literals
import json


class ComplexJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, 'to_json'):
                return o.to_json()
        else:
            raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(o), repr(o)))


class ComplexJsonDecoder(json.JSONDecoder):
    def __init__(self, diagram):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)
        self.diagram_instance = diagram

    def dict_to_object(self, d):
        if "_type" in d and d["_type"] == "element":
            class_name = d["class"]
            module_name = d["module"]
            from .elements import get_element
            element = get_element(module_name + "." + class_name)()
            element.from_json(d)
            return element
        if "_type" in d and d["_type"] == "diagram":
            self.diagram_instance.from_json(d)
            return self.diagram_instance
        return d
