import json

from .parameters import PathParameter


class ComplexJsonEncoder(json.JSONEncoder):
    def __init__(self, base_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = base_path

    def encode(self, o):
        PathParameter.base_path = self.base_path
        encoded = super().encode(o)
        PathParameter.base_path = None
        return encoded

    def default(self, o):
        if hasattr(o, 'to_json'):
                return o.to_json()
        else:
            raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(o), repr(o)))


class ComplexJsonDecoder(json.JSONDecoder):
    def __init__(self, diagram, base_path):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)
        self.diagram_instance = diagram
        self.base_path = base_path

    def decode(self, s):
        PathParameter.base_path = self.base_path
        super().decode(s)
        PathParameter.base_path = None

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
