from _thread import get_ident

import itertools

from PyQt5.QtCore import pyqtSignal, QObject, QReadWriteLock, QTimer, pyqtSlot

from ..view.styles import StyleManager
from .element import *
from .errors import GeneralException
from .serialization import ComplexJsonEncoder, ComplexJsonDecoder
from ..version import __version__


class ReadLocker:
    def __init__(self, parent):
        assert isinstance(parent, DiagramLock)
        self.parent = parent

    def __enter__(self):
        if not self.parent.tryLockForRead():
            if self.parent.owner == get_ident():
                self.parent.lockForWrite()
            else:
                self.parent.lockForRead()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.parent.unlock()


class WriteLocker:
    def __init__(self, parent):
        assert isinstance(parent, DiagramLock)
        self.parent = parent
        self.count = 0

    def __enter__(self):
        self.parent.lockForWrite()
        self.count += 1
        self.parent.owner = get_ident()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count -= 1
        if not self.count:
            self.parent.owner = None
        self.parent.unlock()


class DiagramLock(QReadWriteLock):
    def __init__(self):
        QReadWriteLock.__init__(self, QReadWriteLock.Recursive)
        self.writer = WriteLocker(self)
        self.reader = ReadLocker(self)
        self.owner = None


class Diagram(QObject):
    """The processing diagram"""

    # todo: if the singals and QObject dependency are removed, Workarea or ScrolledWorkarea could derive from
    # Diagram, which would possibly make thing simpler
    element_added = pyqtSignal(Element, tuple)
    element_deleted = pyqtSignal(Element)
    connection_created = pyqtSignal(Output, Input)
    connection_deleted = pyqtSignal(Output, Input)

    # global diagram lock for connecting and disconnecting elements
    diagram_lock = DiagramLock()

    def __init__(self):
        super(Diagram, self).__init__()
        self.elements = set()
        self.connections = []
        self.painter = None
        self.zoom_level = 1.0

    def clear(self):
        for e in list(self.elements):
            self.delete_element(e)

    def add_element(self, e, position):
        if not self.painter:
            raise GeneralException("Elements cannot be added to Diagram until the painter is set")
        e.diagram = self
        self.elements.add(e)
        self.element_added.emit(e, position)

    def delete_element(self, e):
        # todo: if we remove element, to which other elements try to access, we gonna have trouble
        with self.diagram_lock.writer:
            to_connect = []
            if len(e.inputs) == 1 and len(e.outputs) == 1:
                input_ = list(e.inputs.values())[0]
                output = list(e.outputs.values())[0]
                pre = input_.connected_from
                post = output.connected_to
                if len(pre) > 0 and len(post) > 0:
                    for i, o in itertools.product(pre, post):
                        # todo: it's not optimal when 'i' is multiple but 'o' is not
                        #self.connect_io(i, o)
                        to_connect.append((i, o))
        for p in list(e.parameters.values()):
            p.disconnect_all_children()
        for io in list(e.outputs.values()) + list(e.inputs.values()):
            io.disconnect_all()
            self.delete_connections_with_connector(io)
        e.delete()
        self.elements.remove(e)
        self.element_deleted.emit(e)
        for i, o in to_connect:
            self.connect_io(i, o)

    def delete_connections_with_connector(self, io):
        for connection in [c for c in self.connections if io in c]:
            self.connections.remove(connection)

    def connect_io(self, o1, o2):
        if o1 is o2: return
        with self.diagram_lock.writer:
            if isinstance(o1, Input):
                input_ = o1
                output = o2
            else:
                input_ = o2
                output = o1
            if Diagram.makes_loop(output, input_):
                print("WARNING Connection loop: {}:{} -> {}:{}".format(output.parent.name, output.name, input_.parent.name, input_.name))
                # raise ConnectError("This connection would create an infinite loop!")
                return
            input_.connect(output)
            output.connect(input_)
            if output.desequencing:
                output.hook.actualize_outputs()
        self.connections.append((output, input_))
        self.connection_created.emit(output, input_)

    def disconnect_io(self, o1, o2):
        if o1 is o2: return
        with self.diagram_lock.writer:
            if isinstance(o1, Input):
                input_ = o1
                output = o2
            else:
                input_ = o2
                output = o1
            output.disconnect(input_)
            input_.disconnect(output)
            if output.desequencing:
                output.hook.actualize_outputs()
        self.connections.remove((output, input_))

    def notify_disconnect(self, output, input_):
        self.connection_deleted.emit(output, input_)

    @staticmethod
    def makes_loop(output, input_):
        # DFS
        with Diagram.diagram_lock.writer:
            q = [input_.parent]
            c = set()
            while q:
                act = q.pop()
                if act in c:
                    continue
                if act is output.parent:
                    return True
                c.add(act)
                for i in list(act.outputs.values()):
                    for o in i.connected_to:
                        if o.parent is output.parent:
                            return True
                        if o.parent not in c:
                            q.append(o.parent)
        return False

    def set_painter(self, painter):
        self.painter = painter

    def save_to_json(self, base_path):
        return ComplexJsonEncoder(indent=2, sort_keys=True, base_path=base_path).encode(self)

    def load_from_json(self, ascii_data, base_path):
        if not self.painter:
            raise GeneralException("Diagram cannot be filled with data until the painter is set")
        ComplexJsonDecoder(self,base_path).decode(ascii_data)
        QTimer.singleShot(100, self.update_previews)

    @pyqtSlot()
    def update_previews(self):
        for e in self.elements:
            e.preview.force_update()

    def to_json(self):
        elements = {}
        elements_orders = {}
        wires = {}

        for e in self.elements:
            e_order = self.painter.element_z_index(e)
            # z-indexes are unique, but just in case:
            while e_order in elements:
                e_order += 1
            elements[e_order] = e
            elements_orders[e] = e_order

        for c_order, c in enumerate(self.connections):
            wire = {
                "from_element": elements_orders[c[0].parent],
                "from_output": c[0].id,
                "to_element": elements_orders[c[1].parent],
                "to_input": c[1].id
            }
            wires[c_order] = wire

        param_ids = {}
        for e_id, e in sorted(elements.items()):
            for par in e.parameters.values():
                param_ids[par] = len(param_ids)

        connected_params = []
        for par, par_id in param_ids.items():
            assert isinstance(par, Parameter)
            for to in par.children:
                assert isinstance(to, Parameter)
                connected_params.append({
                    "from": par_id,
                    "to": param_ids[to]
                })
                # connected_params.append([par_id, param_ids[to]])

        filetype = "CV-Lab diagram save file. See: https://github.com/cvlab-ai/cvlab "

        return {"_type": "diagram", "elements": elements, "wires": wires, "params": connected_params,
                "zoom_level": self.zoom_level, "_version": __version__, "_filetype": filetype}

    def from_json(self, data):
        #TODO: catch json parsing errors and present proper message
        elements = {}
        sorted_orders = sorted(map(int, data["elements"]))  # sorting is important for preserving z-index

        for e_order in sorted_orders:
            e = data["elements"][str(e_order)]

            # workaround for old versions, where hidpi was ignored in saved diagrams
            if StyleManager.is_highdpi and data.get("_version","") < "1.2.1":
                e.move(e.pos().x()//2,e.pos().y()//2)
                e.preview.preview_size //= 2

            self.add_element(e, (e.pos().x(), e.pos().y()))
            elements[e_order] = e

        sorted_orders = sorted(map(int, data["wires"]))     # sorting is important for preserving connections order
        for c_order in sorted_orders:
            connection = data["wires"][str(c_order)]
            from_e_id = connection["from_element"]
            from_o_id = connection["from_output"]
            to_e_id = connection["to_element"]
            to_i_id = connection["to_input"]

            # todo: remove it. It's only a workaround for old Forwarder element
            if elements[from_e_id].name == "Forwarder": from_o_id = "output"
            if elements[to_e_id].name == "Forwarder": to_i_id = "input"

            a = elements[from_e_id].outputs[from_o_id]
            b = elements[to_e_id].inputs[to_i_id]
            self.connect_io(a, b)

        if 'params' in data:
            param_ids = {}
            for e_id, e in sorted(elements.items()):
                for par in e.parameters.values():
                    param_ids[len(param_ids)] = par

            for con in data['params']:
                from_ = con['from']
                to_ = con['to']
                param_ids[from_].connect_child(param_ids[to_])

        self.zoom_level = data.get("zoom_level", 1.0)
