class Input:
    def __init__(self, id, name=None, multiple=False, optional=False):
        super(Input, self).__init__()
        self.id = id
        self.parent = None
        if name is None: name = id
        self.name = name
        self.multiple = multiple
        self.optional = optional
        self.connected_from = []
        self.hook = None
        from .diagram import Diagram
        self.diagram_write_lock = Diagram.diagram_lock.writer

    def connect(self, output):
        with self.diagram_write_lock:
            if output in self.connected_from: return
            if not self.multiple:
                self.disconnect_all()
            self.connected_from.append(output)
            self.hook.connected(output.hook)

    def disconnect(self, output):
        with self.diagram_write_lock:
            if output not in self.connected_from: return
            self.connected_from.remove(output)
            output.disconnect(self)
            self.hook.disconnected(output.hook)
            self.parent.diagram.notify_disconnect(output, self)

    def disconnect_all(self):
        with self.diagram_write_lock:
            for output in list(self.connected_from):
                self.disconnect(output)

    def get(self):
        return self.hook.get_data()


class Output:
    def __init__(self, id, name=None, desequencing=False):
        super(Output, self).__init__()
        self.id = id
        self.parent = None
        if name is None: name = id
        self.name = name
        self.desequencing = desequencing
        self.connected_to = []
        self.hook = None
        from .diagram import Diagram
        self.diagram_write_lock = Diagram.diagram_lock.writer

    def connect(self, input_):
        with self.diagram_write_lock:
            if input_ in self.connected_to: return
            self.connected_to.append(input_)
            self.hook.connected(input_.hook)

    def disconnect(self, input_):
        with self.diagram_write_lock:
            if input_ not in self.connected_to: return
            self.connected_to.remove(input_)
            input_.disconnect(self)
            self.hook.disconnected(input_.hook)

    def disconnect_all(self):
        with self.diagram_write_lock:
            for input_ in list(self.connected_to):
                self.disconnect(input_)

    def put(self, data):
        self.hook.set_data(data)

    def get(self):
        return self.hook.get_data()


