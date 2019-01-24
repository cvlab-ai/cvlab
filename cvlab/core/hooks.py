from ..diagram.diagram import Diagram
from ..diagram.interface import *


class Notifier:
    def __init__(self):
        self._lock = threading.Condition()
        self._do_notify = False

    def notify(self):
        with self._lock:
            self._do_notify = True
            self._lock.notifyAll()

    def wait(self):
        with self._lock:
            while not self._do_notify:
                self._lock.wait()
            self._do_notify = False

    def is_set(self):
        return self._do_notify


class Hook(QObject, object):
    """Hook for joining diagram elements together"""

    def __init__(self, connector):
        super(Hook, self).__init__()
        self.connector = connector
        self.lock = RLock()


class InputHook(Hook):
    """Hook for getting some input"""

    def __init__(self, connector):
        super(InputHook, self).__init__(connector)
        assert isinstance(self.connector, Input)
        self.sequence_indices = OrderedDict()
        if not self.connector.optional:
            self.empty_data = Sequence() if self.connector.multiple else EmptyData()
        else:
            self.empty_data = EmptyOptionalData()
        self.data = self.empty_data

    def get_data(self):
        return self.data

    def set_data(self, data, from_hook):
        assert data is not None
        assert from_hook is not None
        if self.connector.multiple:
            with self.lock:
                assert from_hook in self.sequence_indices
                index = self.sequence_indices[from_hook]
                if data is self.data.value[index]:
                    return
                self.data.value[index] = data
        else:
            with self.lock:
                if self.data is data: return
                self.data = data
        self.connector.parent.recalculate(False, True, True)

    def delete(self):
        pass

    def connected(self, from_hook):
        assert from_hook is not None
        if self.connector.multiple:
            with self.lock:
                assert from_hook not in self.sequence_indices
                self.sequence_indices[from_hook] = len(self.data.value)
                self.data = Sequence(self.data.value + [EmptyData()])
        # self.connector.parent.recalculate(False, True, True)

    def disconnected(self, from_hook):
        if self.connector.multiple:
            with self.lock:
                assert from_hook in self.sequence_indices
                del self.sequence_indices[from_hook]
                self.data = Sequence([self.data.value[i] for i in self.sequence_indices.values()])
                for i, hook in enumerate(self.sequence_indices.keys()):
                    self.sequence_indices[hook] = i
        else:
            self.data = self.empty_data
        self.connector.parent.recalculate(False, True, True)


class OutputHook(Hook):
    """Hook for giving output"""

    def __init__(self, connector):
        super(OutputHook, self).__init__(connector)
        assert isinstance(self.connector, Output)
        self.empty_data = Data(None, Data.NONE)
        self.data = self.empty_data

    # gui thread only
    # @pyqtSlot(InputHook)
    def connected(self, to_hook):
        self.actualize_outputs()  # fixme: to jest chyba nadmiarowe...

    # gui thread only
    # @pyqtSlot(InputHook)
    def disconnected(self, to_hook):
        self.actualize_outputs()  # fixme: to jest chyba nadmiarowe...

    # element thread only
    # @pyqtSlot(object)
    def set_data(self, data):
        with self.lock:
            self.data = data if data is not None else self.empty_data
        self.actualize_outputs()

    # gui thread or element thread
    # @pyqtSlot()
    def actualize_outputs(self):
        with Diagram.diagram_lock.reader, self.lock:
            if self.connector.desequencing and self.data and self.data.type() == Data.SEQUENCE and len(
                    self.connector.connected_to) > 1:
                for input, data in zip(self.connector.connected_to, self.data.value):
                    input.hook.set_data(data, self)
            else:
                for i in self.connector.connected_to:
                    i.hook.set_data(self.data, self)

    # gui thread or other element thread
    def get_data(self):
        return self.data


class InputQtHook(InputHook):
    set_data_signal = pyqtSignal(object, OutputHook)

    def __init__(self, connector):
        super(InputQtHook, self).__init__(connector)
        self.set_data_signal.connect(super(InputQtHook, self).set_data)

    def set_data(self, data, from_hook):
        # print "set_data signal emit from", thread.get_ident()
        self.set_data_signal.emit(data, from_hook)



class OutputQtHook(OutputHook):
    set_data_signal = pyqtSignal(object)
    outdate_signal = pyqtSignal()
    actualize_outputs_signal = pyqtSignal()

    def __init__(self, connector):
        super(OutputQtHook, self).__init__(connector)
        self.set_data_signal.connect(super(OutputQtHook, self).set_data)
        self.outdate_signal.connect(super(OutputQtHook, self).outdate)
        self.actualize_outputs_signal.connect(super(OutputQtHook, self).actualize_outputs)

    def set_data(self, data):
        self.set_data_signal.emit(data)

    def outdate(self):
        self.outdate_signal.emit()

    def actualize_outputs(self):
        self.actualize_outputs_signal.emit()

