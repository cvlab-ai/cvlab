import time

from .hooks import *
from .exceptions import *
from .processing_time import ProcessingTimeInfo
from .core_element import CoreElement


class ThreadedElement(CoreElement):
    """Base class for all threaded elements"""

    def __init__(self):
        super(ThreadedElement, self).__init__()
        self.state = self.STATE_UNSET
        self._do_abort = False
        self._do_break = False
        self._notifier = Notifier()
        self._worker = threading.Thread(target=self.work)
        self._worker.daemon = True
        self._worker.name = "Worker for '" + self.__class__.__name__ + "'"
        self._worker.start()
        self.processing_time_info = None

    def recalculate(self, refresh_parameters, refresh_structure, force_break, force_units_recalc=False):
        self.structure_changed |= refresh_structure
        self.parameters_changed |= refresh_parameters
        self._do_break |= force_break
        self._notifier.notify()

    def may_interrupt(self):
        if self._do_break:
            raise InterruptException()

    def delete(self):
        self._do_abort = True
        self._do_break = True
        self._notifier.notify()
        self._worker.join()
        CoreElement.delete(self)

    def work(self):
        self.set_state(self.STATE_UNSET)
        while True:
            self._notifier.wait()
            if self._do_abort: break
            try:
                self.set_state(self.STATE_BUSY)
                start = time.clock()
                self._do_break = False
                self.process()
                self.may_interrupt()
                end = time.clock()
                previous_time_infos = self.get_previous_time_infos()
                self.processing_time_info = ProcessingTimeInfo(start, end, len(self.units), previous_time_infos)
                self.set_state(self.STATE_READY)
            except (InterruptException, ProcessingBreak):
                pass
            except Exception as e:
                self.set_state(self.STATE_ERROR, e)
        self.set_state(self.STATE_UNSET)

    def get_previous_time_infos(self):
        time_infos = []
        for connector in self.inputs.values():
            for inpt in connector.connected_from:
                if hasattr(inpt.parent, "processing_time_info"):    # fixme: ugly hack...
                    time_infos.append(inpt.parent.processing_time_info)
        return time_infos


