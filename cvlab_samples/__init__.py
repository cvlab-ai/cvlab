import os
from glob import glob

from PyQt5.QtWidgets import QAction

from cvlab.diagram.elements import add_plugin_callback


class OpenExampleAction(QAction):
    def __init__(self, parent, path):
        super().__init__(parent)
        name = os.path.basename(path).replace(".cvlab","").title()
        self.setText(name)
        self.path = path
        self.triggered.connect(self.open)

    def open(self):
        self.parent().diagram_manager.open_diagram_from_path(self.path)


def add_samples(main_window):
    samples = glob(os.path.dirname(__file__) + "/*.cvlab")
    samples.sort()

    print("Adding {} sample diagrams to main menu".format(len(samples)))

    menu = main_window.menuBar()

    samples_menu = menu.addMenu('E&xamples')

    for sample in samples:
        samples_menu.addAction(OpenExampleAction(main_window, sample))


add_plugin_callback(add_samples)

