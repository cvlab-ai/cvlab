import os
from glob import glob

from PyQt5.QtWidgets import QAction, QMenu

from cvlab.diagram.elements import add_plugin_callback


def get_menu(main_window, title):
    titles = title.split("/")

    menu = main_window.menuBar()

    for title in titles:
        for child in menu.findChildren(QMenu):
            if child.title() == title:
                menu = child
                break
        else:
            menu = menu.addMenu(title)

    return menu


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

    print("Adding {} sample diagrams to 'Examples' menu".format(len(samples)))

    menu = get_menu(main_window, 'Examples/Basics')

    for sample in samples:
        menu.addAction(OpenExampleAction(main_window, sample))


add_plugin_callback(add_samples)

