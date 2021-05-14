import os
import re
from glob import glob

from PyQt5.QtWidgets import QAction, QMenu

from cvlab.diagram.elements import add_plugin_callback


def get_menu(main_window, title):
    titles = title.split("/")

    menu = main_window.menuBar()

    simple = lambda text: re.sub(r"[^0-9a-zA-Z]+", "", text)

    for title in titles:
        for child in menu.findChildren(QMenu):
            if simple(child.title()) == simple(title):
                menu = child
                break
        else:
            menu = menu.addMenu(title)

    return menu


class OpenExampleAction(QAction):
    def __init__(self, parent, path):
        super().__init__(parent)
        name = os.path.basename(path).replace(".cvlab", "").title()
        self.setText(name)
        self.path = path
        self.triggered.connect(self.open)

    def open(self):
        self.parent().diagram_manager.open_diagram_from_path(self.path)


def add_samples_submenu_callback(main_window, submenu_name, samples_directory):
    menu_title = 'E&xamples/'+submenu_name
    samples = glob(samples_directory + "/*.cvlab")
    samples.sort()

    print("Adding {} sample diagrams to '{}'".format(len(samples), menu_title))

    menu = get_menu(main_window, menu_title)

    for sample in samples:
        menu.addAction(OpenExampleAction(main_window, sample))


def add_samples_submenu(submenu_name, samples_directory):
    callback = lambda main_window: add_samples_submenu_callback(main_window, submenu_name, samples_directory)
    add_plugin_callback(callback)


add_samples_submenu('Basics', os.path.dirname(__file__))
