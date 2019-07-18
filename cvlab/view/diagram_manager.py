import ntpath
import os
import json
from os.path import isfile

from PyQt5.QtWidgets import *

from ..diagram.diagram import Diagram
from .workarea import ScrolledWorkArea

SETTINGS_LAST_OPEN_SECTION = 'diagram'
SETTINGS_LAST_OPEN_OPTION = 'last_open'

last_file_name = ""


class DiagramManager:

    FILE_TYPES = "CV Lab diagram (*.cvlab);;JSON (*.json);;All Files (*)"

    def __init__(self, tabs_container, style_manager):
        self.tabs_container = tabs_container
        self.style_manager = style_manager

    def count(self):
        return self.tabs_container.count()

    def get_open_file_name(self):
        global last_file_name
        main_window = self.tabs_container.parent()
        path, _ = QFileDialog.getOpenFileName(main_window, "Open diagram file", last_file_name, self.FILE_TYPES)
        if path:
            last_file_name = path
        return path

    def get_save_file_name(self):
        global last_file_name
        main_window = self.tabs_container.parent()
        path, _ = QFileDialog.getSaveFileName(main_window, "Save diagram as", last_file_name, self.FILE_TYPES)
        if path:
            last_file_name = path
        return path

    def open_diagram_browse(self):
        # TODO: block the UI for the time of loading?
        diagram_path = self.get_open_file_name()
        self.open_diagram_from_path(diagram_path)

    def open_diagram_from_path(self, path):
        # TODO: block the UI for the time of loading?
        if not path or not isfile(path):
            return
        with open(path, 'r') as fp:
            try:
                encoded = fp.read()
                scrolled_wa = ScrolledWorkArea(Diagram(), self.style_manager)
                base_path = os.path.abspath(path + "/../").replace("\\","/")
                scrolled_wa.load_diagram_from_json(encoded, base_path)
                full_path = os.path.abspath(str(path))
                self.open_diagram(scrolled_wa, full_path)
                scrolled_wa.workarea.actualize_style()
            except Exception as e:
                print("Error: could not load diagram {} - {}".format(path, e))

    def open_from_settings(self, config):
        try:
            files = config.get(SETTINGS_LAST_OPEN_SECTION, SETTINGS_LAST_OPEN_OPTION)
            if files is None:
                return
            paths = json.loads(files)
            for path in paths:
                self.open_diagram_from_path(path)
        except ValueError:
            print("Error loading settings file: wrong entry in the field: [diagram] last_open.")

    def open_diagram(self, scrolled_wa=None, full_path=None):
        # TODO: block the UI for the time of loading?
        to_open = scrolled_wa
        name = None
        if scrolled_wa is None:
            to_open = ScrolledWorkArea(Diagram(), self.style_manager)
        if full_path is None:
            name = "new*"
        else:
            name = get_file_name_from_path(full_path)
            if name.endswith(".cvlab"): name = name[:-6]
        idx = self.tabs_container.addTab(to_open, name)
        if full_path is not None:
            self.tabs_container.setTabToolTip(idx, full_path)
        self.tabs_container.setCurrentIndex(idx)
        for e in to_open.diagram.elements:  # need to do this here, as we want to process repaint events
            e.state_changed.emit()

    def save_to_settings(self, settings):
        count = self.tabs_container.count()
        paths = [str(self.tabs_container.tabToolTip(i)) for i in range(count)]
        entry = json.dumps(paths)
        settings.set(SETTINGS_LAST_OPEN_SECTION, SETTINGS_LAST_OPEN_OPTION, entry)

    def save_diagram_as(self, tab_idx=None):
        path = self.get_save_file_name()
        if path:
            tab_idx = tab_idx if tab_idx is not None else self.tabs_container.currentIndex()
            scrolled_workarea = self.tabs_container.widget(tab_idx)
            diagram = scrolled_workarea.workarea.diagram
            # TODO: block the UI for the time of saving
            save_diagram_to_file(diagram, path)
            diagram_name = get_file_name_from_path(str(path))
            self.tabs_container.setTabText(tab_idx, diagram_name)
            self.tabs_container.setTabToolTip(tab_idx, str(path))

    def save_diagram(self, tab_idx=None):
        tab_idx = tab_idx or self.tabs_container.currentIndex()
        path = self.tabs_container.tabToolTip(tab_idx)
        if not path or path == "":
            self.save_diagram_as(tab_idx)
            return
        diagram = self.tabs_container.widget(tab_idx).diagram
        if diagram:
            save_diagram_to_file(diagram, path)

    def close_diagram(self, tab_idx=None):
        # TODO: block UI?
        if tab_idx is None:
            tab_idx = self.tabs_container.currentIndex()
        scrolled_workarea = self.tabs_container.widget(tab_idx)
        self.tabs_container.removeTab(tab_idx)
        if scrolled_workarea:
            scrolled_workarea.diagram.clear()
            scrolled_workarea.deleteLater()

    def close_all_diagrams(self):
        while self.tabs_container.count():
            self.close_diagram()

    def current_workarea(self):
        tab_idx = self.tabs_container.currentIndex()
        workarea = self.tabs_container.widget(tab_idx)
        return workarea


def get_file_name_from_path(path):
    return os.path.basename(str(path))


def save_diagram_to_file(diagram, path):
    base_path = os.path.abspath(path + "/../").replace("\\","/")
    encoded = diagram.save_to_json(base_path)
    with open(path, 'w') as fp:
        fp.write(encoded)
