from distutils.util import strtobool

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from . import config
from .styles import StyleManager


class MenuBar(QMenuBar):
    def __init__(self, parent):
        super(MenuBar, self).__init__(parent)
        main_window = parent
        main_window.setMenuBar(self)

        file_menu = self.addMenu('&File')
        file_menu.addAction(NewAction(file_menu, main_window))
        file_menu.addAction(OpenAction(file_menu, main_window))
        file_menu.addAction(SaveAction(file_menu, main_window))
        file_menu.addAction(SaveAsAction(file_menu, main_window))
        file_menu.addAction(CloseDiagramAction(file_menu, main_window))
        file_menu.addSeparator()
        file_menu.addAction(CloseAppAction(file_menu, main_window))

        edit_menu = self.addMenu("&Edit")
        edit_menu.addAction(CenterElementsAction(edit_menu, main_window))
        edit_menu.addAction(SelectAllAction(edit_menu, main_window))
        edit_menu.addAction(DeleteSelectedAction(edit_menu, main_window))

        view_menu = self.addMenu('&View')
        view_menu.addMenu(ColorThemeMenu(view_menu, main_window))
        view_menu.addAction(HighQualityAction(view_menu, main_window))
        view_menu.addAction(LivePreviewsAction(view_menu, main_window))
        view_menu.addAction(PreviewOnTopAction(view_menu, main_window))
        view_menu.addAction(ResetViewAction(view_menu, main_window))
        view_menu.addAction(ExperimentalElementsAction(view_menu, main_window))

        help_menu = self.addMenu("&Help")
        help_menu.addAction(AboutAction(help_menu, main_window))


class Action(QAction):
    def __init__(self, name, parent, main_window):
        super(Action, self).__init__(name, parent)
        self.main_window = main_window
        self.settings = config.ConfigWrapper.get_settings()


class SimpleAction(Action):
    def __init__(self, name, parent, main_window, action=None, shortcut=None, icon=None):
        super().__init__(name, parent, main_window)

        if shortcut:
            if not isinstance(shortcut, QKeySequence):
                shortcut = QKeySequence(shortcut)
            self.setShortcut(shortcut)

        if icon:
            StyleManager.icons.set_icon(self, icon)

        self.action = action
        self.triggered.connect(self.do_action)

    @pyqtSlot()
    def do_action(self):
        self.action(self.main_window)


class WorkareaAction(SimpleAction):
    def __init__(self, name, parent, main_window, action, shortcut=None, icon=None):
        super().__init__(name, parent, main_window, action, shortcut=shortcut, icon=icon)

    @pyqtSlot()
    def do_action(self):
        workarea = self.main_window.diagram_manager.current_workarea()
        if workarea: self.action(workarea)


class NewAction(SimpleAction):
    def __init__(self, parent, main_window):
        super().__init__('&New', parent, main_window, shortcut=Qt.CTRL + Qt.Key_N, icon="doc 6")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.open_diagram()


class OpenAction(SimpleAction):
    def __init__(self, parent, main_window):
        super(OpenAction, self).__init__('&Open...', parent, main_window, shortcut=Qt.CTRL + Qt.Key_O, icon="folder")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.open_diagram_browse()


class SaveAsAction(SimpleAction):
    def __init__(self, parent, main_window):
        super().__init__('&Save as...', parent, main_window, None, Qt.CTRL + Qt.SHIFT + Qt.Key_S, "floppy disk")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.save_diagram_as()


class SaveAction(SimpleAction):
    def __init__(self, parent, main_window):
        super(SaveAction, self).__init__('Save', parent, main_window, None, Qt.CTRL + Qt.Key_S, "floppy disk")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.save_diagram()


class CloseDiagramAction(SimpleAction):
    def __init__(self, parent, main_window):
        super(CloseDiagramAction, self).__init__('Close diagram', parent, main_window, None, Qt.CTRL + Qt.Key_W, "doc 8")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.close_diagram()


class CloseAppAction(SimpleAction):
    def __init__(self, parent, main_window):
        super(CloseAppAction, self).__init__('E&xit', parent, main_window, None, Qt.ALT + Qt.Key_F4, "power button")

    @pyqtSlot()
    def do_action(self):
        self.main_window.diagram_manager.close_all_diagrams()
        self.main_window.application.quit()


class ColorThemeMenu(QMenu):
    def __init__(self, parent, main_window):
        super(ColorThemeMenu, self).__init__("&Set color theme", parent)
        self.main_window = main_window
        self.fill_styles()
        StyleManager.icons.set_icon(self, "settings 8")

    def fill_styles(self):
        styles = self.main_window.style_manager.get_available_stylesheets()
        current = self.main_window.settings.get_with_default(config.VIEW_SECTION, config.STYLE)
        for style in styles:
            action = self.addAction(style)
            action.setCheckable(True)
            if style == current:
                action.setChecked(True)
            else:
                action.setChecked(False)
            action.triggered.connect(self.style_chosen)

    @pyqtSlot()
    def style_chosen(self):
        chosen_style = self.sender()
        self.apply_style(chosen_style.text())

    def apply_style(self, style):
        for action in self.actions():
            if action.text() == style:
                action.setChecked(True)
            else:
                action.setChecked(False)
        self.main_window.style_manager.apply_stylesheet(style)


class HighQualityAction(Action):
    def __init__(self, parent, main_window):
        super(HighQualityAction, self).__init__('&High quality previews', parent, main_window)
        self.setCheckable(True)
        self.value = bool(strtobool(self.settings.get_with_default(config.VIEW_SECTION, config.VIEW_HQ_OPTION)))
        self.setChecked(self.value)
        StyleManager.icons.set_icon(self, "pic")
        self.triggered.connect(self.toggle_quality)

    @pyqtSlot()
    def toggle_quality(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.VIEW_SECTION, config.VIEW_HQ_OPTION, self.value)


class LivePreviewsAction(Action):
    def __init__(self, parent, main_window):
        super(LivePreviewsAction, self).__init__('&Show live results', parent, main_window)
        self.setCheckable(True)
        self.value = bool(strtobool(self.settings.get_with_default(config.VIEW_SECTION,
                                                                   config.LIVE_IMAGE_PREVIEW_OPTION)))
        self.setChecked(self.value)
        StyleManager.icons.set_icon(self, "play")
        self.triggered.connect(self.switch)

    @pyqtSlot()
    def switch(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.VIEW_SECTION, config.LIVE_IMAGE_PREVIEW_OPTION, self.value)


class PreviewOnTopAction(Action):
    def __init__(self, parent, main_window):
        super(PreviewOnTopAction, self).__init__('&Preview always on top', parent, main_window)
        self.setCheckable(True)
        self.value = bool(strtobool(self.settings.get_with_default(config.VIEW_SECTION, config.PREVIEW_ON_TOP_OPTION)))
        self.setChecked(self.value)
        StyleManager.icons.set_icon(self, "pin")
        self.triggered.connect(self.switch)

    @pyqtSlot()
    def switch(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.VIEW_SECTION, config.PREVIEW_ON_TOP_OPTION, self.value)


class CenterElementsAction(WorkareaAction):
    def __init__(self, parent, main_window):
        super().__init__('Center elements', parent, main_window, self.center_elements, icon="arrow 13")

    def center_elements(self, workarea):
        workarea.workarea.center_elements()
        workarea.scroll_to_upperleft()


class SelectAllAction(WorkareaAction):
    def __init__(self, parent, main_window):
        super().__init__('Select all', parent, main_window, self.select_all, Qt.CTRL + Qt.Key_A, "select")

    def select_all(self, workarea):
        workarea.workarea.selection_manager.select_all_elements()


class DeleteSelectedAction(WorkareaAction):
    def __init__(self, parent, main_window):
        super().__init__('Delete selected', parent, main_window, self.delete_selected, Qt.Key_Delete, "delete")

    def delete_selected(self, workarea):
        workarea.workarea.selection_manager.delete_selected()


class ResetViewAction(WorkareaAction):
    def __init__(self, parent, main_window):
        super(ResetViewAction, self).__init__('Reset view', parent, main_window, self.reset_view, Qt.CTRL + Qt.Key_0, "search")

    def reset_view(self, workarea):
        workarea.workarea.zoom(1.0)
        workarea.scroll_to_upperleft()


class ExperimentalElementsAction(Action):
    def __init__(self, parent, main_window):
        super(ExperimentalElementsAction, self).__init__('&Experimental elements', parent, main_window)
        self.setCheckable(True)
        self.value = bool(strtobool(self.settings.get_with_default(config.ELEMENTS_SECTION, config.EXPERIMENTAL_ELEMENTS)))
        self.setChecked(self.value)
        StyleManager.icons.set_icon(self, "veil")
        self.triggered.connect(self.toggle)

    @pyqtSlot()
    def toggle(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.ELEMENTS_SECTION, config.EXPERIMENTAL_ELEMENTS, self.value)
        QMessageBox.information(self.main_window, "Information", "You must restart CV Lab to enable/disable experimental elements.")


class AboutAction(SimpleAction):
    message = """\
<h1>CV Lab</h1>
<h2>Computer Vision Laboratory</h2>
<h3>A rapid prototyping tool for computer vision algorithms</h3>

<p>
Homepage on GitHub: <a href="https://github.com/cvlab-ai/cvlab">https://github.com/cvlab-ai/cvlab</a><br/>  
PyPI package: <a href="https://pypi.python.org/pypi/cvlab">https://pypi.python.org/pypi/cvlab</a>
</p>

<p>
<b>Authors:</b> Adam Brzeski, Jan Cychnerski<br/>
<b>License:</b> AGPL-3.0+<br/><br/>
<i>Copyright 2013-2019</i>
</p>"""

    def __init__(self, parent, main_window):
        super().__init__("About", parent, main_window, icon="info")

    @pyqtSlot()
    def do_action(self):
        QMessageBox.about(self.main_window, self.main_window.windowTitle(), self.message)
