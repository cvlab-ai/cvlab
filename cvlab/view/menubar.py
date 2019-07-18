from distutils.util import strtobool

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from . import config


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

        view_menu = self.addMenu('&View')
        view_menu.addMenu(ColorThemeMenu(view_menu, main_window))
        view_menu.addAction(HighQualityAction(view_menu, main_window))
        view_menu.addAction(LivePreviewsAction(view_menu, main_window))
        view_menu.addAction(PreviewOnTopAction(view_menu, main_window))
        view_menu.addAction(ResetZoomAction(view_menu, main_window))
        view_menu.addAction(ExperimentalElementsAction(view_menu, main_window))

        help_menu = self.addMenu("&Help")
        help_menu.addAction(AboutAction(help_menu, main_window))


class Action(QAction):
    def __init__(self, name, parent, main_window):
        super(Action, self).__init__(name, parent)
        self.main_window = main_window
        self.settings = config.ConfigWrapper.get_settings()


class NewAction(Action):
    def __init__(self, parent, main_window):
        super(NewAction, self).__init__('&New', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_N))
        self.triggered.connect(self.open_new)

    @pyqtSlot()
    def open_new(self):
        self.main_window.diagram_manager.open_diagram()


class OpenAction(Action):
    def __init__(self, parent, main_window):
        super(OpenAction, self).__init__('&Open...', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_O))
        self.triggered.connect(self.open)

    @pyqtSlot()
    def open(self):
        self.main_window.diagram_manager.open_diagram_browse()


class SaveAsAction(Action):
    def __init__(self, parent, main_window):
        super(SaveAsAction, self).__init__('&Save as...', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.SHIFT + Qt.Key_S))
        self.triggered.connect(self.save_as)

    @pyqtSlot()
    def save_as(self):
        self.main_window.diagram_manager.save_diagram_as()


class SaveAction(Action):
    def __init__(self, parent, main_window):
        super(SaveAction, self).__init__('Save', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_S))
        self.triggered.connect(self.save)

    @pyqtSlot()
    def save(self):
        self.main_window.diagram_manager.save_diagram()


class CloseDiagramAction(Action):
    def __init__(self, parent, main_window):
        super(CloseDiagramAction, self).__init__('Close diagram', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_W))
        self.triggered.connect(self.close)

    @pyqtSlot()
    def close(self):
        self.main_window.diagram_manager.close_diagram()


class CloseAppAction(Action):
    def __init__(self, parent, main_window):
        super(CloseAppAction, self).__init__('E&xit', parent, main_window)
        self.setShortcut(QKeySequence(Qt.ALT + Qt.Key_F4))
        self.triggered.connect(self.exit)

    @pyqtSlot()
    def exit(self):
        self.main_window.diagram_manager.close_all_diagrams()
        self.main_window.application.quit()


class ColorThemeMenu(QMenu):
    def __init__(self, parent, main_window):
        super(ColorThemeMenu, self).__init__("&Set color theme", parent)
        self.main_window = main_window
        self.fill_styles()

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
        self.triggered.connect(self.switch)

    @pyqtSlot()
    def switch(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.VIEW_SECTION, config.PREVIEW_ON_TOP_OPTION, self.value)


class ResetZoomAction(Action):
    def __init__(self, parent, main_window):
        super(ResetZoomAction, self).__init__('Reset zoom', parent, main_window)
        self.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_0))
        self.triggered.connect(self.reset_zoom)

    @pyqtSlot()
    def reset_zoom(self):
        workarea = self.main_window.diagram_manager.current_workarea()
        if workarea:
            workarea.workarea.zoom(1.0)


class ExperimentalElementsAction(Action):
    def __init__(self, parent, main_window):
        super(ExperimentalElementsAction, self).__init__('&Experimental elements', parent, main_window)
        self.setCheckable(True)
        self.value = bool(strtobool(self.settings.get_with_default(config.ELEMENTS_SECTION, config.EXPERIMENTAL_ELEMENTS)))
        self.setChecked(self.value)
        self.triggered.connect(self.toggle)

    @pyqtSlot()
    def toggle(self):
        self.value = not self.value
        self.setChecked(self.value)
        self.settings.set(config.ELEMENTS_SECTION, config.EXPERIMENTAL_ELEMENTS, self.value)
        QMessageBox.information(self.main_window, "Information", "You must restart CV Lab to enable/disable experimental elements.")


class AboutAction(Action):
    message = """\
<h1>CV Lab - Computer Vision Laboratory</h1>
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
        super().__init__("About", parent, main_window)
        self.triggered.connect(self.execute)

    @pyqtSlot()
    def execute(self):
        QMessageBox.about(self.main_window, self.main_window.windowTitle(), self.message)
