import os
from weakref import WeakKeyDictionary

from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QRect
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainter
from PyQt5.QtWidgets import *

from .. import CVLAB_DIR
from .wires import WireStyle
from . import config


AUTO_REFRESH_STYLESHEET = 0     # Use 1 when working on stylesheets (warning: option causes some artifacts, like flickering
                                # menu bar and disappearing stylesheet when opening File Dialog)


class StyleManager(QObject):

    style_changed = pyqtSignal()

    is_highdpi = QApplication.desktop().screenGeometry().width() > 2500
    icons = None

    def __init__(self, main_window):
        super(StyleManager, self).__init__()
        self.styles_dir = CVLAB_DIR + "/styles/themes"
        self.main_window = main_window
        self.wire_style = None
        self.style_name = "default"
        self.stylesheet = ""
        StyleManager.icons = Icons(self)
        self.apply_default_stylesheet()

        # Only for developing stylesheets
        if AUTO_REFRESH_STYLESHEET:
            self.timer = QTimer()
            self.timer.timeout.connect(self.apply_default_stylesheet)
            self.timer.start(500)

    def apply_stylesheet(self, style_name):
        style_name = str(style_name)
        selected_style_path = os.path.join(self.styles_dir, style_name, style_name + ".stylesheet")
        common_style_path = os.path.normpath(self.styles_dir + "/common.stylesheet")
        highdpi_style_path = os.path.normpath(self.styles_dir + "/highdpi.stylesheet")
        stylesheet = ""
        try:
            # Common
            with open(common_style_path, "r") as common:
                stylesheet += common.read()

            # Highdpi
            if self.is_highdpi:
                with open(highdpi_style_path, "r") as highdpi:
                    stylesheet += highdpi.read()

            # Selected
            with open(selected_style_path, "r") as selected:
                stylesheet += selected.read()

        except IOError:
            pass

        cvlab_dir = os.path.abspath(__file__ + "/../../").replace("\\","/")
        stylesheet = stylesheet.replace("$CVLAB_DIR", cvlab_dir)

        # Switch background for highdpi
        if self.is_highdpi:
            stylesheet = stylesheet.replace("background.png", "background-highdpi.png")

        self.style_name = style_name
        self.stylesheet = stylesheet
        self.main_window.setStyleSheet(stylesheet)
        refresh_style_recursive(self.main_window)

        # Load wires style
        self.wire_style = WireStyle(stylesheet)

        # Notify
        self.style_changed.emit()

        # Update settings
        current_style = self.main_window.settings.get_with_default(config.VIEW_SECTION, config.STYLE)
        if style_name == config.DEFAULTS[config.VIEW_SECTION][config.STYLE]:
            if self.main_window.settings.has_option(config.VIEW_SECTION, config.STYLE):
                self.main_window.settings.remove_option(config.VIEW_SECTION, config.STYLE)
        elif style_name != current_style:
            self.main_window.settings.set(config.VIEW_SECTION, config.STYLE, style_name)

    def apply_default_stylesheet(self):
        current_style = self.main_window.settings.get_with_default(config.VIEW_SECTION, config.STYLE)
        if current_style is not None:
            self.apply_stylesheet(current_style)

    def get_available_stylesheets(self):
        # Todo: fill this basing on 'themes' directory content
        return ["default", "dark"]


def refresh_style_recursive(base_widget):
    widgets = [base_widget]
    while widgets:
        widget = widgets.pop()
        base_widget.style().polish(widget)
        for child in widget.children():
            if isinstance(child, QWidget):
                widgets.append(child)


class Icons:
    def __init__(self, style_manager):
        self.style_manager = style_manager
        self.buffer = {}
        self.objects = WeakKeyDictionary()
        self.style_manager.style_changed.connect(self.actualize_style)

    def actualize_style(self):
        for name, old_icon in self.buffer.items():
            new_icon = self.get_no_buffer(name)
            old_icon.swap(new_icon)

        for object, name in self.objects.items():
            new_icon = self.get(name)
            object.setIcon(new_icon)

    def get_no_buffer(self, name):
        cvlab_dir = CVLAB_DIR
        style = self.style_manager.style_name
        path = "{cvlab_dir}/styles/themes/{style}/icons/{name}.png".format(**locals())

        if os.path.isfile(path):
            icon = QIcon()

            pixmap = QPixmap(path)
            icon.addPixmap(pixmap.copy(), QIcon.Normal, QIcon.Off)

            painter = QPainter(pixmap)
            rect = QRect(pixmap.width()*2//3,pixmap.height()*2//3, pixmap.width()//3, pixmap.height()//3)
            painter.fillRect(rect, QColor.fromRgb(64, 115, 178))
            painter.end()
            icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)

        else:
            print("WARNING: Cannot find icon:", name)
            icon = QIcon()

        return icon

    def get(self, name):
        if name not in self.buffer:
            self.buffer[name] = self.get_no_buffer(name)
        return self.buffer[name]

    def set_icon(self, object, icon):
        if isinstance(icon, str):
            self.objects[object] = icon  # object -> icon name
            icon = self.get(icon)
        object.setIcon(icon)
