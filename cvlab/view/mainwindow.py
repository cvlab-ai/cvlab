import os

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .. import CVLAB_DIR
from ..diagram.elements import plugin_callbacks
from ..core.update import Updater, parse_version
from .diagram_manager import DiagramManager
from .menubar import MenuBar
from .tabs_container import TabsContainer
from .toolbox import Toolbox
from .config import ConfigWrapper, UPDATES_SECTION, UPDATE_DONT_REMIND_VERSION
from .styles import StyleManager


LOAD_LAST_DIAGRAMS = True
LOAD_DEFAULT_DIAGRAM = True
DEFAULT_DIAGRAM_PATH = "default.cvlab"
ICON_PATH = CVLAB_DIR + "/images/icon.png"
FONT_PATH = CVLAB_DIR + "/styles/fonts/Carlito-Regular.ttf"


class MainWindow(QMainWindow):
    update_available = pyqtSignal(bool, str)

    def __init__(self, application):
        super(MainWindow, self).__init__()

        assert isinstance(application, QApplication)

        self.application = application
        self.settings = ConfigWrapper.get_settings()
        self.style_name = None
        self.resize(1000, 600)
        self.setWindowTitle('CV Lab')

        icon = QIcon(ICON_PATH)
        application.setWindowIcon(icon)
        QFontDatabase.addApplicationFont(FONT_PATH)
        self.style_manager = StyleManager(self)

        self.toolbox = Toolbox()
        self.tabs_container = TabsContainer()
        self.diagram_manager = DiagramManager(self.tabs_container, self.style_manager)
        self.tabs_container.diagram_manager = self.diagram_manager
        MenuBar(self)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbox, 0, 0)
        layout.addWidget(self.tabs_container, 0, 1)

        layout.setColumnMinimumWidth(0, 220)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 6)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.showMaximized()
        self.raise_()   # bring to front on a Mac

        QTimer.singleShot(100, self.process_plugins_callbacks)
        QTimer.singleShot(200, self.load_diagrams)

        # Prevent automatic focus on toolbox search field
        self.toolbox.filter_input.clearFocus()

        # check for updates
        self.updater = Updater()
        self.update_available.connect(self.show_update_info)
        self.updater.check_async(self.update_available.emit)

    def load_diagrams(self):
        if LOAD_LAST_DIAGRAMS:
            self.diagram_manager.open_from_settings(self.settings)
            if self.diagram_manager.count() == 0:
                if LOAD_LAST_DIAGRAMS:
                    self.diagram_manager.open_diagram_from_path(DEFAULT_DIAGRAM_PATH)
                else:
                    self.diagram_manager.open_diagram()
        elif LOAD_DEFAULT_DIAGRAM:
            self.diagram_manager.open_diagram_from_path(DEFAULT_DIAGRAM_PATH)
        else:
            self.diagram_manager.open_diagram()

    def process_plugins_callbacks(self):
        for callback in plugin_callbacks:
            callback(self)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.toolbox.filter_input.setFocus()
            self.toolbox.filter_input.selectAll()
        if (key == Qt.Key_Tab and (event.modifiers() & Qt.ControlModifier)) or \
                (key == Qt.Key_Backtab and (event.modifiers() & Qt.ControlModifier)):
            self.tabs_container.keyPressEvent(event)

    def closeEvent(self, event):
        QMainWindow.closeEvent(self, event)
        self.diagram_manager.save_to_settings(self.settings)
        self.diagram_manager.close_all_diagrams()
        # Close any remaining windows:
        for window in [x for x in QApplication.topLevelWidgets() if x.isWindow() and x != self and x.parent() is None and x.isVisible()]:
            window.close()

    def show_update_info(self, can_update, newest_version):
        from ..version import __version__
        newest_version = str(newest_version)
        if can_update:
            act_version = parse_version(__version__)
            newest_version = parse_version(newest_version)
            dont_remind_version = parse_version(self.settings.get_with_default(UPDATES_SECTION, UPDATE_DONT_REMIND_VERSION))

            command = self.updater.update_command()
            self.application.clipboard().setText(command)

            message = """\
There is a new version of CV Lab.

Your version: {__version__}
Newest version: {newest_version}

To update you may run:
{command}

(command copied to clipboard)
""".format(**locals())

            if newest_version <= dont_remind_version:
                print("--- UPDATE INFORMATION ---")
                print(message)
            else:
                mb = QMessageBox(QMessageBox.Information,"Update information", message)
                mb.addButton(QPushButton("OK"), QMessageBox.AcceptRole)
                ignore = QPushButton("Ignore this update")
                mb.addButton(ignore, QMessageBox.DestructiveRole)
                mb.exec_()
                if mb.clickedButton() == ignore:
                    self.settings.set(UPDATES_SECTION, UPDATE_DONT_REMIND_VERSION, str(newest_version))
        else:
            print("You're using newest version of CV Lab ({__version__}).".format(**locals()))

