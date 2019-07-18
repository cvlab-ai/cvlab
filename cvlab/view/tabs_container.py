from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class TabsContainer(QTabWidget):

    def __init__(self):
        super(TabsContainer, self).__init__()
        self.diagram_manager = None
        self.tab_manager = TabManager(self.tabBar())
        self.setTabShape(QTabWidget.Triangular)
        self.tabBar().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabBar().customContextMenuRequested.connect(self.tab_manager.show_context_menu)


class TabManager:
    def __init__(self, tab_bar):
        self.tab_bar = tab_bar

    def show_context_menu(self, point):
        if not point:
            return
        idx = self.tab_bar.tabAt(point)
        if idx == -1:
            return
        menu = QMenu(self.tab_bar)
        menu.addAction(get_action(self.tab_bar, TabAction.CLOSE, idx))
        menu.addSeparator()
        menu.addAction(get_action(self.tab_bar, TabAction.SAVE, idx))
        menu.addAction(get_action(self.tab_bar, TabAction.SAVE_AS, idx))
        menu.exec_(self.tab_bar.mapToGlobal(point))


def get_action(parent, action_type, idx):
        if action_type == TabAction.CLOSE:
            action = TabAction("Close", parent, idx)
            action.triggered.connect(action.close_diagram)
            return action
        elif action_type == TabAction.SAVE:
            action = TabAction("Save", parent, idx)
            action.triggered.connect(action.save_diagram)
            return action
        elif action_type == TabAction.SAVE_AS:
            action = TabAction("Save as...", parent, idx)
            action.triggered.connect(action.save_diagram_as)
            return action


class TabAction(QAction):

    CLOSE = 0
    SAVE = 1
    SAVE_AS = 2

    def __init__(self, text, parent, tab_idx):
        super(TabAction, self).__init__(text, parent)
        self.tab_idx = tab_idx

    def get_diagram_manager(self):
        return self.parent().parent().diagram_manager

    def close_diagram(self):
        self.get_diagram_manager().close_diagram(self.tab_idx)

    def save_diagram(self):
        self.get_diagram_manager().save_diagram(self.tab_idx)

    def save_diagram_as(self):
        self.get_diagram_manager().save_diagram_as(self.tab_idx)
