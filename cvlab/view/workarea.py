import os
import re
from datetime import datetime, timedelta

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt, QTimer, QPoint
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from ..diagram.element import Element
from .elements import GuiElement
from .mimedata import Mime
from .styles import refresh_style_recursive
from .wires import WiresForeground, NO_FOREGROUND_WIRES, WiresBackground, WireTools


class ScrolledWorkArea(QScrollArea):
    def __init__(self, diagram, style_manager):
        super(ScrolledWorkArea, self).__init__()
        self.setObjectName("ScrolledWorkArea")
        self.workarea = WorkArea(diagram, style_manager)
        self.workarea.setParent(self)
        self.diagram = diagram
        self.setWidget(self.workarea)
        diagram.set_painter(self)
        self.mouse_press_pos = None
        QTimer.singleShot(50, self.scroll_to_absolute_center)

    def load_diagram_from_json(self, ascii_data, base_path):
        self.diagram.load_from_json(ascii_data, base_path)
        QTimer.singleShot(100, self.scroll_to_upperleft)

    def element_z_index(self, element):
        return self.workarea.children().index(element)

    def mousePressEvent(self, e):
        self.mouse_press_pos = e.pos() + QtCore.QPoint(self.horizontalScrollBar().value(), self.verticalScrollBar().value())
        e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MiddleButton and self.mouse_press_pos is not None:
            pos = self.mouse_press_pos - e.pos()
            self.horizontalScrollBar().setValue(pos.x())
            self.verticalScrollBar().setValue(pos.y())
            e.accept()

    def wheelEvent(self, event):
        assert isinstance(event, QWheelEvent)
        if event.modifiers() in (Qt.ControlModifier, Qt.MetaModifier):

            zoom_before = self.diagram.zoom_level
            origin = (event.x() + self.horizontalScrollBar().value(),
                      event.y() + self.verticalScrollBar().value())

            if event.angleDelta().y() < 0:
                self.workarea.zoom(index=-1)
            else:
                self.workarea.zoom(index=1)

            zoom_after = self.diagram.zoom_level
            zoom_factor = zoom_after/zoom_before

            scroll_x = origin[0] * zoom_factor - event.x()
            scroll_y = origin[1] * zoom_factor - event.y()

            self.horizontalScrollBar().setValue(int(scroll_x))
            self.verticalScrollBar().setValue(int(scroll_y))

            event.accept()
        else:
            super(ScrolledWorkArea, self).wheelEvent(event)

    def scroll_to_center(self):
        if not self.diagram.elements:
            self.scroll_to_absolute_center()
            return
        pos, n = QtCore.QPoint(0, 0), 0
        for e in self.diagram.elements:
            n += 1
            pos += e.pos() + QtCore.QPoint(e.width()//2, e.height()//2)
        pos /= n
        pos -= QtCore.QPoint(self.width()//2, self.height()//2)
        self.horizontalScrollBar().setValue(pos.x())
        self.verticalScrollBar().setValue(pos.y())

    def scroll_to_upperleft(self):
        if not self.diagram.elements:
            self.scroll_to_absolute_center()
            return
        left = min(e.x() for e in self.diagram.elements) - 20
        top = min(e.y() for e in self.diagram.elements) - 20
        self.horizontalScrollBar().setValue(left)
        self.verticalScrollBar().setValue(top)

    def scroll_to_absolute_center(self):
        left = self.workarea.width()//2 - (self.width() - self.verticalScrollBar().width())//2
        top = self.workarea.height()//2 - (self.height() - self.horizontalScrollBar().height())//2
        self.horizontalScrollBar().setValue(left)
        self.verticalScrollBar().setValue(top)


class UserActions(QObject):
    element_relocated = pyqtSignal(Element)
    cursor_line_started = pyqtSignal()
    cursor_line_moved = pyqtSignal(tuple)
    cursor_line_dropped = pyqtSignal()


class WorkArea(QWidget):
    zoom_levels = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]
    DEFAULT_POSITION_GRID = 10
    DEFAULT_SIZE = 100000

    help = """\
Diagram work area

Left click drag & drop - select elements
Middle click drag & drop - move around the view
Ctrl + mouse wheel - zoom in/out"""

    def __init__(self, diagram, style_manager):
        super(WorkArea, self).__init__()
        self.setObjectName("WorkArea")
        self.setAcceptDrops(True)
        self.setFixedSize(self.DEFAULT_SIZE, self.DEFAULT_SIZE)
        self.diagram = diagram
        self.user_actions = UserActions()
        self.style_manager = style_manager
        self.wire_tools = WireTools(style_manager)
        self.wires_in_foreground = WiresForeground(self, self.user_actions, self.wire_tools)
        self.wires_in_background = WiresBackground(self, self.user_actions, self.wire_tools)
        self.selection_manager = SelectionManager(self)
        self.connectors_map = {}
        self.diagram.element_added.connect(self.on_element_added)
        self.diagram.element_deleted.connect(self.on_element_deleted)
        self.element_move_start = None
        self.last_auto_scroll_time = datetime.now()
        self.style_manager.style_changed.connect(self.actualize_style)
        self.setToolTip(self.help)

    @property
    def position_grid(self):
        return int(round(self.DEFAULT_POSITION_GRID * self.diagram.zoom_level))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selection_manager.on_workarea_mouse_left_pressed(e)
        if e.button() in (Qt.LeftButton, Qt.RightButton):
            self.wires_in_background.mousePressEvent(e)
        e.ignore()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.selection_manager.on_workarea_mouse_left_move(e)
        if e.buttons() == Qt.MiddleButton:
            e.ignore()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selection_manager.on_workarea_mouse_left_released(e)

    def on_element_moused_pressed(self, element, event):
        self.element_move_start = event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            self.selection_manager.on_element_moused_left_pressed(element, event)
        if event.button() == QtCore.Qt.RightButton:
            self.selection_manager.on_element_right_pressed(element, event)
        self.wires_in_background.unselect_wires()

    def on_element_moused_left_released(self, element, event):
        if (event.pos() == self.element_move_start):
            self.selection_manager.on_element_left_released_inplace(element, event)
        self.element_move_start = None

    def on_element_moused_left_moved(self, element, event):
        if self.element_move_start is not None:
            for e in self.selection_manager.selected_elements:
                pos = e.pos() - self.element_move_start + event.pos()
                x, y = self.nearest_grid_point(pos.x(), pos.y())
                e.move(x, y)
                e.element_relocated.emit(e)
            if datetime.now() - self.last_auto_scroll_time > timedelta(milliseconds=150):
                cursor_pos = element.pos() + event.pos()
                self.parent().parent().ensureVisible(cursor_pos.x(), cursor_pos.y(), 50, 50)  # pozwala na scrollowanie
                self.last_auto_scroll_time = datetime.now()

    @pyqtSlot(Element, tuple)
    def on_element_added(self, element, position):
        element.setParent(self)
        element.move(position[0], position[1])
        element.setVisible(True)
        element.setFocus()
        element.set_workarea(self)
        element.element_relocated.connect(self.user_actions.element_relocated)
        self.adjustSize()

        if os.name == "posix":
            refresh_style_recursive(element)

        element.actualize_style()

        if not NO_FOREGROUND_WIRES:
            self.wires_in_foreground.raise_()

        self.connectors_map.update(element.input_connectors)
        self.connectors_map.update(element.output_connectors)

    @pyqtSlot(Element)
    def on_element_deleted(self, element):
        # Todo: potential memory leak - see: http://stackoverflow.com/questions/5899826/pyqt-how-to-remove-a-widget
        for connector in list(element.outputs.values()):
            if connector.preview_only:  # preview only - no InOutConnecotr is created - nothing to delete
                continue
            del self.connectors_map[connector]
        for connector in list(element.inputs.values()):
            del self.connectors_map[connector]
        element.setParent(None)
        element.deleteLater()

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        mime = e.mimeData().text()
        if mime == Mime.NEW_ELEMENT and e.source().last_spawned_element != 0:
            new_elem = e.source().last_spawned_element
            position = (e.pos().x() - new_elem.width()//2,
                        e.pos().y() - new_elem.height()//2)
            self.diagram.add_element(new_elem, position)
        elif mime in [Mime.INCOMING_CONNECTION, Mime.OUTGOING_CONNECTION]:
            self.user_actions.cursor_line_dropped.emit()
        e.accept()

    def dragMoveEvent(self, e):
        mime = e.mimeData().text()
        # todo: forcing update here causes GUI lags REGARDLESS of the Lines painEvent() (method can be even empty)
        # - the problem occurs at least on OS X
        # - possible solutions is updating parts of the Lines widget instead of all of it, by detecting possibly 3
        #   rectangles for each line that is being affected by the drag:
        #   http://stackoverflow.com/questions/18043492/qt-custom-widget-update-big-overhead
        # - some improvement can be made to paintEvent() - using QImage to draw lines and then dump it to painter,
        #   which will use software renderer OR/AND QGLWidget:
        #   http://stackoverflow.com/questions/6089642/qpainter-painting-alternatives-performance-sucks-on-mac
        if mime == Mime.INCOMING_CONNECTION:
            points = (e.source(), e.pos())
            self.user_actions.cursor_line_moved.emit(points)
        elif mime == Mime.OUTGOING_CONNECTION:
            points = (e.pos(), e.source())
            self.user_actions.cursor_line_moved.emit(points)

    def resizeEvent(self, e):
        self.wires_in_foreground.setGeometry(self.rect())
        self.wires_in_background.setGeometry(self.rect())

    def zoom(self, level=None, index=None):

        assert (level is None and index is not None) or (level is not None and index is None)

        if index is not None:
            new_level = None
            if index > 0:
                for i in range(len(self.zoom_levels)):
                    if self.zoom_levels[i] > self.diagram.zoom_level:
                        new_level = i
                        break
            else:
                for i in reversed(range(len(self.zoom_levels))):
                    if self.zoom_levels[i] < self.diagram.zoom_level:
                        new_level = i
                        break
            if new_level is None:
                return

            level = self.zoom_levels[new_level]

        factor = level / self.diagram.zoom_level
        self.diagram.zoom_level = level

        self.setFixedSize(int(self.DEFAULT_SIZE * level), (self.DEFAULT_SIZE * level))

        for e in self.diagram.elements:
            assert isinstance(e, GuiElement)
            e.zoom(factor)

        self.actualize_style()
        return factor

    def actualize_style(self):
        def sub(match):
            value = int(match.group(1))
            unit = match.group(2)
            if value == 0:
                return "0" + unit
            value = int(round(value * self.diagram.zoom_level))
            value = max(value, 1)
            return str(value) + unit

        style = self.style_manager.stylesheet
        style = re.sub(r"(\d+)\s*(px|pt)", sub, style)
        self.setStyleSheet(style)

        # workaround for styles not updating on linux
        if os.name == 'posix':
            refresh_style_recursive(self)

        # adjust layout spacings, as they cannot be set in stylesheets (meh...)
        for element in self.diagram.elements:
            element.actualize_style()

    def nearest_grid_point(self, x, y):
        return int(round(float(x)/self.position_grid) * self.position_grid), int(round(float(y)/self.position_grid) * self.position_grid)

    def center_elements(self):
        pos = QPoint(0, 0)
        for e in self.diagram.elements:
            pos += e.pos() + QPoint(e.width()//2, e.height()//2)
        pos /= len(self.diagram.elements)
        pos = QPoint(self.width()//2, self.height()//2) - pos
        for e in self.diagram.elements:
            e.move(e.pos() + pos)
            e.element_relocated.emit(e)


class SelectionManager:

    def __init__(self, workarea):
        self.workarea = workarea
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.workarea)
        self.selection_origin = None
        self.selected_elements = []
        self.last_selection = []
        self.workarea.diagram.element_deleted.connect(self.on_element_deleted)

    def selected_count(self):
        return len(self.selected_elements)

    def on_element_moused_left_pressed(self, element, event):
        ctrl_pressed = self.is_control_pressed()
        if not element.selected:
            if not ctrl_pressed:
                self.clear_selection()
            self.select_element(element)
        else:
            if ctrl_pressed:
                self.unselect_element(element)
            else:
                self.select_element(element)

    def on_element_right_pressed(self, element, event):
        if not element.selected:
            self.clear_selection()
            self.select_element(element)

    def on_element_left_released_inplace(self, element, event):
        ctrl_pressed = self.is_control_pressed()
        if not ctrl_pressed:
            self.clear_selection()
            self.select_element(element)

    def on_workarea_mouse_left_pressed(self, event):
        if not self.is_control_pressed():
            self.clear_selection()
        self.selection_origin = QtCore.QPoint(event.pos())
        self.rubberBand.setGeometry(QtCore.QRect(self.selection_origin, QtCore.QSize()))
        self.rubberBand.show()
        self.last_selection = self.selected_elements[:]

    def on_workarea_mouse_left_move(self, event):
        if self.selection_origin is not None and not self.selection_origin.isNull():
            rect = QtCore.QRect(self.selection_origin, event.pos()).normalized()
            # FIXME: Hide and show prevent a strange effect of QRubberBand of shaking the widgets beneath, which occur
            # when making selection in a direction different the right-bottom, but only after placing an element from
            # the palette - on a freshly loaded diagram everything is fine
            self.rubberBand.hide()
            self.rubberBand.setGeometry(rect)
            self.rubberBand.show()
            for e in self.workarea.diagram.elements:
                ctrl_pressed = self.is_control_pressed()
                if self.is_element_in_rect(e, rect):
                    if ctrl_pressed and e in self.last_selection:
                        self.unselect_element(e)
                    else:
                        self.select_element(e)
                else:
                    if not ctrl_pressed:
                        self.unselect_element(e)

    def on_workarea_mouse_left_released(self, event):
        self.rubberBand.hide()
        self.last_selection[:] = []

    def delete_selected(self):
        while len(self.selected_elements) != 0:
            self.workarea.diagram.delete_element(self.selected_elements[0])

    def select_element(self, element):
        if not element.selected:
            self.selected_elements.append(element)
            element.set_selected(True)

    def select_all_elements(self):
        self.clear_selection()
        for e in self.workarea.diagram.elements:
            self.select_element(e)

    def unselect_element(self, element):
        if element.selected:
            if element in self.selected_elements:
                self.selected_elements.remove(element)
            element.set_selected(False)

    def clear_selection(self):
        for e in self.selected_elements:
            e.set_selected(False)
        self.selected_elements = []

    def is_element_in_rect(self, element, rect):
        return rect.contains(element.frameGeometry())

    def is_control_pressed(self):
        return (int(QApplication.keyboardModifiers()) & QtCore.Qt.ControlModifier) != 0

    def on_element_deleted(self, element):
        if element in self.selected_elements:
            self.selected_elements.remove(element)
        if element in self.last_selection:
            self.last_selection.remove(element)
