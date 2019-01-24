import os
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

import tinycss2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ..diagram.element import Element
from ..diagram.connectors import Input, Output
from .widgets import InOutConnector


# workaround for Windows 8
NO_FOREGROUND_WIRES = os.name == 'nt' and sys.getwindowsversion().major >= 6 and sys.getwindowsversion().minor >= 2


class WiresBase(QWidget):
    def __init__(self, workarea, user_actions, wire_tools):
        super(WiresBase, self).__init__(workarea)
        self.workarea = workarea
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.user_actions = user_actions
        self.wire_tools = wire_tools

    def paintEvent(self, e):
        super(WiresBase, self).paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.draw_wires(painter)

    def draw_wires(self, painter):
        pass


class WiresForeground(WiresBase):
    def __init__(self, workarea, user_actions, wire_tools):
        super(WiresForeground, self).__init__(workarea, user_actions, wire_tools)
        self.cursor_wire = None
        self.cursor_line = None
        self.is_cursor_line_active = False
        user_actions.cursor_line_started.connect(self.on_cursor_line_started)
        user_actions.cursor_line_moved.connect(self.on_cursor_line_moved)
        user_actions.cursor_line_dropped.connect(self.on_cursor_line_dropped)

    def draw_wires(self, painter):
        if self.cursor_wire is not None:
            foreground_pen = self.wire_tools.pen_selected
            painter.strokePath(self.cursor_wire.line, foreground_pen.line)
            self.wire_tools.draw_start_symbol(painter, self.cursor_wire.start_point, foreground_pen)
            self.wire_tools.draw_end_symbol(painter, self.cursor_wire.end_point, foreground_pen)

    @pyqtSlot()
    def on_cursor_line_started(self):
        self.is_cursor_line_active = True

    @pyqtSlot(tuple)
    def on_cursor_line_moved(self, points):
        self.cursor_line = list(points)
        self.update()

    @pyqtSlot()
    def on_cursor_line_dropped(self):
        self.is_cursor_line_active = False
        self.update()

    def update(self):
        self.handle_cursor_line()
        super(WiresForeground, self).update()

    def handle_cursor_line(self):
        if self.cursor_wire is not None:
            self.cursor_wire = None
        if self.cursor_line is not None:
            self.cursor_wire = Wire(self.cursor_line[0], self.cursor_line[1], workarea=self.workarea)
            self.cursor_line = None


class WiresBackground(WiresBase):
    def __init__(self, workarea, user_actions, wire_tools):
        super(WiresBackground, self).__init__(workarea, user_actions, wire_tools)
        self.manager = Manager(workarea)
        self.wire_click_margin = 10
        workarea.user_actions.element_relocated.connect(self.on_element_relocated)
        workarea.diagram.connection_created.connect(self.on_connection_created)
        workarea.diagram.connection_deleted.connect(self.on_connection_deleted)

    def draw_wires(self, painter):
        # all lines could be potentially drawn as a single QPainterPath for better performance,
        # but this would break nice covering of wires by other wires
        for wire in self.manager.wires:
            if wire.selected:
                painter.strokePath(wire.line, self.wire_tools.pen_selected_background.line)
                painter.strokePath(wire.line, self.wire_tools.pen_selected.line)
            else:
                painter.strokePath(wire.line, self.wire_tools.pen_regular.line)

    @pyqtSlot(Output, Input)
    def on_connection_created(self, output, input_):
        gui_output = self.workarea.connectors_map[output]
        gui_input = self.workarea.connectors_map[input_]
        self.manager.create_wire_if_not_exists(gui_output, gui_input)
        self.update()

    @pyqtSlot(Output, Input)
    def on_connection_deleted(self, output, input_):
        self.manager.remove_wire_by_connectors(output, input_)
        self.update()

    @pyqtSlot(Element)
    def on_element_relocated(self, element):
        self.manager.update_wires_by_element(element)
        self.update()

    def mousePressEvent(self, e):
        wire_clicked = False
        self.unselect_wires()
        for wire in self.manager.wires:
            if wire.is_point_on_wire(e.pos(), self.wire_click_margin):
                wire.selected = True
                e.accept()
                if e.button() == QtCore.Qt.RightButton:
                    menu = QMenu()
                    action = menu.addAction("&Delete")
                    action.triggered.connect(wire.selfdestroy)
                    menu.exec_(QtCore.QPoint(e.globalX(), e.globalY()))
                wire_clicked = True
                break
        if wire_clicked:
            self.update()

    def unselect_wires(self):
        for w in self.manager.wires:
           w.selected = False
        self.update()


class Wire:
    def __init__(self, start_object, end_object, manager=None, workarea=None):
        self._selected = False
        self.manager = manager
        self.workarea = workarea if manager is None else manager.workarea
        self.start_widget = None
        self.start_point = None
        self.end_widget = None
        self.end_point = None
        self.line_points = None
        self.arrow_points = None
        self.line = None
        self.arrow = None
        self.extender = 25
        self.get_ends_points(start_object, end_object)
        self.prepare_paths()

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value):
        self._selected = value
        for widget in [self.start_widget, self.end_widget]:
            if widget is not None:
                widget.has_connected_and_selected_wire = value

    def selfdestroy(self):
        self.selected = False
        if self.manager is not None:
            diagram = self.manager.workarea.diagram
            diagram.disconnect_io(self.start_widget.io_handle, self.end_widget.io_handle)

    def update_position(self):
        if self.start_widget is None or self.end_widget is None:
            return
        if not self.is_up_to_date():
            self.start_point = self.start_widget.get_center_point()
            self.end_point = self.end_widget.get_center_point()
            self.prepare_paths()

    def is_point_on_wire(self, point, margin):
        for i in range(len(self.line_points) - 1):
            point_a = self.line_points[i]
            point_b = self.line_points[i + 1]
            if self.is_point_inside_line_segment(point_a, point_b, point, margin):
                return True
        return False

    def is_point_inside_line_segment(self, point_a, point_b, point, margin):
        # assuming that line segment is vetical or horizontal
        left_margin = min(point_a.x(), point_b.x()) - margin
        right_margin = max(point_a.x(), point_b.x()) + margin
        top_margin = min(point_a.y(), point_b.y()) - margin
        bottom_margin = max(point_a.y(), point_b.y()) + margin
        if left_margin < point.x() < right_margin and top_margin < point.y() < bottom_margin:
            return True
        else:
            return False

    def is_up_to_date(self):
        return self.start_point == self.start_widget.get_center_point() and \
               self.end_point == self.end_widget.get_center_point()

    def prepare_paths(self):
        self.line_points = self.get_line_points()
        self.arrow_points = self.workarea.wire_tools.get_arrow_points(self.end_point)
        self.line = self.workarea.wire_tools.get_path_from_points(self.line_points)
        self.arrow = self.workarea.wire_tools.get_path_from_points(self.arrow_points)

    def get_line_points(self):
        points = []
        start = self.start_point
        end = self.end_point
        points.append(start)

        # Determine the optimal shape of the wire
        if end.x() > start.x() and end.y() == start.y():    # 1-segment, vertical wire
            # no middle points are needed
            pass

        elif end.x() - start.x() > self.extender * 2:       # 3-segment wire, with horizontal line position optimized to omit Elements
            mid_x = WireOptimizer.get_optimal_vertical_midline_x_position(self)
            points.append(QtCore.QPoint(mid_x, start.y()))
            points.append(QtCore.QPoint(mid_x, end.y()))

        else:           # 5-segment wire, with vertical line position optimized to omit Elements
            mid_y = WireOptimizer.optimize_y_mid_point(self)
            points.append(QtCore.QPoint(start.x() + self.extender, start.y()))
            points.append(QtCore.QPoint(start.x() + self.extender, mid_y))
            points.append(QtCore.QPoint(end.x() - self.extender, mid_y))
            points.append(QtCore.QPoint(end.x() - self.extender, end.y()))

        points.append(end)
        return points

    def get_ends_points(self, start_object, end_object):
        if isinstance(start_object, InOutConnector):
            self.start_point = start_object.get_center_point()
            self.start_widget = start_object
        elif isinstance(start_object, QtCore.QPoint):
            self.start_point = start_object

        if isinstance(end_object, InOutConnector):
            self.end_point = end_object.get_center_point()
            self.end_widget = end_object
        elif isinstance(end_object, QtCore.QPoint):
            self.end_point = end_object


class WireOptimizer:

    """
    Class optimizing two kinds of wires shapes:

    - 3-segment wire, looking like:  -----|
                                          |----->
                                          ^ a vertical mid line

    - 5-segment wires, looking like:                --|
                                       |--------------|  < a horizontal mid line
                                       |-->
    """

    DISTANCE = 20   # a minimum distance between a wire and an element (in the considered x or y direction)

    @staticmethod
    def get_optimal_vertical_midline_x_position(wire):

        """
        Get optimal x position of a the vertical mid line of a 3-segment wire, which omits other Elements in the Workarea.
        If omitting is not possible, return the middle between the wire start and end points.
        """

        elements = WireOptimizer.get_all_colliding_elements(wire)
        min_x, max_x = WireOptimizer.get_valid_x_range(wire, elements)
        mid_x = (wire.start_point.x() + wire.end_point.x())//2
        x = min(max(mid_x, min_x), max_x)
        element_left = element_right = WireOptimizer.get_colliding_element_by_x(wire, x, elements)
        _continue = True
        while (_continue):
            _continue = False
            if element_right is not None:
                midx_right = element_right.x() + element_right.width() + WireOptimizer.DISTANCE
                element_right = WireOptimizer.get_colliding_element_by_x(wire, midx_right, elements)
                if midx_right < max_x:
                    if element_right is None:
                        return midx_right
                    _continue = True
            if element_left is not None:
                midx_left = element_left.x() - WireOptimizer.DISTANCE
                element_left = WireOptimizer.get_colliding_element_by_x(wire, midx_left, elements)
                if midx_left > min_x:
                    if element_left is None:
                        return midx_left
                    _continue = True
        return x

    @staticmethod
    def optimize_y_mid_point(wire):

        """
        Get optimal y position of a the horizontal mid line of a 5-segment wire, which omits other Elements in the Workarea.
        If omitting is not possible, return the middle between the wire start and end points.
        """

        elements = WireOptimizer.get_all_colliding_elements(wire, wire.extender)
        min_y, max_y = WireOptimizer.get_valid_y_range(wire, elements)
        mid_y = (wire.start_point.y() + wire.end_point.y())//2
        y = min(max(mid_y, min_y), max_y)
        element_top = element_bottom = WireOptimizer.get_colliding_element_by_y(wire, y, elements)
        _continue = True
        while _continue:
            _continue = False
            if element_bottom is not None:
                midy_bottom = element_bottom.y() + element_bottom.height() + WireOptimizer.DISTANCE
                element_bottom = WireOptimizer.get_colliding_element_by_y(wire, midy_bottom, elements)
                if midy_bottom < max_y:
                    if element_bottom is None:
                        return midy_bottom
                    _continue = True
            if element_top is not None:
                midy_top = element_top.y() - WireOptimizer.DISTANCE
                element_top = WireOptimizer.get_colliding_element_by_y(wire, midy_top, elements)
                if midy_top > min_y:
                    if element_top is None:
                        return midy_top
                    _continue = True
        return y

    @staticmethod
    def get_valid_x_range(wire, elements):
        min_x = wire.start_point.x()
        max_x = wire.end_point.x()
        for element in elements:
            if WireOptimizer.is_element_on_y(element, wire.start_point.y()):
                max_x = min(max_x, element.x())
            if WireOptimizer.is_element_on_y(element, wire.end_point.y()):
                min_x = max(min_x, element.x() + element.width())
        min_x += WireOptimizer.DISTANCE
        max_x -= WireOptimizer.DISTANCE
        return min_x, max_x

    @staticmethod
    def get_valid_y_range(wire, elements):
        min_y = wire.start_point.y()
        max_y = wire.end_point.y()
        for element in elements:
            if WireOptimizer.is_element_on_x(element, wire.start_point.x() + wire.extender):
                max_y = min(max_y, element.y())
            if WireOptimizer.is_element_on_x(element, wire.end_point.x() - wire.extender):
                min_y = max(min_y, element.y() + element.height())
        min_y += WireOptimizer.DISTANCE
        max_y -= WireOptimizer.DISTANCE
        return min_y, max_y

    @staticmethod
    def is_element_on_x(element, x):
        return element.x() <= x <= element.x() + element.width()

    @staticmethod
    def is_element_on_y(element, y):
        return element.y() <= y <= element.y() + element.height()

    @staticmethod
    def get_all_colliding_elements(wire, x_extension=0):

        """
        Get list of all elements utterly or partly lying on the rectangle defined by start and end points of the wire.

        param: wire: The considered wire
        param: x_extension: A value by which the rectangle is extended both on the right and left
        """

        elements = []
        top, bottom = sorted([wire.start_point.y(), wire.end_point.y()])
        left, right = sorted([wire.start_point.x(), wire.end_point.x()])
        left -= x_extension
        right += x_extension
        rect = ((left, top), (right, bottom))
        wire_elements = []
        if wire.start_widget is not None:
            wire_elements.append(wire.start_widget.element)
        if wire.end_widget is not None:
            wire_elements.append(wire.end_widget.element)
        for element in wire.workarea.diagram.elements:
            if element not in wire_elements:
                x1 = element.x()
                y1 = element.y()
                x2 = element.x() + element.width()
                y2 = element.y() + element.height()
                for point in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
                    if WireOptimizer.is_point_in_rect(point, rect):
                        elements.append(element)
                        break
        return elements


    @staticmethod
    def is_point_in_rect(point, rect):
        return rect[0][0] <= point[0] <= rect[1][0] and \
               rect[0][1] <= point[1] <= rect[1][1]

    @staticmethod
    def get_colliding_element_by_x(wire, x, elements):
        for element in elements:
            if element.x() - WireOptimizer.DISTANCE < x < element.x() + element.width() + WireOptimizer.DISTANCE:
                return element
        return None

    @staticmethod
    def get_colliding_element_by_y(wire, y, elements):
        for element in elements:
            if element.y() - WireOptimizer.DISTANCE < y < element.y() + element.height() + WireOptimizer.DISTANCE:
                return element
        return None


class Manager:
    def __init__(self, workarea):
        self.wires = []
        self.connectors_map = {}
        self.workarea = workarea

    def create_wire_if_not_exists(self, output_connector, input_connector):
        wire_exists = False
        for wire in self.wires:
            if wire.start_widget == output_connector and wire.end_widget == input_connector:
                wire_exists = True
        if not wire_exists:
            wire = Wire(output_connector, input_connector, self)
            self.wires.append(wire)
            self.connectors_map[(output_connector.io_handle, input_connector.io_handle)] = wire

    def remove_wire_by_connectors(self, output, input_):
        key = (output, input_)
        if key in self.connectors_map:
            wire = self.connectors_map[key]
            self.wires.remove(wire)
            del self.connectors_map[key]

    def update_wires_by_element(self, element):
        for input_ in element.inputs.values():
            for output in input_.connected_from:
                key = (output, input_)
                if key in self.connectors_map:
                    wire = self.connectors_map[key]
                    wire.update_position()
        for output in element.outputs.values():
            for input_ in output.connected_to:
                key = (output, input_)
                if key in self.connectors_map:
                    wire = self.connectors_map[key]
                    wire.update_position()


class WirePen:
    def __init__(self, color, size, dotted=False):
        self.arrow = QPen(color, size)
        self.line = QPen(color, size)
        if dotted:
            self.line.setStyle(QtCore.Qt.DotLine)


class WireTools:

    def __init__(self, style_manager):
        self.pen_regular = None
        self.pen_selected = None
        self.pen_selected_background = None

        self.style_manager = style_manager
        self.wire_style = None
        self.style_manager.style_changed.connect(self.update_style)
        self.update_style()

    def update_style(self):
        # Update wire style
        self.wire_style = self.style_manager.wire_style

        # Prepare new drawing tools
        self.pen_regular = WirePen(self.wire_style.pen_regular_color,
                                   self.wire_style.pen_regular_size,
                                   dotted=True)
        self.pen_selected = WirePen(self.wire_style.pen_selected_color,
                                    self.wire_style.pen_selected_size,
                                    dotted=True)
        self.pen_selected_background = WirePen(self.wire_style.pen_selected_bg_color,
                                               self.wire_style.pen_selected_bg_size)

    def get_arrow_points(self, point):
        points = []
        end = QtCore.QPoint(point) + QtCore.QPoint(self.wire_style.end_arrow_move_to_left, 0)
        width = self.wire_style.end_arrow_width
        half_height = self.wire_style.end_arrow_height//2
        points.append(QtCore.QPoint(end.x() - width, end.y() - half_height))
        points.append(end)
        points.append(QtCore.QPoint(end.x() - width, end.y() + half_height))
        return points

    @staticmethod
    def get_path_from_points(points):
        path = QPainterPath()
        if len(points) > 0:
            path.moveTo(points[0])
            for point in points[1::]:
                path.lineTo(point)
        return path

    def draw_start_symbol(self, painter, point, wire_pen):
        path = QPainterPath()
        size = self.wire_style.start_square_size
        half_size = size//2
        path.addRect(point.x() - half_size, point.y() - half_size, size, size)
        painter.fillPath(path, wire_pen.arrow.brush())

    def draw_end_symbol(self, painter, point, wire_pen):
        points = self.get_arrow_points(point)
        path = self.get_path_from_points(points)
        painter.fillPath(path, wire_pen.arrow.brush())


class WireStyle:

    def __init__(self, stylesheet):

        # Connector icons of the wire
        self.start_square_size = None
        self.end_arrow_width = None
        self.end_arrow_height = None
        self.end_arrow_move_to_left = None

        # Lines of the wire
        self.pen_regular_color = None
        self.pen_regular_size = None
        self.pen_selected_color = None
        self.pen_selected_size = None
        self.pen_selected_bg_color = None
        self.pen_selected_bg_size = None

        # Parse stylesheet
        self.simple_parse_qss(stylesheet)

    def simple_parse_qss(self, stylesheet):
        sheet = tinycss2.parse_stylesheet(str(stylesheet), skip_comments=True, skip_whitespace=True)
        for rule in sheet:
            name = tinycss2.parse_one_component_value(rule.prelude)
            if name.type == 'error':
                continue
            if name.value == 'Wire':
                decs = tinycss2.parse_declaration_list(rule.content)
                for dec in decs:
                    if dec.type == 'declaration':
                        for token in dec.value:
                            if token.type in ['ident', 'hash', 'dimension']:
                                self.update_style(dec.name, token.value, token.type)

    def update_style(self, name, value, token_type):
        attribute_name = name.replace("-", "_")
        parsed_value = None
        if token_type == 'dimension':
            parsed_value = int(value)
        elif token_type == 'ident':
            parsed_value = QColor(value)
        elif token_type == 'hash':
            parsed_value = QColor('#' + value)
        setattr(self, attribute_name, parsed_value)
