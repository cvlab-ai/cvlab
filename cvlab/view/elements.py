from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication

from .styles import StyleManager
from ..diagram import code_generator
from ..diagram.element import Element
from .parameters import *
from .mimedata import Mime
from .widgets import InOutConnector, ElementStatusBar, PreviewsContainer, StyledWidget
from .wires import NO_FOREGROUND_WIRES


SHOW_ELEMENT_ID = False


class GuiElement(Element, StyledWidget):
    state_changed = pyqtSignal()
    element_relocated = pyqtSignal(Element)

    help = """\
Right click - menu
Double click - toggle previews 
Drag & drop - move element around"""

    def __init__(self):
        super(GuiElement, self).__init__()
        self.setObjectName("Element")
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.status_bar = ElementStatusBar(self)
        self.label = None
        self.preview = None
        self.params = None
        self.input_connectors = {}
        self.output_connectors = {}
        self.param_sliders = []
        self.setAcceptDrops(True)
        self.hints_shown = False
        self.standard_actions = []
        self.group_actions = []
        self.workarea = None
        self.state_notified = False
        self.selected = False
        self.setToolTip(self.name + "\n-------------------------\n" + self.help + "\n-------------------------\n" + self.comment)

    def set_selected(self, select):
        self.setProperty("selected", select)
        self.label.setProperty("selected", select)
        self.workarea.style().polish(self)
        self.workarea.style().polish(self.label)
        self.workarea.style().unpolish(self)
        self.workarea.style().unpolish(self.label)
        self.selected = select

    def set_workarea(self, workarea):
        self.workarea = workarea
        for connector in list(self.input_connectors.values()) + list(self.output_connectors.values()):
            connector.set_workarea(workarea)
        self.recreate_group_actions()

    def actualize_style(self):
        layouts = [self.layout()]
        while layouts:
            layout = layouts.pop()

            dpi_factor = 2 if StyleManager.is_highdpi else 1

            base_contents_margins = getattr(layout, "base_contents_margins", None)
            if base_contents_margins:
                margins = (np.array(base_contents_margins) * dpi_factor * self.workarea.diagram.zoom_level).clip(1,1000).round().astype(int).tolist()
                layout.setContentsMargins(*margins)

            base_spacing = getattr(layout, "base_spacing", None)
            if base_spacing:
                spacing = max(int(base_spacing * self.workarea.diagram.zoom_level * dpi_factor),1)
                layout.setSpacing(spacing)

            for child in layout.children():
                if isinstance(child, QLayout):
                    layouts.append(child)

    def create_label(self, layout):
        if SHOW_ELEMENT_ID:
            self.label = QLabel("{} #{}".format(self.name, self.unique_id))
        else:
            self.label = QLabel("{}".format(self.name))
        self.label.setObjectName("ElementLabel")
        layout.addWidget(self.label)

    def create_params(self, container):
        self.params = container
        layout = container.layout()
        params = list(self.parameters.items())
        for name, param in params:
            if isinstance(param, ButtonParameter):
                layout.addLayout(GuiButtonParameter(param))
            elif isinstance(param, PathParameter):
                layout.addLayout(GuiPathParameter(param))
            elif isinstance(param, MultiPathParameter):
                layout.addLayout(GuiMultiPathParameter(param))
            elif isinstance(param, DirectoryParameter):
                layout.addLayout(GuiDirectoryParameter(param))
            elif isinstance(param, IntParameter):
                layout.addLayout(GuiIntParameter(param, self))
            elif isinstance(param, FloatParameter):
                layout.addLayout(GuiFloatParameter(param, self))
            elif isinstance(param, ComboboxParameter):
                layout.addLayout(GuiComboboxParameter(param, self))
            elif isinstance(param, SizeParameter) or isinstance(param, PointParameter):
                layout.addLayout(GuiMultiNumberParameter(param, self, 2, int))
            elif isinstance(param, ScalarParameter):
                layout.addLayout(GuiMultiNumberParameter(param, self, 4, float))
            elif isinstance(param, TextParameter):
                layout.addLayout(GuiTextParameter(param, self))

    def create_inputs(self, layout):
        layout.base_contents_margins = [0,3,0,3]
        layout.base_spacing = 7
        for input_ in self.inputs.values():
            is_input = True
            input_connector = InOutConnector(self, input_, is_input)
            layout.addWidget(input_connector)
            self.input_connectors[input_] = input_connector

    def create_outputs(self, layout):
        layout.base_contents_margins = [0,3,0,3]
        layout.base_spacing = 7
        for output in self.outputs.values():
            is_input = False
            output_connector = InOutConnector(self, output, is_input)
            layout.addWidget(output_connector)
            self.output_connectors[output] = output_connector

    def create_preview(self, layout):
        self.preview = PreviewsContainer(self, list(self.outputs.values()))
        layout.addWidget(self.preview)

    def create_switch_params_action(self):
        action = QAction('Show p&arams', self)
        action.triggered.connect(self.switch_params)
        action.setCheckable(True)
        self.standard_actions.append(action)
        self.addAction(action)

    def create_switch_preview_action(self):
        action = QAction('Show &preview\t[Double click]', self)
        action.triggered.connect(self.switch_preview)
        action.setCheckable(True)
        self.standard_actions.append(action)
        self.addAction(action)

    def create_switch_sliders_action(self):
        action = QAction('Show &sliders', self)
        action.triggered.connect(self.switch_sliders)
        action.setCheckable(True)
        self.standard_actions.append(action)
        self.addAction(action)

    def create_del_action(self):
        del_action = QAction('&Delete', self)
        del_action.triggered.connect(self.selfdestroy)
        self.standard_actions.append(del_action)
        self.addAction(del_action)

    def create_code_action(self):
        code_action = QAction('&Generate code', self)
        code_action.setToolTip("Generates python code for executing the whole diagram up to this element")
        code_action.triggered.connect(self.gen_code_action)
        self.standard_actions.append(code_action)
        self.addAction(code_action)

    def create_duplicate_action(self):
        dup_action = QAction('D&uplicate', self)
        dup_action.setToolTip("Duplicates the element.\nAll parameter values will be SHARED with the copy!")
        dup_action.triggered.connect(self.duplicate)
        self.standard_actions.append(dup_action)
        self.addAction(dup_action)

    def create_break_action(self):
        action = QAction('Disable parameter sharing', self)
        action.triggered.connect(self.break_connections)
        self.standard_actions.append(action)
        self.addAction(action)

    def create_menu_separator(self):
        separator = QAction(self)
        separator.setSeparator(True)
        self.standard_actions.append(separator)
        self.addAction(separator)

    def recreate_group_actions(self):
        self.group_actions[:] = []
        action = QAction('&Delete selected', self)
        action.triggered.connect(self.workarea.selection_manager.delete_selected)
        self.group_actions.append(action)
        self.addAction(action)

    def prepare_actions(self):
        group_selected = self.workarea.selection_manager.selected_count() > 1
        for action in self.standard_actions:
            action.setVisible(not group_selected)
        for action in self.group_actions:
            action.setVisible(group_selected)

    def update_id(self):
        if SHOW_ELEMENT_ID:
            self.label.setText("{} #{}".format(self.name, self.unique_id))

    @pyqtSlot()
    def gen_code_action(self):
        code = code_generator.generate(self)
        QApplication.instance().clipboard().setText(code)
        msg = QMessageBox(QMessageBox.Information, "Code copied",
                                "Code copied to system clipboard.\n\n" + code[:200] + "...", QMessageBox.Ok, self)
        msg.setModal(True)
        msg.show()

    @pyqtSlot()
    def switch_params(self, value=None):
        if not self.params: return
        if value is None: value = not self.params.isVisible()
        self.params.setVisible(value)
        for action in self.actions():
            if action.text() == 'Show p&arams':
                action.setChecked(value)

    @pyqtSlot()
    def switch_preview(self, value=None):
        self.preview.switch_visibility(value)
        for action in self.actions():
            if action.text() == 'Show &preview':
                action.setChecked(self.preview.isVisible())

    @pyqtSlot()
    def switch_sliders(self, value=None):
        if not self.param_sliders:
            return
        if value is None: value = not self.param_sliders[0].isVisible()
        for slider in self.param_sliders:
            slider.setVisible(value)
        for action in self.actions():
            if action.text() == 'Show &sliders':
                action.setChecked(value)

    @pyqtSlot()
    def break_connections(self):
        if not self.parameters: return
        for par in list(self.parameters.values()):
            par.disconnect_all_children()

    @pyqtSlot()
    def selfdestroy(self):
        #TODO: do we need this here or somewhere else?
        self.diagram.delete_element(self)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MiddleButton:
            e.ignore()
            return
        self.workarea.on_element_moused_pressed(self, e)
        if e.button() == QtCore.Qt.RightButton:
            self.prepare_actions()
        self.raise_()
        if not NO_FOREGROUND_WIRES:
            self.workarea.wires_in_foreground.raise_()
        # see: WiresBase.__init__ comments!
        # self.setGraphicsEffect(QGraphicsOpacityEffect())
        self.hide_hints()

    def mouseDoubleClickEvent(self, e):
        if e.button() == QtCore.Qt.MiddleButton:
            e.ignore()
            return
        self.switch_preview()

    def mouseReleaseEvent(self, e):
        if e.button() == QtCore.Qt.MiddleButton:
            e.ignore()
            return
        if e.button() == QtCore.Qt.LeftButton:
            self.workarea.on_element_moused_left_released(self, e)
        # self.setGraphicsEffect(None)
        self.show_hints()

    def mouseMoveEvent(self, e):
        if e.buttons() & QtCore.Qt.MiddleButton:
            e.ignore()
            return
        if e.buttons() & QtCore.Qt.LeftButton:
            self.workarea.on_element_moused_left_moved(self, e)
            #self.show_hints()

    def show_hints(self):
        if self.hints_shown: return
        self.hints_shown = True
        for i in list(self.input_connectors.values()) + list(self.output_connectors.values()):
            i.show_hint()

    def hide_hints(self):
        if not self.hints_shown: return
        self.hints_shown = False
        for i in list(self.input_connectors.values()) + list(self.output_connectors.values()):
            i.hide_hint()

    def enterEvent(self, e):
        self.show_hints()

    def leaveEvent(self, e):
        if not self.workarea.wires_in_foreground.is_cursor_line_active:
            self.hide_hints()

    def forward_event_to_io_connector(self, name, event):
        if event.mimeData().text() == Mime.INCOMING_CONNECTION and self.input_connectors:
            connector = self.find_nearest_connector(event, list(self.input_connectors.values()))
            method_to_call = getattr(connector, name)
            method_to_call(event)
        elif event.mimeData().text() == Mime.OUTGOING_CONNECTION and self.output_connectors:
            connector = self.find_nearest_connector(event, list(self.output_connectors.values()))
            method_to_call = getattr(connector, name)
            method_to_call(event)

    def find_nearest_connector(self, event, connectors):
        key = lambda c: abs(event.pos().y() - c.pos().y() - c.height() // 2)
        return min(connectors, key=key)

    def resizeEvent(self, e):
        self.element_relocated.emit(self)

    def dragEnterEvent(self, e):
        self.forward_event_to_io_connector(self.dragEnterEvent.__name__, e)

    def dropEvent(self, e):
        self.forward_event_to_io_connector(self.dropEvent.__name__, e)

    def dragMoveEvent(self, e):
        #todo: fix shifted wire when to connectors of the same element are to be connected
        self.forward_event_to_io_connector(self.dragMoveEvent.__name__, e)

    def notify_state_changed(self):
        if not self.state_notified:
            self.state_notified = True
            # We need to use a signal here since the method is called by a worker thread and we need to alter the GUI
            self.state_changed.emit()

    def duplicate(self):
        el = self.__class__()
        pos = (self.pos().x() + 20, self.pos().y() + 20)
        self.diagram.add_element(el, pos)
        for my, his in zip(list(self.parameters.values()), list(el.parameters.values())):
            my.connect_child(his)
            his.connect_child(my)
        for my, his in zip(list(self.inputs.values()), list(el.inputs.values())):
            for outp in my.connected_from:
                self.diagram.connect_io(his, outp)
        if self.params:
            el.switch_params(self.params.isVisible())
        if self.param_sliders:
            el.switch_sliders(self.param_sliders[0].isVisible())
        el.switch_preview(self.preview.isVisible())

    def to_json(self):
        parent_d = Element.to_json(self)

        dpi_factor = 2 if StyleManager.is_highdpi else 1

        d = {
            "show_parameters": (self.params.isVisible() if self.params else None),
            "show_sliders": (self.param_sliders[0].isVisible() if self.param_sliders else None),
            "show_preview": (self.preview.isVisible() if self.preview else None),
            "position": (self.pos().x()//dpi_factor, self.pos().y()//dpi_factor),
            "preview_size": self.preview.preview_size//dpi_factor,
        }
        parent_d["gui_options"] = d
        return parent_d

    def from_json(self, data):
        dpi_factor = 2 if StyleManager.is_highdpi else 1

        options = data["gui_options"]
        self.switch_params(options['show_parameters'] is True)
        self.switch_sliders(options["show_sliders"])
        if "preview_size" in options \
            and options["preview_size"] \
            and options["preview_size"] != self.preview.preview_size:
                self.preview.preview_size = options["preview_size"] * dpi_factor
        self.switch_preview(options["show_preview"])

        self.move(options["position"][0]*dpi_factor,options["position"][1]*dpi_factor)

        Element.from_json(self, data)
        self.update_id()
        self.preview.force_update()

    def zoom(self, factor, origin):
        assert isinstance(self.preview, PreviewsContainer)

        # fixme: this is not accurate (elements are zoomed by top-left position, sizes and positions are integers...)

        factor = float(factor)
        origin_x, origin_y = origin

        x, y = self.workarea.nearest_grid_point((self.x() - origin_x) * factor + origin_x,
                                                (self.y() - origin_y) * factor + origin_y)
        self.move(x, y)

        self.preview.resize_previews(self.preview.preview_size * factor)

        self.element_relocated.emit(self)

    def deleteLater(self):
        self.state_changed.disconnect()
        # self.setParent(None)
        super(GuiElement, self).deleteLater()


class FunctionGuiElement(GuiElement):
    def __init__(self):
        super(FunctionGuiElement, self).__init__()
        vb_main = QVBoxLayout()
        hb_content = QHBoxLayout()
        hb_label = QHBoxLayout()
        vb_inputs = QVBoxLayout()
        vb_params = QVBoxLayout()
        vb_outputs = QVBoxLayout()
        vb_inputs.setAlignment(QtCore.Qt.AlignTop)
        vb_outputs.setAlignment(QtCore.Qt.AlignTop)

        hb_label.setContentsMargins(0,0,0,0)
        hb_label.setSpacing(0)

        w_params = QWidget()
        w_params.setLayout(vb_params)
        vb_params.setContentsMargins(0,0,0,0)
        vb_params.setSpacing(1)

        vb_inputs.base_contents_margins = [4, 4, 4, 4]
        vb_inputs.base_spacing = 4
        vb_outputs.base_contents_margins = [4, 4, 4, 4]
        vb_outputs.base_spacing = 4

        self.create_label(hb_label)
        self.create_params(w_params)
        self.create_inputs(vb_inputs)
        self.create_outputs(vb_outputs)
        hb_content.setSpacing(0)
        hb_content.setContentsMargins(0,0,0,0)
        hb_content.addLayout(vb_inputs)
        hb_content.addWidget(w_params)
        hb_content.addStretch(1)
        hb_content.addLayout(vb_outputs)
        vb_main.addLayout(hb_label)
        vb_main.addLayout(hb_content)
        vb_main.addWidget(self.status_bar)
        vb_main.setSizeConstraint(QLayout.SetFixedSize)
        vb_main.base_contents_margins = [0, 4, 0, 4]
        vb_main.base_spacing = 4
        self.setLayout(vb_main)

        self.create_preview(vb_main)
        self.create_switch_params_action()
        self.create_switch_preview_action()
        self.create_switch_sliders_action()
        self.create_menu_separator()
        self.create_duplicate_action()
        self.create_break_action()
        self.create_del_action()
        self.create_code_action()
        #self.setFocusPolicy(QtCore.Qt.ClickFocus + QtCore.Qt.TabFocus)
        #self.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 1)     # enable showing focus on a Mac


class OperatorGuiElement(GuiElement):
    def __init__(self):
        super(OperatorGuiElement, self).__init__()
        vb_main = QVBoxLayout()
        hb = QHBoxLayout()
        vb_inputs = QVBoxLayout()
        vb_outputs = QVBoxLayout()
        vb_inputs.setAlignment(QtCore.Qt.AlignTop)
        vb_outputs.setAlignment(QtCore.Qt.AlignTop)

        self.create_label(vb_main)
        self.create_inputs(vb_inputs)
        self.create_outputs(vb_outputs)
        hb.addLayout(vb_inputs)
        hb.addStretch(1)
        hb.addLayout(vb_outputs)
        vb_main.addLayout(hb)
        vb_main.addWidget(self.status_bar)
        vb_main.setSizeConstraint(QLayout.SetFixedSize)
        vb_main.base_contents_margins = [0, 4, 0, 4]
        vb_main.base_spacing = 4
        self.create_preview(vb_main)
        self.setLayout(vb_main)

        self.create_switch_preview_action()
        self.create_menu_separator()
        self.create_del_action()
        self.create_code_action()
        self.create_duplicate_action()


class InputGuiElement(GuiElement):
    def __init__(self):
        super(InputGuiElement, self).__init__()
        vb_main = QVBoxLayout()
        hb = QHBoxLayout()
        vb_params = QVBoxLayout()
        vb_outputs = QVBoxLayout()
        vb_outputs.setAlignment(QtCore.Qt.AlignTop)

        w_params = QWidget()
        w_params.setLayout(vb_params)
        vb_params.setContentsMargins(0,0,0,0)
        vb_params.setSpacing(1)

        self.create_label(vb_main)
        self.create_params(w_params)
        self.create_outputs(vb_outputs)
        hb.addWidget(w_params)
        hb.addLayout(vb_outputs)
        vb_main.addLayout(hb)
        vb_main.addWidget(self.status_bar)
        vb_main.setSizeConstraint(QLayout.SetFixedSize)
        vb_main.base_contents_margins = [0, 4, 0, 4]
        vb_main.base_spacing = 0
        self.create_preview(vb_main)
        self.setLayout(vb_main)

        self.create_switch_preview_action()
        self.create_menu_separator()
        self.create_duplicate_action()
        self.create_break_action()
        self.create_del_action()

