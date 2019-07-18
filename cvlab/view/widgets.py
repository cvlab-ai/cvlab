from distutils.util import strtobool

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .. import CVLAB_DIR
from ..diagram.interface import *
from .mimedata import *
from . import image_preview
from . import config

ALLOW_UPSIZE = True


class StyledWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        self.setAttribute(QtCore.Qt.WA_StyleSheet)


class InOutConnector(StyledWidget):
    help = """\
Input / output connector anchor
Drag & drop to another element to connect their inputs/outputs"""

    def __init__(self, element, io_handle, is_input=False):
        super(InOutConnector, self).__init__()
        self.setObjectName("InOutButton")
        self.setAcceptDrops(True)
        self.setToolTip(self.help)
        self.is_input = is_input
        self.io_handle = io_handle
        self.element = element
        self.workarea = None
        # workaround: the hint should be a tooltip but it is not - because it can't be transparent for mouse (Qt bug)
        self.hint = QLabel(io_handle.name)
        self.hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.hint.setVisible(False)
        self.has_connected_and_selected_wire = False
        element.element_relocated.connect(self.update_hint_position)

    def set_workarea(self, workarea):
        self.workarea = workarea
        self.hint.setParent(workarea)
        self.workarea.user_actions.cursor_line_started.connect(self.show_hint)
        self.workarea.user_actions.cursor_line_dropped.connect(self.hide_hint)

    def get_center_point(self):
        return self.pos() + self.element.pos() + QtCore.QPoint(self.width() // 2, self.height() // 2)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.element.workarea.user_actions.cursor_line_started.emit()
            drag = QDrag(self)
            mime_data = QtCore.QMimeData()
            if self.is_input:
                mime_data.setText(Mime.OUTGOING_CONNECTION)
            else:
                mime_data.setText(Mime.INCOMING_CONNECTION)
            drag.setMimeData(mime_data)
            drag.setPixmap(QPixmap(1, 1))     # todo: can this hack be removed?
            drag.exec_()

    def dropEvent(self, e):
        source = e.source()
        if self.element is not source.element and self.is_input ^ source.is_input:
            diagram = self.element.workarea.diagram
            diagram.connect_io(self.io_handle, source.io_handle)
            e.accept()
        self.element.workarea.user_actions.cursor_line_dropped.emit()

    def dragEnterEvent(self, e):
        mime = e.mimeData().text()
        if ((mime == Mime.INCOMING_CONNECTION and self.is_input) or
                (mime == Mime.OUTGOING_CONNECTION and not self.is_input)):
            e.accept()

    def dragMoveEvent(self, e):
        mime = e.mimeData().text()
        if mime in [Mime.INCOMING_CONNECTION, Mime.OUTGOING_CONNECTION]:
            if e.source().element is not self.element:
                points = [e.source().get_center_point(), self.get_center_point()]
            else:
                points = [e.source().get_center_point(), e.pos() + self.element.pos()]
            if not self.is_input:
                points.reverse()
            self.element.workarea.user_actions.cursor_line_moved.emit(tuple(points))

    def paintEvent(self, e):
        super(InOutConnector, self).paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if not self.is_input and len(self.io_handle.connected_to):
            pen, point = self.prepare_drawing_details()
            self.workarea.wire_tools.draw_start_symbol(painter, point, pen)
        if self.is_input and len(self.io_handle.connected_from):
            pen, point = self.prepare_drawing_details()
            self.workarea.wire_tools.draw_end_symbol(painter, point, pen)

    def prepare_drawing_details(self):
        wire_tools = self.workarea.wire_tools
        pen = wire_tools.pen_selected if self.has_connected_and_selected_wire else wire_tools.pen_regular
        point = QtCore.QPoint(self.width() // 2, self.width() // 2)
        return pen, point

    @pyqtSlot()
    def show_hint(self):
        self.hint.setVisible(True) #to musi byc w tej kolejnosci, bo inaczej hint.width jest losowe :/
        self.update_hint_position()
        self.hint.raise_()

    @pyqtSlot()
    def hide_hint(self):
        self.hint.setVisible(False)

    @pyqtSlot()
    def update_hint_position(self):
        if self.is_input:
            delta = -self.hint.width() - 5
        else:
            delta = self.width() + 5
        self.hint.move(self.pos() + self.parent().pos() + QtCore.QPoint(delta, 0))


class PreviewsContainer(StyledWidget):
    help = """\
Element outputs preview
Mouse wheel - resize the previews
Double click - open the preview in separate window"""

    def __init__(self, element, outputs):
        super(PreviewsContainer, self).__init__()
        self.element = element
        self.outputs = outputs
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.preview_size = 120
        self.setObjectName("OutputsPreview")
        self.setVisible(False)
        self.previews = []
        self.create_previews(layout)
        self.setLayout(layout)
        self.element.state_changed.connect(self.update)
        self.image_dialogs_count = 0
        self.setToolTip(self.help)

    def wheelEvent(self, event):
        assert isinstance(event, QWheelEvent)
        if event.modifiers() != QtCore.Qt.NoModifier:
            event.ignore()
            return
        new_size = self.preview_size
        up = event.angleDelta().y() > 0
        for _ in range(abs(event.angleDelta().y() // 80)):
            if up:
                new_size *= 1.125
            else:
                new_size /= 1.125
        self.resize_previews(new_size)
        event.accept()

    def resize_previews(self, new_size):
        out_size = np.clip(int(new_size), 16, 2048)
        self.preview_size = new_size
        for preview in self.previews:
            preview.preview_size = out_size
        self.update()

    def create_previews(self, layout):
        for output in self.outputs:
            preview = OutputPreview(output, self)
            self.previews.append(preview)
            layout.addLayout(preview)

    @pyqtSlot()
    def update(self):
        if not hasattr(self, "element"):
            # fixme: tymczasowy hack, bo leci w tym miejscu wyjątek, nie wiem czemu!
            print("Error: ", self, " nie posiada atrybutu 'element'!")
            return
        self.element.state_notified = False  # todo: this must be called by all slots connected to self.element.state_changed
        state = self.element.state
        #if state == self.element.STATE_READY:
        #    self.update_previews(state)
        #else:
        #    self.set_outdated()
        self.update_previews(state) #todo: czy tak, czy lepiej powyzsze z komentarza?

    def switch_visibility(self, value):
        if value is None:
            value = not self.isVisible()
        if not self.isVisible():
            self.force_update()
        self.setVisible(value)

    def set_outdated(self):
        for preview in self.previews:
            preview.set_outdated()

    def update_previews(self, state):
        if self.isVisible() or self.image_dialogs_count:
            for preview in self.previews:
                preview.update()

    def force_update(self):
        for preview in self.previews:
            preview.update(True)


class OutputPreview(QHBoxLayout):
    default_image = None

    def __init__(self, output, previews_container):
        super(OutputPreview, self).__init__()
        if not self.default_image:
            self.default_image = QPixmap(CVLAB_DIR + "/images/default.png")
        self.output = output
        self.previews_container = previews_container
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)
        self.previews = []
        self.previews.append(ActionImage(self))
        self.img = self.default_image
        self.previews[0].setPixmap(self.img)
        self.addWidget(self.previews[0])

    def update(self, forced=False):
        images = self.get_preview_images()
        if not images:
            images = [None]
        self.adjust_number_of_previews(images)
        for i, arr in enumerate(images):
            if forced or self.previews_container.isVisible() or self.previews[i].image_dialog is not None:
                if isinstance(arr, np.ndarray):
                    self.previews[i].set_image(arr)
                elif isinstance(arr, str):
                    self.previews[i].set_text(arr)
                elif isinstance(arr, bool):
                    self.previews[i].set_bool(arr)

    def get_preview_images(self):
        return self.output.get().desequence_all()

    def adjust_number_of_previews(self, preview_images):
        while len(preview_images) > len(self.previews):
            new_label = ActionImage(self)
            self.addWidget(new_label)
            self.previews.append(new_label)
        while len(preview_images) < len(self.previews):
            label_out = self.previews[-1]
            self.previews.pop(-1)
            # todo: again - a memory leak?
            label_out.deleteLater()

    def set_outdated(self):
        # todo: implement this
        pass


class ActionImage(QLabel):

    DATA_TYPE_IMAGE = 0
    DATA_TYPE_TEXT = 1
    DATA_TYPE_VALUE = 3

    def __init__(self, image_preview):
        super(ActionImage, self).__init__()
        self.image_preview = image_preview
        self.previews_container = image_preview.previews_container
        self.id = len(image_preview.previews)
        self.element = self.previews_container.element
        #todo: better Element number than object_id would be appreciated here
        self.name = "{} {}, Output {}, Image {}".format(self.element.name, str(self.element.object_id), image_preview.output.name, str(self.id))
        self.__connected = False
        self.image_dialog = None
        self.data_type = ActionImage.DATA_TYPE_IMAGE
        self.number_output_helper = NumberOutputHelper()
        self.setMargin(0)
        self.prepare_actions()
        self.setObjectName("OutputPreview")

    def set_image(self, arr):
        # remember not to modify arr !!!
        if self.data_type != ActionImage.DATA_TYPE_IMAGE:
            self.prepare_actions()
        self.data_type = ActionImage.DATA_TYPE_IMAGE
        if isinstance(arr, np.ndarray):
            arr = self.preprocess_array(arr)
        if arr is None:
            pass
            # self.setPixmap(self.image_preview.default_image)  # todo: na pewno to chcemy? moze to nam opozniac interfejs!
        elif isinstance(arr, np.ndarray):
            qpix = image_preview.array_to_pixmap(arr)
            qpix_scaled = self.scale_pixmap(qpix)
            self.setPixmap(qpix_scaled)
            if self.image_dialog is not None:
                image_preview.imshow(self.name, qpix, show=False)

    def set_text(self, arr):
        if self.data_type != ActionImage.DATA_TYPE_TEXT:
            self.prepare_actions(enable=False)
        self.data_type = ActionImage.DATA_TYPE_TEXT
        self.setText(arr)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

    def set_bool(self, value):
        if self.data_type != ActionImage.DATA_TYPE_VALUE:
            self.prepare_actions(enable=False)
        self.data_type = ActionImage.DATA_TYPE_VALUE
        qpix = self.number_output_helper.get_output(value)
        qpix = self.scale_pixmap(qpix)
        self.setPixmap(qpix)

    def preprocess_array(self, arr):
        if arr.dtype == np.uint16:
            arr = cv.convertScaleAbs(arr, alpha=255.0/65535.0)
        elif arr.dtype != np.uint8:
            if arr.dtype in [np.float, np.float32, np.float64]:
                arr = arr * 255.
            if arr.min() < 0:
                arr = arr // 2 + 127
            arr = arr.clip(0, 255)
            arr = np.uint8(arr)
        if len(arr.shape) == 2:
            return cv.cvtColor(arr, cv.COLOR_GRAY2BGRA)
        elif len(arr.shape) == 3 and arr.shape[2] == 3:
            return cv.cvtColor(arr, cv.COLOR_BGR2BGRA)
        else:
            return None

    def scale_pixmap(self, qpix):
        hq = bool(strtobool(config.ConfigWrapper.get_settings().get_with_default(config.VIEW_SECTION,
                                                                                 config.VIEW_HQ_OPTION)))
        quality = QtCore.Qt.SmoothTransformation if hq else QtCore.Qt.FastTransformation
        size = self.previews_container.preview_size
        if not ALLOW_UPSIZE and size > max(qpix.width(), qpix.height()):
            size = max(qpix.width(), qpix.height())
        if qpix.width() > qpix.height():
            result = qpix.scaledToWidth(size, quality)
        else:
            result = qpix.scaledToHeight(size, quality)
        return result

    def mouseDoubleClickEvent(self, mouse_event):
        self.open_image_dialog()

    @pyqtSlot()
    def open_image_dialog(self):
        if self.data_type != ActionImage.DATA_TYPE_IMAGE:
            return
        if not self.__connected and self.element.diagram is not None:
            self.element.diagram.element_deleted.connect(self.on_element_destroy)
            self.__connected = True
        if self.image_dialog is None:
            image = self.image_preview.get_preview_images()[self.id]
            self.image_dialog = image_preview.manager.manager.window(self.name, image=image, position='cursor')
            self.image_dialog.setImage(image)
            settings = config.ConfigWrapper.get_settings()
            if bool(strtobool(settings.get_with_default(config.VIEW_SECTION, config.PREVIEW_ON_TOP_OPTION))):
                flags = self.image_dialog.windowFlags()
                flags |= QtCore.Qt.WindowStaysOnTopHint
                self.image_dialog.setWindowFlags(flags)
                self.image_dialog.showNormal()
            self.image_dialog.installEventFilter(self)
            self.previews_container.image_dialogs_count += 1

    def prepare_actions(self, enable=True):
        if enable:
            self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
            action = QAction('Open in window', self)
            action.triggered.connect(self.open_image_dialog)
            self.addAction(action)
        else:
            self.removeAction(self.actions()[0])

    def close_image_dialog(self):
        if self.image_dialog is not None:
            self.image_dialog.close()
            self.image_dialog = None
            self.previews_container.image_dialogs_count -= 1
            if self.previews_container.image_dialogs_count < 0:
                self.previews_container.image_dialogs_count = 0
                # todo: report error here: reaching here would mean unexpected circumstances in counting spawning and closing dialogs

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.Close and self.image_dialog is not None:
            self.close_image_dialog()
            return True
        return False

    @pyqtSlot(Element)
    def on_element_destroy(self, element):
        if element is self.element and self.image_dialog is not None:
            self.close_image_dialog()

    def deleteLater(self):
        QObject.deleteLater(self)
        if self.image_dialog is not None:
            self.close_image_dialog()


class NumberOutputHelper:

    def __init__(self):
        true_ = np.ones((11, 11, 4), dtype=np.uint8) * 255
        cv2.line(true_, (5, 3), (5, 7), color=(0, 0, 0, 255))
        cv2.line(true_, (4, 4), (4, 4), color=(0, 0, 0, 255))
        self.true_ = image_preview.array_to_pixmap(true_)
        false_ = np.zeros((11, 11, 4), dtype=np.uint8) + np.array([0, 0, 0, 255], dtype=np.uint8)
        cv2.rectangle(false_, (4, 3), (6, 7), color=(255, 255, 255, 255), thickness=1)
        self.false_ = image_preview.array_to_pixmap(false_)

    def get_output(self, value):
        if isinstance(value, bool):
            return self.true_ if value else self.false_


class ElementStatusBar(StyledWidget):
    def __init__(self, element):
        super(ElementStatusBar, self).__init__()
        self.element = element
        # self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.default_message = "Element unset."
        self.message = QLabel(self.default_message)
        self.message.setWordWrap(True)
        self.message.setObjectName("ElementStatusLabel")
        # self.message.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        self.timings = QLabel("")
        self.timings.setVisible(False)
        self.timings.setObjectName("ElementStatusLabel")
        self.timings.setAlignment(QtCore.Qt.AlignRight)
        # self.timings.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        hb = QHBoxLayout()
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        hb.addWidget(self.message)
        hb.addWidget(self.timings)
        self.setLayout(hb)
        self.element.state_changed.connect(self.update)

    def set_status(self, message=""):
        # Todo: the below is slow enough to slightly freeze the GUI; QT SIGNALS (state_changed) might be the cause
        if message != "":
            self.message.setText(message)
        else:
            self.message.setText(self.default_message)
            # Todo: self.message.adjustSize() might be required here
        self.display_processing_time()

    def display_processing_time(self):
        if not hasattr(self.element, "processing_time_info"): return  # fixme: tymczasowy hack...
        time_info = self.element.processing_time_info
        if time_info is None: return
        if self.element.state == self.element.STATE_READY and time_info is not None:
            text = self.prepare_time_label()
            tooltip = self.prepare_tooltip()
            self.timings.setText(text)
            self.timings.setToolTip(tooltip)
            self.timings.setVisible(True)
        else:
            self.timings.setVisible(False)

    def prepare_time_label(self):
        time_info = self.element.processing_time_info
        unit_element_time = self.get_text_for_milis(time_info.work_time_per_unit)
        unit_total_time = self.get_text_for_milis(time_info.total_work_time_per_unit)
        return unit_element_time + " / " + unit_total_time

    def prepare_tooltip(self):
        time_info = self.element.processing_time_info
        element_time = self.get_text_for_milis(time_info.work_time)
        total_time = self.get_text_for_milis(time_info.total_work_time)
        if time_info.work_time == time_info.work_time_per_unit:
            tooltip = "%s - Element processing time\n%s - Total processing time" % (element_time, total_time)
            return tooltip
        else:
            unit_element_time = self.get_text_for_milis(time_info.work_time_per_unit)
            unit_total_time = self.get_text_for_milis(time_info.total_work_time_per_unit)
            tooltip = "%s - Unit element processing time\n%s - Unit total processing time\n%s - Element processing time\n%s - Total processing time" \
                      % (unit_element_time, unit_total_time, element_time, total_time)
            return tooltip

    def get_text_for_milis(self, milis):
        return (str(milis) + " ms") if milis < 1000 else ("%.2f" % (milis / 1000.0) + " s")

    @pyqtSlot()
    def update(self):
        if not hasattr(self, "element"): return  # fixme: tymczasowy hack, bo leci w tym miejscu wyjątek, nie wiem czemu!
        self.element.state_notified = False  # todo: this must be called by all slots connected to self.element.state_changed
        self.set_status(self.element.message)
