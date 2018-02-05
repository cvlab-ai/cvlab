# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from six import itervalues

import threading
from os import path
from threading import Lock, Event, Thread
import numpy as np
import cv2 as cv

from PyQt4.QtCore import Qt, QObject, QThread, pyqtSlot, pyqtSignal
from PyQt4.QtGui import QContextMenuEvent, QFileDialog, QKeyEvent, QLabel, QMainWindow, QMenu, QMessageBox, \
    QMouseEvent, QPixmap, QScrollArea, QSizePolicy, QWheelEvent, QApplication, QColor, QTransform, QCursor, QImage


def array_to_pixmap(arr):
    arr = np.array(arr)

    if len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[2] == 1):
        arr = cv.cvtColor(arr, cv.COLOR_GRAY2BGRA)
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv.cvtColor(arr, cv.COLOR_BGR2BGRA)

    if arr.dtype in (np.float, np.float16, np.float32, np.float64):
        arr = arr * 255
    elif arr.dtype in (np.int16, np.uint16):
        arr = arr // 256

    if arr.min() < 0:
        arr = arr // 2 + 127

    arr = arr.clip(0, 255).astype(np.uint8)

    image = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_ARGB32)
    return QPixmap.fromImage(image)


KEY_MOUSE = -1


class PreviewScrollArea(QScrollArea):
    def __init__(self):
        QScrollArea.__init__(self)
        self.lastMouse = None

    def wheelEvent(self, event):
        assert isinstance(event, QWheelEvent)
        event.ignore()

    def mouseMoveEvent(self, event):
        assert isinstance(event, QMouseEvent)
        if self.lastMouse and self.lastMouse != event.pos():
            delta = event.pos() - self.lastMouse
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        self.lastMouse = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.lastMouse = None
        event.accept()


class PreviewWindow(QMainWindow):
    minsize = (32, 32)
    maxsize = (1024, 1024)
    last_save_dir = ""

    key_signal = pyqtSignal(int, int, int)

    def __init__(self, parent, name, image=None, message=None, position=None, size=None, high_quality=False):
        super(PreviewWindow, self).__init__(parent)
        self.setObjectName("Preview window {}".format(name))
        self.setWindowTitle(name)

        desktop = QApplication.instance().desktop()
        self.maxsize = desktop.screenGeometry(desktop.screenNumber(self)).size() * 0.95

        self.setMinimumSize(*self.minsize)
        self.setMaximumSize(self.maxsize)

        self.image = None
        self.original = None
        self.message = message
        self.scale = 1.
        self.rotation = 0
        self.quality = Qt.SmoothTransformation if high_quality else Qt.FastTransformation
        self.fixed_size = size

        self.scrollarea = PreviewScrollArea()
        self.scrollarea.setFrameStyle(0)
        self.scrollarea.setFocusPolicy(Qt.NoFocus)

        self.setCentralWidget(self.scrollarea)

        self.preview = QLabel()
        self.preview.setMouseTracking(False)
        self.preview.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.scrollarea.setWidget(self.preview)

        self.setImage(image, show=False)

        if image is not None and not size:
            size = self.autoSize()

        if size:
            self.resize(*size)

        if position == 'cursor':
            position = (QCursor.pos().x() - self.size().width()//2, QCursor.pos().y() - self.size().height()//2)

        if position:
            self.move(*position)

        self.showNormal()

    def setImage(self, image, show=True, scale=None):
        if image is None: return
        self.original = image
        if isinstance(image, QImage):
            image = QPixmap.fromImage(image)
        elif isinstance(image, QPixmap):
            pass
        else:
            image = array_to_pixmap(image)
        self.image = image
        if not self.fixed_size:
            size = self.autoSize()
            self.resize(*size)
        if not scale:
            scale = self.scale
        self.setZoom(scale)
        if show:
            self.setShown(True)
            self.raise_()

    def closeEvent(self, event):
        self.scale = 1.0
        self.fixed_size = None

    def setImageAndParams(self, image, show=True, scale=None, position=None, size=None):
        if size:
            self.fixed_size = size
        if position:
            self.move(*position)
        self.setImage(image, show=show, scale=scale)

    def setZoom(self, scale):
        self.setParams(scale=scale)

    def setRotation(self, rotation):
        self.setParams(rotation=rotation)

    def setParams(self, scale=None, rotation=None):
        assert isinstance(self.image, QPixmap)

        if scale is None: scale = self.scale
        if rotation is None: rotation = self.rotation

        if scale != 1.0 or rotation:
            transform = QTransform().rotate(rotation).scale(scale,scale)
            pixmap = self.image.transformed(transform, self.quality)
        else:
            pixmap = self.image

        w = pixmap.width()
        h = pixmap.height()

        self.scale = scale
        self.rotation = rotation
        self.preview.setPixmap(pixmap)
        self.preview.setFixedSize(pixmap.size())

        if not self.fixed_size:
            self.resize(w, h)

    def autoSize(self):
        size = self.image.size()
        w = size.width()
        h = size.height()
        return w, h

    def wheelEvent(self, event):
        assert isinstance(event, QWheelEvent)
        event.accept()

        if event.delta() > 0:
            s = 1.1
        elif event.delta() < 0:
            s = 1 / 1.1
        self.setZoom(s * self.scale)

        scrollX = self.scrollarea.horizontalScrollBar().value()
        posX = event.x()
        newX = s * (scrollX + posX) - posX
        self.scrollarea.horizontalScrollBar().setValue(int(newX))

        scrollY = self.scrollarea.verticalScrollBar().value()
        posY = event.y()
        newY = s * (scrollY + posY) - posY
        self.scrollarea.verticalScrollBar().setValue(int(newY))

    def mousePressEvent(self, event):
        assert isinstance(event, QMouseEvent)
        self.key_signal.emit(KEY_MOUSE, event.x(), event.y())
        event.accept()

    def keyPressEvent(self, event):
        assert isinstance(event, QKeyEvent)
        self.key_signal.emit(int(event.key()), 0, 0)
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()

    def contextMenuEvent(self, event):
        assert isinstance(event, QContextMenuEvent)

        menu = QMenu()

        copy = menu.addAction("Copy to clipboard")

        reset = menu.addAction("Reset view")

        hq = menu.addAction("High quality")
        hq.setCheckable(True)
        if self.quality == Qt.SmoothTransformation:
            hq.setChecked(True)

        fixed = menu.addAction("Fixed size")
        fixed.setCheckable(True)
        if self.fixed_size:
            fixed.setChecked(True)

        rotate_right = menu.addAction("Rotate +")
        rotate_left = menu.addAction("Rotate -")

        save = menu.addAction("Save...")

        quit = menu.addAction("Close")

        action = menu.exec_(self.mapToGlobal(event.pos()))

        if action == quit:
            self.close()
        elif action == reset:
            self.setParams(1,0)
        elif action == hq:
            if self.quality == Qt.SmoothTransformation:
                self.quality = Qt.FastTransformation
            else:
                self.quality = Qt.SmoothTransformation
            self.setZoom(self.scale)
        elif action == rotate_right:
            rotation = (self.rotation + 90) % 360
            self.setRotation(rotation)
        elif action == rotate_left:
            rotation = (self.rotation + 270) % 360
            self.setRotation(rotation)
        elif action == save:
            filename, filter = QFileDialog.getSaveFileNameAndFilter(self, "Save image...", filter="*.png;;*.jpg;;*.bmp;;*.tiff;;*.gif", directory=self.last_save_dir)
            if filename:
                try:
                    if not str(filename).endswith(filter[1:]):
                        filename = filename + filter[1:]
                    PreviewWindow.last_save_dir = path.dirname(str(filename))
                    success = self.image.save(filename, quality=100)
                    if not success: raise Exception("unknown error")
                except Exception as e:
                    QMessageBox.critical(self, "Saving error", "Cannot save.\nError: {}".format(e.message))
                    print("Saving error:", e)
        elif action == fixed:
            if self.fixed_size:
                self.fixed_size = None
            else:
                self.fixed_size = self.size()
        elif action == copy:
            print("copy")
            clipboard = QApplication.instance().clipboard()
            clipboard.setPixmap(self.image)


class WindowManager(QObject):
    imshow_signal = pyqtSignal(dict)
    moveWindow_signal = pyqtSignal(str, int, int)

    def __init__(self):
        QObject.__init__(self)
        self.lock = Lock()
        self.windows = {}
        self.key = -1
        self.key_position = (0,0)
        self.key_lock = threading.Condition()
        self.imshow_signal.connect(self._imshow_slot)
        self.moveWindow_signal.connect(self._moveWindow)

    def imshow(self, winname, mat, **kwargs):
        args = {'winname': winname, 'mat': mat}
        args.update(kwargs)
        self.imshow_signal.emit(args)

    @pyqtSlot(dict)
    def _imshow_slot(self, kwargs):
        self._imshow(**kwargs)

    def _imshow(self, winname, mat, **kwargs):
        window = self.window(winname)
        assert isinstance(window, PreviewWindow)
        window.setImageAndParams(mat, **kwargs)

    def window(self, winname, **kwargs):
        winname = str(winname)
        with self.lock:
            if winname not in self.windows:
                position = kwargs.pop('position', 'auto')
                if position == 'auto': position = self.find_best_place()
                self.windows[winname] = PreviewWindow(None, winname, position=position, **kwargs)
                self.windows[winname].key_signal.connect(self.key_slot)
            return self.windows[winname]

    def waitKey(self, delay=0):
        while True:
            key, position = self.waitKeyMouse(delay)
            if key != KEY_MOUSE: return key

    def waitKeyMouse(self, delay=0):
        if delay <= 0: timeout = None
        else: timeout = delay * 0.001
        with self.key_lock:
            self.key = -1
            self.key_lock.wait(timeout)
            # if self.key >= 0: print "KEY:", self.key
            return self.key, self.key_position

    @pyqtSlot(int, int, int)
    def key_slot(self, key, x, y):
        with self.key_lock:
            self.key = key
            self.key_position = (x,y)
            self.key_lock.notify()

    def find_best_place(self):
        positions_x = [w.frameGeometry().x()+w.width() for w in itervalues(self.windows) if w.isVisible()]
        if not positions_x: return None
        x = max(positions_x) + 4
        y = 50
        return x, y

    def moveWindow(self, winname, x, y):
        self.moveWindow_signal.emit(winname, x, y)

    @pyqtSlot(str, int, int)
    def _moveWindow(self, winname, x, y):
        self.window(winname).move(x, y)


class ManagerThread(Thread):
    def __init__(self):
        super(ManagerThread, self).__init__()
        self.ready = Event()
        self.terminate = False
        self.daemon = True
        self.manager = None
        self.start()
        self.ready.wait()

    def run(self):
        self.manager = WindowManager()
        self.imshow = self.manager.imshow
        app = QApplication([])
        self.ready.set()
        while not self.terminate and self.is_main_thread_alive():
            app.exec_()
        app.closeAllWindows()

    def is_main_thread_alive(self):
        for i in threading.enumerate():
            if i.name == "MainThread":
                return i.is_alive()
        return None

    def imshow(self, winname, mat, **kwargs):
        return self.manager.imshow(winname, mat, **kwargs)

    def waitKey(self, delay=0):
        return self.manager.waitKey(delay)

    def waitKeyMouse(self, delay=0):
        return self.manager.waitKeyMouse(delay)

    def moveWindow(self, winname, x, y):
        return self.manager.moveWindow(winname, x, y)


class Manager(object):
    def __init__(self):
        self._manager = None
        self.lock = Lock()

    @property
    def manager(self):
        if not self._manager:
            self.create_manager()
        return self._manager

    def create_manager(self):
        with self.lock:
            if self._manager:
                return
            if QApplication.instance() is None:
                print("Creating threaded window manager")
                self._manager = ManagerThread()
            else:
                print("Creating normal window manager")
                self._manager = WindowManager()

    def imshow(self, winname, mat, **kwargs):
        return self.manager.imshow(winname, mat, **kwargs)

    def waitKey(self, time=0):
        return self.manager.waitKey(time)

    def waitKeyMouse(self, time=0):
        return self.manager.waitKeyMouse(time)

    def moveWindow(self, winname, x, y):
        return self.manager.moveWindow(winname, x, y)


manager = Manager()
imshow = manager.imshow
waitKey = manager.waitKey
moveWindow = manager.moveWindow
waitKeyMouse = manager.waitKeyMouse

def namedWindow(*a,**kw):
    pass

__ALL__ = ['imshow', 'waitKey', 'moveWindow', 'namedWindow']
