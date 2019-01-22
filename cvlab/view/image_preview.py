import json
import os
from datetime import datetime, timedelta

import threading
from os import path
from threading import Lock, Event, Thread
import numpy as np
import cv2 as cv

from PyQt5.QtCore import Qt, QObject, pyqtSlot, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Keys:
    NONE = -1
    MOUSE = -2

    NUM0 = 48
    NUM1 = 49
    NUM2 = 50
    NUM3 = 51
    NUM4 = 52
    NUM5 = 53
    NUM6 = 54
    NUM7 = 55
    NUM8 = 56
    NUM9 = 57
    A = 65
    Aacute = 193
    Acircumflex = 194
    acute = 180
    AddFavorite = 16777408
    Adiaeresis = 196
    AE = 198
    Agrave = 192
    Alt = 16777251
    AltGr = 16781571
    Ampersand = 38
    Any = 32
    Apostrophe = 39
    ApplicationLeft = 16777415
    ApplicationRight = 16777416
    Aring = 197
    AsciiCircum = 94
    AsciiTilde = 126
    Asterisk = 42
    At = 64
    Atilde = 195
    AudioCycleTrack = 16777478
    AudioForward = 16777474
    AudioRandomPlay = 16777476
    AudioRepeat = 16777475
    AudioRewind = 16777413
    Away = 16777464
    B = 66
    Back = 16777313
    BackForward = 16777414
    Backslash = 92
    Backspace = 16777219
    Backtab = 16777218
    Bar = 124
    BassBoost = 16777331
    BassDown = 16777333
    BassUp = 16777332
    Battery = 16777470
    Bluetooth = 16777471
    Book = 16777417
    BraceLeft = 123
    BraceRight = 125
    BracketLeft = 91
    BracketRight = 93
    BrightnessAdjust = 16777410
    brokenbar = 166
    C = 67
    Calculator = 16777419
    Calendar = 16777444
    Call = 17825796
    Camera = 17825824
    CameraFocus = 17825825
    Cancel = 16908289
    CapsLock = 16777252
    Ccedilla = 199
    CD = 16777418
    cedilla = 184
    cent = 162
    Clear = 16777227
    ClearGrab = 16777421
    Close = 16777422
    Codeinput = 16781623
    Colon = 58
    Comma = 44
    Community = 16777412
    Context1 = 17825792
    Context2 = 17825793
    Context3 = 17825794
    Context4 = 17825795
    ContrastAdjust = 16777485
    Control = 16777249
    Copy = 16777423
    copyright = 169
    currency = 164
    Cut = 16777424
    D = 68
    Dead_Abovedot = 16781910
    Dead_Abovering = 16781912
    Dead_Acute = 16781905
    Dead_Belowdot = 16781920
    Dead_Breve = 16781909
    Dead_Caron = 16781914
    Dead_Cedilla = 16781915
    Dead_Circumflex = 16781906
    Dead_Diaeresis = 16781911
    Dead_Doubleacute = 16781913
    Dead_Grave = 16781904
    Dead_Hook = 16781921
    Dead_Horn = 16781922
    Dead_Iota = 16781917
    Dead_Macron = 16781908
    Dead_Ogonek = 16781916
    Dead_Semivoiced_Sound = 16781919
    Dead_Tilde = 16781907
    Dead_Voiced_Sound = 16781918
    degree = 176
    Delete = 16777223
    diaeresis = 168
    Direction_L = 16777305
    Direction_R = 16777312
    Display = 16777425
    division = 247
    Documents = 16777427
    Dollar = 36
    DOS = 16777426
    Down = 16777237
    E = 69
    Eacute = 201
    Ecircumflex = 202
    Ediaeresis = 203
    Egrave = 200
    Eisu_Shift = 16781615
    Eisu_toggle = 16781616
    Eject = 16777401
    End = 16777233
    Enter = 16777221
    Equal = 61
    Escape = 16777216
    ETH = 208
    Excel = 16777428
    Exclam = 33
    exclamdown = 161
    Execute = 16908291
    Explorer = 16777429
    F = 70
    F1 = 16777264
    F10 = 16777273
    F11 = 16777274
    F12 = 16777275
    F13 = 16777276
    F14 = 16777277
    F15 = 16777278
    F16 = 16777279
    F17 = 16777280
    F18 = 16777281
    F19 = 16777282
    F2 = 16777265
    F20 = 16777283
    F21 = 16777284
    F22 = 16777285
    F23 = 16777286
    F24 = 16777287
    F25 = 16777288
    F26 = 16777289
    F27 = 16777290
    F28 = 16777291
    F29 = 16777292
    F3 = 16777266
    F30 = 16777293
    F31 = 16777294
    F32 = 16777295
    F33 = 16777296
    F34 = 16777297
    F35 = 16777298
    F4 = 16777267
    F5 = 16777268
    F6 = 16777269
    F7 = 16777270
    F8 = 16777271
    F9 = 16777272
    Favorites = 16777361
    Finance = 16777411
    Flip = 17825798
    Forward = 16777314
    G = 71
    Game = 16777430
    Go = 16777431
    Greater = 62
    guillemotleft = 171
    guillemotright = 187
    H = 72
    Hangul = 16781617
    Hangul_Banja = 16781625
    Hangul_End = 16781619
    Hangul_Hanja = 16781620
    Hangul_Jamo = 16781621
    Hangul_Jeonja = 16781624
    Hangul_PostHanja = 16781627
    Hangul_PreHanja = 16781626
    Hangul_Romaja = 16781622
    Hangul_Special = 16781631
    Hangul_Start = 16781618
    Hangup = 17825797
    Hankaku = 16781609
    Help = 16777304
    Henkan = 16781603
    Hibernate = 16777480
    Hiragana = 16781605
    Hiragana_Katakana = 16781607
    History = 16777407
    Home = 16777232
    HomePage = 16777360
    HotLinks = 16777409
    Hyper_L = 16777302
    Hyper_R = 16777303
    hyphen = 173
    I = 73
    Iacute = 205
    Icircumflex = 206
    Idiaeresis = 207
    Igrave = 204
    Insert = 16777222
    iTouch = 16777432
    J = 74
    K = 75
    Kana_Lock = 16781613
    Kana_Shift = 16781614
    Kanji = 16781601
    Katakana = 16781606
    KeyboardBrightnessDown = 16777398
    KeyboardBrightnessUp = 16777397
    KeyboardLightOnOff = 16777396
    L = 76
    LastNumberRedial = 17825801
    Launch0 = 16777378
    Launch1 = 16777379
    Launch2 = 16777380
    Launch3 = 16777381
    Launch4 = 16777382
    Launch5 = 16777383
    Launch6 = 16777384
    Launch7 = 16777385
    Launch8 = 16777386
    Launch9 = 16777387
    LaunchA = 16777388
    LaunchB = 16777389
    LaunchC = 16777390
    LaunchD = 16777391
    LaunchE = 16777392
    LaunchF = 16777393
    LaunchG = 16777486
    LaunchH = 16777487
    LaunchMail = 16777376
    LaunchMedia = 16777377
    Left = 16777234
    Less = 60
    LightBulb = 16777405
    LogOff = 16777433
    M = 77
    macron = 175
    MailForward = 16777467
    Market = 16777434
    masculine = 186
    Massyo = 16781612
    MediaLast = 16842751
    MediaNext = 16777347
    MediaPause = 16777349
    MediaPlay = 16777344
    MediaPrevious = 16777346
    MediaRecord = 16777348
    MediaStop = 16777345
    MediaTogglePlayPause = 16777350
    Meeting = 16777435
    Memo = 16777404
    Menu = 16777301
    MenuKB = 16777436
    MenuPB = 16777437
    Messenger = 16777465
    Meta = 16777250
    Minus = 45
    Mode_switch = 16781694
    MonBrightnessDown = 16777395
    MonBrightnessUp = 16777394
    mu = 181
    Muhenkan = 16781602
    MultipleCandidate = 16781629
    multiply = 215
    Multi_key = 16781600
    Music = 16777469
    MySites = 16777438
    N = 78
    News = 16777439
    No = 16842754
    nobreakspace = 160
    notsign = 172
    Ntilde = 209
    NumberSign = 35
    NumLock = 16777253
    O = 79
    Oacute = 211
    Ocircumflex = 212
    Odiaeresis = 214
    OfficeHome = 16777440
    Ograve = 210
    onehalf = 189
    onequarter = 188
    onesuperior = 185
    Ooblique = 216
    OpenUrl = 16777364
    Option = 16777441
    ordfeminine = 170
    Otilde = 213
    P = 80
    PageDown = 16777239
    PageUp = 16777238
    paragraph = 182
    ParenLeft = 40
    ParenRight = 41
    Paste = 16777442
    Pause = 16777224
    Percent = 37
    Period = 46
    periodcentered = 183
    Phone = 16777443
    Pictures = 16777468
    Play = 16908293
    Plus = 43
    plusminus = 177
    PowerDown = 16777483
    PowerOff = 16777399
    PreviousCandidate = 16781630
    Print = 16777225
    Printer = 16908290
    Q = 81
    Question = 63
    questiondown = 191
    QuoteDbl = 34
    QuoteLeft = 96
    R = 82
    Refresh = 16777316
    registered = 174
    Reload = 16777446
    Reply = 16777445
    Return = 16777220
    Right = 16777236
    Romaji = 16781604
    RotateWindows = 16777447
    RotationKB = 16777449
    RotationPB = 16777448
    S = 83
    Save = 16777450
    ScreenSaver = 16777402
    ScrollLock = 16777254
    Search = 16777362
    section = 167
    Select = 16842752
    Semicolon = 59
    Send = 16777451
    Shift = 16777248
    Shop = 16777406
    SingleCandidate = 16781628
    Slash = 47
    Sleep = 16908292
    Space = 32
    Spell = 16777452
    SplitScreen = 16777453
    ssharp = 223
    Standby = 16777363
    sterling = 163
    Stop = 16777315
    Subtitle = 16777477
    Super_L = 16777299
    Super_R = 16777300
    Support = 16777454
    Suspend = 16777484
    SysReq = 16777226
    T = 84
    Tab = 16777217
    TaskPane = 16777455
    Terminal = 16777456
    THORN = 222
    threequarters = 190
    threesuperior = 179
    Time = 16777479
    ToDoList = 16777420
    ToggleCallHangup = 17825799
    Tools = 16777457
    TopMenu = 16777482
    Touroku = 16781611
    Travel = 16777458
    TrebleDown = 16777335
    TrebleUp = 16777334
    twosuperior = 178
    U = 85
    Uacute = 218
    Ucircumflex = 219
    Udiaeresis = 220
    Ugrave = 217
    Underscore = 95
    unknown = 33554431
    Up = 16777235
    UWB = 16777473
    V = 86
    Video = 16777459
    View = 16777481
    VoiceDial = 17825800
    VolumeDown = 16777328
    VolumeMute = 16777329
    VolumeUp = 16777330
    W = 87
    WakeUp = 16777400
    WebCam = 16777466
    WLAN = 16777472
    Word = 16777460
    WWW = 16777403
    X = 88
    Xfer = 16777461
    Y = 89
    Yacute = 221
    ydiaeresis = 255
    yen = 165
    Yes = 16842753
    Z = 90
    Zenkaku = 16781608
    Zenkaku_Hankaku = 16781610
    Zoom = 16908294
    ZoomIn = 16777462
    ZoomOut = 16777463


KEY_NONE = Keys.NONE
KEY_MOUSE = Keys.MOUSE


def array_to_pixmap(arr):
    arr = np.array(arr)

    if arr.dtype in (np.float, np.float16, np.float32, np.float64):
        arr = arr * 255
    elif arr.dtype in (np.int16, np.uint16):
        arr = arr // 256

    arr = arr.clip(0, 255).astype(np.uint8)

    if len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[2] == 1):
        arr = cv.cvtColor(arr, cv.COLOR_GRAY2BGRA)
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv.cvtColor(arr, cv.COLOR_BGR2BGRA)

    image = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_ARGB32)
    return QPixmap.fromImage(image)


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


class PreviewWindow(QFrame):
    minsize = (32, 32)
    maxsize = None
    last_save_dir = ""
    raise_window = False

    key_signal = pyqtSignal(int, int, int)
    move_signal = pyqtSignal()

    def __init__(self, manager, name, image=None, message=None, position=None, size=None, high_quality=False):
        super(PreviewWindow, self).__init__()
        self.setObjectName("Preview window {}".format(name))
        self.setWindowTitle(name)

        self.manager = manager

        desktop = QApplication.instance().desktop()
        if self.maxsize:
            self.maxsize = QSize(*self.maxsize)
        else:
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

        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        layout.addWidget(self.scrollarea, 0, 0)

        self.preview = QLabel()
        self.preview.setMouseTracking(False)
        self.preview.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.scrollarea.setWidget(self.preview)

        self.message_label = QLabel(" ")
        self.layout().addWidget(self.message_label, 0, 0, Qt.AlignTop)
        self.message_label.setStyleSheet("QLabel {color:black;background:rgba(255,255,255,32)}")
        self.message_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.message_label.setText("")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(4)
        shadow.setColor(Qt.white)
        shadow.setOffset(0,0)
        self.message_label.setGraphicsEffect(shadow)

        self.blink_widget = QWidget()
        self.blink_widget.hide()
        self.blink_widget.setStyleSheet("border:3px solid red")
        self.blink_timer = QTimer()
        self.blink_timer.setInterval(1000)
        self.blink_timer.timeout.connect(self.blink_)
        layout.addWidget(self.blink_widget, 0, 0)

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

    def setImage(self, image, show=True, scale=None, blink=False):
        if image is None: return
        self.original = image
        if isinstance(image, QImage):
            image = QPixmap.fromImage(image)
        elif isinstance(image, QPixmap):
            pass
        else:
            image = array_to_pixmap(image)
        self.image = image
        if not scale:
            scale = self.scale
            if image.width()*scale > self.maxsize.width():
                scale = self.maxsize.width() / image.width()
            if image.height()*scale > self.maxsize.height():
                scale = self.maxsize.height() / image.height()
        self.setZoom(scale)
        if self.message is not None:
            self.message_label.setText(self.message)
        if blink:
            self.blink(True)
        if show:
            self.show()
            if self.raise_window:
                self.raise_()

    def closeEvent(self, event):
        self.scale = 1.0
        self.fixed_size = None

    def setImageAndParams(self, image, show=True, scale=None, position=None, size=None, hq=None, message=None, blink=False):
        if size:
            self.fixed_size = size
        if position:
            self.move(*position)
        if hq is not None:
            if hq: self.quality = Qt.SmoothTransformation
            else: self.quality = Qt.FastTransformation
        if message is not None:
            self.message = message
        self.setImage(image, show=show, scale=scale, blink=blink)

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

        if event.angleDelta().y() > 0:
            s = 1.1
        else:
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

        self.blink(False)

    def mousePressEvent(self, event):
        assert isinstance(event, QMouseEvent)
        self.key_signal.emit(KEY_MOUSE, event.x(), event.y())
        self.blink(False)
        event.accept()

    def keyPressEvent(self, event):
        assert isinstance(event, QKeyEvent)
        self.key_signal.emit(int(event.key()), 0, 0)
        self.blink(False)
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()

    def moveEvent(self, event):
        self.move_signal.emit()

    def contextMenuEvent(self, event):
        assert isinstance(event, QContextMenuEvent)

        self.blink(False)

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

    def blink(self, enable):
        if enable:
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            self.blink_widget.hide()

    @pyqtSlot()
    def blink_(self):
        if self.blink_widget.isHidden():
            self.blink_widget.show()
        else:
            self.blink_widget.hide()


class WindowManager(QObject):
    imshow_signal = pyqtSignal(dict)
    moveWindow_signal = pyqtSignal(str, int, int)
    positions_file = "~/.image_preview"

    def __init__(self):
        QObject.__init__(self)
        self.lock = Lock()
        self.windows = {}
        self.positions = {}
        self.positions_last_save = datetime.now()
        self.load_positions()
        self.key = KEY_NONE
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
                if position == 'auto':
                    position = self.positions.get(winname, self.find_best_place())
                window = self.windows[winname] = PreviewWindow(self, winname, position=position, **kwargs)
                window.key_signal.connect(self.key_slot)
                window.move_signal.connect(self.save_positions)
            return self.windows[winname]

    def waitKey(self, delay=0):
        while True:
            key, position = self.waitKeyMouse(delay)
            if key != KEY_MOUSE: return key

    def waitKeyMouse(self, delay=0):
        if delay <= 0: timeout = None
        else: timeout = delay * 0.001
        with self.key_lock:
            self.key = KEY_NONE
            self.key_lock.wait(timeout)
            # print("KEY", self.key, self.key_position)
            return self.key, self.key_position

    @pyqtSlot(int, int, int)
    def key_slot(self, key, x, y):
        with self.key_lock:
            self.key = key
            self.key_position = (x,y)
            self.key_lock.notify()

    def find_best_place(self):
        positions_x = [w.frameGeometry().x()+w.width() for w in self.windows.values() if w.isVisible()]
        if not positions_x: return None
        x = max(positions_x) + 4
        y = 50
        return x, y

    def moveWindow(self, winname, x, y):
        self.moveWindow_signal.emit(winname, x, y)

    @pyqtSlot(str, int, int)
    def _moveWindow(self, winname, x, y):
        self.window(winname).move(x, y)

    def load_positions(self):
        try:
            path = os.path.expanduser(self.positions_file)
            if os.path.isfile(path):
                with self.lock:
                    self.positions.update(json.load(open(path)))
        except Exception:
            print("WARN: Cannot load window positions")

    @pyqtSlot()
    def save_positions(self, force=False):
        if not force and datetime.now() - self.positions_last_save < timedelta(seconds=5):
            return
        path = os.path.expanduser(self.positions_file)
        with self.lock:
            for name, window in self.windows.items():
                assert isinstance(window, PreviewWindow)
                self.positions[name] = window.x(), window.y()
        with open(path, 'w') as f:
            json.dump(self.positions, f)
        self.positions_last_save = datetime.now()

    def __del__(self):
        self.save_positions(True)


class ManagerThread(Thread):
    def __init__(self):
        super(ManagerThread, self).__init__()
        self.ready = Event()
        self.terminate = False
        self.daemon = True
        self.manager = None
        self.app = None
        self.start()
        self.ready.wait()

    def run(self):
        self.manager = WindowManager()
        self.app = QApplication([])
        self.ready.set()
        while not self.terminate and self.is_main_thread_alive():
            self.app.exec_()
        self.app.closeAllWindows()

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


class Manager:
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


__ALL__ = ['imshow', 'waitKey', 'waitKeyMouse', 'moveWindow', 'namedWindow']
