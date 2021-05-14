import os
import sys

from .version import __version__

CVLAB_DIR = os.path.abspath(__file__ + "/..").replace("\\", "/")


def init_libs():
    # system exception hook
    try:
        sys._excepthook = sys.excepthook
        def exception_hook(exctype, value, traceback):
            sys._excepthook(exctype, value, traceback)
            sys.exit(-1)
        sys.excepthook = exception_hook
    except Exception:
        pass

    # numpy error exceptions
    try:
        import numpy as np
        np.seterr(all='raise')
    except Exception:
        pass

    # PyQt exceptions workaround
    try:
        try:
            from PyQt5 import sip
        except ImportError:
            import sip
        sip.setdestroyonexit(False)
    except Exception:
        pass


def splash():
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QSplashScreen, QLabel, QDesktopWidget
    from PyQt5.QtCore import QTimer

    pixmap = QPixmap(CVLAB_DIR + "/images/splash.png")
    splash = QSplashScreen(pixmap)

    splash.show()
    splash.raise_()

    QTimer.singleShot(5000, splash.close)

    return splash


def main(*args, **kwargs):
    init_libs()

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    s = splash()

    from .view.mainwindow import MainWindow
    main_window = MainWindow(app)
    ret_code = app.exec_()
    return ret_code


if __name__ == '__main__':
    exit(main())
