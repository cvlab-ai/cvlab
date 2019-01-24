from .version import __version__


def main(*args, **kwargs):
    import os
    import sys
    import sip
    import numpy as np

    sys._excepthook = sys.excepthook
    def exception_hook(exctype, value, traceback):
        sys._excepthook(exctype, value, traceback)
        sys.exit(-1)
    sys.excepthook = exception_hook

    # todo: it's an ugly workaround for PyQt stylesheets relative paths
    try:
        os.chdir(os.path.dirname(__file__))
    except Exception:
        pass

    np.seterr(all='raise')
    sip.setdestroyonexit(False)

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    from .view.mainwindow import MainWindow
    main_window = MainWindow(app)
    ret_code = app.exec_()
    return ret_code


if __name__ == '__main__':
    main()
