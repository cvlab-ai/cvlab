from .version import __version__


def main(*args, **kwargs):
    import os
    import sys
    import sip
    import numpy as np

    # todo: it's an ugly workaround for PyQt stylesheets relative paths
    try:
        os.chdir(os.path.dirname(__file__))
    except Exception:
        pass

    from PyQt4 import QtGui
    from .view.mainwindow import MainWindow

    np.seterr(all='raise')
    sip.setdestroyonexit(False)

    app = QtGui.QApplication(sys.argv)
    main_window = MainWindow(app)
    ret_code = app.exec_()
    return ret_code
