import os
import sys
from threading import Thread

from xmlrpc.client import ServerProxy

from pip._vendor.packaging.version import parse as parse_version

from ..version import package_name, __version__


class Updater:
    pypi_url = 'https://pypi.python.org/pypi'

    def check(self):
        act_version = parse_version(__version__)

        pypi = ServerProxy(self.pypi_url)
        versions = pypi.package_releases(package_name)

        if not versions: raise Exception("There are no available releases on PyPI")

        versions = map(parse_version, versions)
        newest_version = max(versions)

        if newest_version > act_version:
            return True, newest_version
        else:
            return False, act_version

    def check_async(self, callback):
        thread = Thread(name="Updater thread", target=self._check_async, args=[callback])
        thread.setDaemon(True)
        thread.start()

    def _check_async(self, callback):
        try:
            can_update, version = self.check()
            callback(can_update, str(version))
        except Exception as e:
            print("WARNING: Cannot get update information:", e)

    def update_command(self):
        cvlab_dir = os.path.normpath(os.path.abspath(__file__) + "/../..")

        # inside git repo?
        git_path = cvlab_dir + "/../.git"
        if os.path.exists(git_path):
            repo = os.path.normpath(cvlab_dir + "/..")
            return "git -C {repo} pull".format(**locals())

        # inside virtualenv?
        if hasattr(sys, 'real_prefix'):
            pip = os.path.dirname(sys.executable) + "/pip3"
            pip = os.path.normpath(pip)
            return "{pip} install --upgrade cvlab".format(**locals())

        # inside user's pip directory?
        if os.access(__file__, os.W_OK) and ".local" in __file__:
            return "pip3 install --user --upgrade cvlab"

        # inside global pip directory, needs sudo?
        if not os.access(__file__, os.W_OK):
            return "sudo -H pip3 install --upgrade cvlab"

        # general
        return "pip3 install --upgrade cvlab"

