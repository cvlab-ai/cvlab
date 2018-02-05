from __future__ import unicode_literals


class GeneralException(Exception):
    pass


class ConnectError(GeneralException):
    pass


class ProcessingBreak(GeneralException):
    pass


class ProcessingError(GeneralException):
    pass


