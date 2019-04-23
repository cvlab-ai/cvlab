import random
import string
from threading import RLock
from weakref import WeakValueDictionary


objects = WeakValueDictionary()
lock = RLock()


def next_id(object2add=None):
    with lock:
        if not objects:
            id = 1
        else:
            id = max(objects.keys()) + 1
        if object2add is not None: objects[id] = object2add
        return id


def change_id(old, new):
    with lock:
        if old not in objects: return
        if new in objects: raise ValueError("ID actualy exists in ID manager")
        objects[new] = objects[old]
        objects.pop(old)


# sure, it's pseudo unique id
def unique_id(digits=6):
    valid_chars = list(set(string.digits.lower()))
    entropy_string = ''
    for i in range(digits):
        entropy_string += random.choice(valid_chars)
    return entropy_string
