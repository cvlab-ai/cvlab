from .base import *


class TextComment(NormalElement):
    name = 'Comment'
    comment = 'Element containing text editor. Double click to edit content. ' \
              'Editor supports text formatting with HTML tags'

    def get_attributes(self):
        return [], [], [CommentParameter(id="editor", value="Double click to edit...")]


register_elements_auto(__name__, locals(), "Diagram utils", 30)
