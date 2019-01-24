from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from pygments.formatter import Formatter
from pygments.lexers.python import PythonLexer


class Highlighter(QSyntaxHighlighter):

    def __init__(self, parent):
        QSyntaxHighlighter.__init__(self, parent)

        self.formatter = Formatter()
        self.lexer = PythonLexer()
        self.style = {}
        for token, style in self.formatter.style:
            char_format = QTextCharFormat()
            if style['color']:
                char_format.setForeground(QColor("#" + style['color']))
            if style['bgcolor']:
                char_format.setBackground(QColor("#" + style['bgcolor']))
            if style['bold']:
                char_format.setFontWeight(QFont.Bold)
            if style['italic']:
                char_format.setFontItalic(True)
            if style['underline']:
                char_format.setFontUnderline(True)

            char_format.setFontStyleHint(QFont.Monospace)
            self.style[token] = char_format

    def highlightBlock(self, text):
        position = self.currentBlock().position()
        length = self.currentBlock().length()
        text = str(self.document().toPlainText()[position:position+length])
        tokens = self.lexer.get_tokens(text)
        i = 0
        for token, text in tokens:
            self.setFormat(i, len(text), self.style[token])
            i += len(text)
