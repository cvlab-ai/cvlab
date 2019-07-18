from PyQt5 import QtGui, QtCore

from ..diagram.elements import get_sorted_elements
from .elements import *
from .mimedata import *


class Toolbox(StyledWidget):
    help = """\
Element toolbox

Drag & drop element name to add it to the active diagram"""
    filter_help = """Search for the elements by part of the name"""

    def __init__(self):
        super(Toolbox, self).__init__()
        self.setObjectName("Toolbox")
        label = QLabel("Image Processing Toolbox")
        label.setObjectName("ToolboxTitle")
        elements_list = ElementsList()
        elements_list.setObjectName("ToolboxTree")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search toolbox...")
        self.filter_input.setToolTip(self.filter_help)
        self.filter_input.textChanged.connect(elements_list.filter_changed)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self.filter_input)
        layout.addWidget(elements_list)
        self.setLayout(layout)

        self.setToolTip(self.help)


class BoldElementGroupDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        isIndexParent = not index.parent().isValid()
        if isIndexParent:
            option.font.setWeight(QFont.Bold)
        QStyledItemDelegate.paint(self, painter, option, index)


class ElementsList(QTreeView):
    def __init__(self):
        super(ElementsList, self).__init__()
        self.class_mapper = None
        elements_tree_model = self.prepare_elements_model()
        self.filter_proxy = FilterProxy(self)
        self.filter_proxy.setSourceModel(elements_tree_model)
        self.filter_proxy.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.setModel(self.filter_proxy)
        self.setItemDelegate(BoldElementGroupDelegate(self))
        self.collapseAll()
        self.header().hide()
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.last_spawned_element = 0

    def prepare_elements_model(self):
        model = QStandardItemModel()
        nodes = [(elements, QStandardItem(package)) for package, elements in get_sorted_elements()]
        element_types = [node[0] for node in nodes]
        elements_types_list = flatten_list(element_types)
        self.class_mapper = ClassStringMapper(elements_types_list)

        for elements, node in nodes:
            node.setCheckable(False)
            node.setEditable(False)
            for element in elements:
                item = QStandardItem(element.name)
                item.setEditable(False)
                item.setCheckable(False)
                row = [item,
                       QStandardItem(element.comment),
                       QStandardItem(self.class_mapper.to_string(element))]
                item.setToolTip(element.name + "\n-----------------------------------------\n" + element.comment)
                node.appendRow(row)
            model.appendRow(node)
        return model

    @pyqtSlot(str)
    def filter_changed(self, text):
        self.filter_proxy.setFilterFixedString(text)
        if text: self.expandAll()
        else: self.collapseAll()

    def is_draggable_item_selected(self):
        return len(self.selectedIndexes()) > 2

    def mousePressEvent(self, event):
        QTreeView.mousePressEvent(self, event)
        if self.is_draggable_item_selected():
            index = self.selectedIndexes()[2]
            source_selected = index.model().mapToSource(index)
            selected = self.model().sourceModel().itemFromIndex(source_selected)
            if event.button() == QtCore.Qt.LeftButton:
                drag = QDrag(self)
                mime_data = QtCore.QMimeData()
                mime_data.setText(Mime.NEW_ELEMENT)
                selected_class = self.class_mapper.to_class(str(selected.text()))
                self.last_spawned_element = selected_class()
                self.last_spawned_element.label.setMinimumWidth(100)  # workaround for drag problems - with pixmap width < 100 and adjusted HotSpot
                self.last_spawned_element.setParent(self)
                img = self.last_spawned_element.grab()
                self.last_spawned_element.label.setMinimumWidth(0)
                drag.setMimeData(mime_data)
                drag.setPixmap(img)
                drag.setHotSpot(QtCore.QPoint(img.width() / 2, img.height() / 2))
                drag.exec_()
                if self.last_spawned_element.parent() is self:
                    self.last_spawned_element.setParent(None)
                    self.last_spawned_element.deleteLater()
                else:
                    # todo: move this functionality to elements self logic
                    self.last_spawned_element.recalculate(True, True, True)
        elif event.button() == QtCore.Qt.LeftButton:
            self.setExpanded(self.currentIndex(), not self.isExpanded(self.currentIndex()))

        self.clearSelection()


class FilterProxy(QtCore.QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        if not self.filterRegExp().isEmpty():
            source_index = self.sourceModel().index(source_row, self.filterKeyColumn(), source_parent)
            if source_index.isValid():
                child_rows_count = self.sourceModel().rowCount(source_index)
                for i in range(child_rows_count):
                    if self.filterAcceptsRow(i, source_index):
                        return True
                key = self.sourceModel().data(source_index, self.filterRole())
                if hasattr(key, "toString"): key = str(key.toString())
                print(self.filterRegExp(), key)
                return self.filterRegExp().indexIn(key) >= 0
        return super(FilterProxy, self).filterAcceptsRow(source_row, source_parent)


class ClassStringMapper:
    def __init__(self, class_list):
        self.map = {}
        for class_type in class_list:
            self.map[class_type.__name__] = class_type

    def to_string(self, class_type):
        return class_type.__name__

    def to_class(self, class_string):
        return self.map[class_string]


def flatten_list(list_of_list):
    return [item for sublist in list_of_list for item in sublist]
