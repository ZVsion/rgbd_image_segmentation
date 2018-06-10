import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import GraphMaker

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.graph_maker = GraphMaker.GraphMaker()
        self.seed_type = 1  #annotation type

        self.initUI()

    def initUI(self):
        self.resize(1000, 600)

        clearButton = QPushButton("Clear All Seeds")
        clearButton.setStyleSheet("background-color:white")
        clearButton.clicked.connect(self.on_clear)

        segmentButton = QPushButton("Segment Image")
        segmentButton.setStyleSheet("background-color:white")
        segmentButton.clicked.connect(self.on_segment)

        StateLine = QLabel()
        StateLine.setText("Click or Drag Left Mouse Button for Foreground Annotation,and Right Mouse Button for Background.")
        palette = QPalette()
        palette.setColor(StateLine.foregroundRole(), Qt.red)
        StateLine.setPalette(palette)

        hbox = QHBoxLayout()
        hbox.addWidget(clearButton)
        hbox.addWidget(segmentButton)
        hbox.addStretch(1)
        hbox.addWidget(StateLine)
        hbox.addStretch(1)

        self.seedLabel = QLabel()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        self.seedLabel.setAlignment(Qt.AlignCenter)
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag

        self.segmentLabel = QLabel()
        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.segmented))))
        self.segmentLabel.setAlignment(Qt.AlignCenter)

        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(False)
        scroll1.setWidget(self.seedLabel)
        scroll1.setAlignment(Qt.AlignCenter)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(False)
        scroll2.setWidget(self.segmentLabel)
        scroll2.setAlignment(Qt.AlignCenter)

        imagebox = QHBoxLayout()
        imagebox.addWidget(scroll1)
        imagebox.addWidget(scroll2)

        vbox = QVBoxLayout()

        vbox.addLayout(hbox)
        vbox.addLayout(imagebox)

        self.setLayout(vbox)

        self.setWindowTitle('Buttons')
        self.show()
    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def mouse_down(self, event):
        if event.button() == Qt.LeftButton:
            self.seed_type = 1
        elif event.button() == Qt.RightButton:
            self.seed_type = 0
        print(str(event.x()) + "," + str(event.y()))
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    def mouse_drag(self, event):
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    @pyqtSlot()
    def on_segment(self):
        # caculate
        self.graph_maker.create_graph()

        self.segmentLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.segmented))))

    @pyqtSlot()
    def on_clear(self):
        self.graph_maker.clear_seeds()
        self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())