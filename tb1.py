

import sys
import os
import cv2 as cv
from PyQt6 import QtGui as QtGui
from PyQt6.QtGui import * 
from PyQt6.QtWidgets import  *
# from PyQt6 import QtWidgets as Qt
import PyQt6.QtCore as Qt
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as cm


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.imgPath = r"D:\PID\TB1PID\TB1PID\peppers.png"
        layout = QGridLayout()
        self.setWindowTitle("QGridLayout")

        self.firstParameter = QLineEdit()
        self.secondParameter = QLineEdit()
        formLayout = QFormLayout()
        formLayout.addRow("1st Parameter", self.firstParameter)
        formLayout.addRow("2nd Parameter", self.secondParameter)
        layout.addLayout(formLayout, 0, 0, 1, 3)
        
        originalImageButton = QPushButton("Original Image")
        originalImageButton.clicked.connect(self.originalImage)
        layout.addWidget(originalImageButton, 0, 4)
        
        grayscaleButton = QPushButton("Grayscale")
        grayscaleButton.clicked.connect(self.grayscale)
        layout.addWidget(grayscaleButton, 1, 0)
        
        robertsButton = QPushButton("Roberts")
        robertsButton.clicked.connect(self.roberts)
        layout.addWidget(robertsButton, 1, 1)

        thresholdButton = QPushButton("Threshold")
        thresholdButton.clicked.connect(self.threshold)
        layout.addWidget(thresholdButton, 1, 3)

        basicLowPassButton = QPushButton("Basic Low Pass")
        basicLowPassButton.clicked.connect(self.basicLowPass)
        layout.addWidget(basicLowPassButton, 1, 4)

        layout.addWidget(QPushButton("Sobel"), 1, 2)
        layout.addWidget(QPushButton("Prewitt"), 2, 0)
        layout.addWidget(QPushButton("Log"), 2, 1)
        layout.addWidget(QPushButton("Poisson"), 2, 2)
        layout.addWidget(QPushButton("Speckle"), 3, 0)
        layout.addWidget(QPushButton("Watershed"), 3, 1)
        layout.addWidget(QPushButton("Canny"), 3, 2)
        self.image = QLabel()
        self.image.height = 640
        self.image.width = 480
        self.pixmap = QPixmap(480, 640)
        
        self.image.setPixmap(self.pixmap)
        layout.addWidget(self.image, 4, 0, 4, 4)
        self.setLayout(layout)

    def cmap2pixmap(self, cmap, steps=50):
        """Convert a maplotlib colormap into a QPixmap
        :param cmap: The colormap to use
        :type cmap: Matplotlib colormap instance (e.g. matplotlib.cm.gray)
        :param steps: The number of color steps in the output. Default=50
        :type steps: int
        :rtype: QPixmap
        """
        
        sm = cm.ScalarMappable(cmap=cmap)
        sm.norm.vmin = 0.0
        sm.norm.vmax = 1.0
        inds = np.linspace(0, 1, steps)
        rgbas = sm.to_rgba(inds)
        rgbas = [QColor(int(r * 255), int(g * 255),
                        int(b * 255), int(a * 255)).rgba() for r, g, b, a in rgbas]
        im = QImage(steps, 1, QImage.Format_Indexed8)
        im.setColorTable(rgbas)
        for i in range(steps):
            im.setPixel(i, 0, i)
        im = im.scaled(100, 100)
        pm = QPixmap.fromImage(im)
        return pm

    def originalImage(self):
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, _ = img.shape        
        convert_to_Qt_format = QtGui.QImage(img, w, h, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def grayscale(self):
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print(gray_image)

        h, w, _ = img.shape        
        convert_to_Qt_format = QtGui.QImage(gray_image.data, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def roberts(self):
        #TODO
        roberts_cross_v = np.array( [[1, 0 ],
                                    [0,-1 ]] )
  
        roberts_cross_h = np.array( [[ 0, 1 ],
                                    [ -1, 0 ]] )
        
        img = cv.imread(self.imgPath, 0).astype('float64')
        # gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # gray_image = gray_image.astype('float64')
        img/=255.0
        vertical = ndimage.convolve(img, roberts_cross_v)
        horizontal = ndimage.convolve(img, roberts_cross_h)
        edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
        edged_img*=255
        h, w = edged_img.shape        
        for i, x in enumerate(edged_img):
            for j, y in enumerate(x):
                edged_img[i][j] = int(edged_img[i][j])
        print(edged_img)
        
        convert_to_Qt_format = QtGui.QImage(edged_img.data, w, h, QtGui.QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))
        # cv.imshow(edged_img)
    
    def threshold(self):
        parameter = self.firstParameter.text()
        if parameter != '':
            img = cv.imread(self.imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgToProcess = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            _,thresh = cv.threshold(imgToProcess,255*float(parameter),255,cv.THRESH_BINARY)
            h, w = thresh.shape
            convert_to_Qt_format = QtGui.QImage(thresh, w, h, QtGui.QImage.Format.Format_Grayscale8)
            p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(QPixmap.fromImage(p))

    def basicLowPass(self):
        parameter = int(self.secondParameter.text())
        if parameter != '':
            img = cv.imread(self.imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            kernel = np.ones((parameter,parameter),np.float32)/(parameter*parameter)
            img2 = cv.filter2D(img,ddepth=-1,kernel=kernel)
            h, w, _ = img2.shape
            convert_to_Qt_format = QtGui.QImage(img2, w, h, QtGui.QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(QPixmap.fromImage(p))


# layout.addWidget(
#     QPushButton("Button (2, 1) + 2 Columns Span"), 3, 1, 1, 3
# )
app = QApplication([])
window = Window()

window.show()
sys.exit(app.exec())
# app.exec()

