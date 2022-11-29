

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
# from tkinter import tk    
from tkinter.filedialog import askopenfilename
import random


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.imgPath = ''
        layout = QGridLayout()
        self.setWindowTitle("QGridLayout")

        self.firstParameter = QLineEdit()
        self.secondParameter = QLineEdit()
        formLayout = QFormLayout()
        formLayout.addRow("1st Parameter", self.firstParameter)
        formLayout.addRow("2nd Parameter", self.secondParameter)
        layout.addLayout(formLayout, 0, 0, 1, 3)
        
        originalImageButton = QPushButton("Select Image")
        originalImageButton.clicked.connect(self.originalImage)
        layout.addWidget(originalImageButton, 0, 4)
        
        grayscaleButton = QPushButton("Grayscale")
        grayscaleButton.clicked.connect(self.grayscale)
        layout.addWidget(grayscaleButton, 1, 0)
        
        robertsButton = QPushButton("Roberts")
        robertsButton.clicked.connect(self.roberts)
        layout.addWidget(robertsButton, 1, 1)

        sobelButton = QPushButton("Sobel")
        sobelButton.clicked.connect(self.sobel)
        layout.addWidget(sobelButton, 1, 2)

        thresholdButton = QPushButton("Threshold")
        thresholdButton.clicked.connect(self.threshold)
        layout.addWidget(thresholdButton, 1, 3)

        basicLowPassButton = QPushButton("Basic Low Pass")
        basicLowPassButton.clicked.connect(self.basicLowPass)
        layout.addWidget(basicLowPassButton, 1, 4)

        prewittButton = QPushButton("Prewitt")
        prewittButton.clicked.connect(self.prewitt)
        layout.addWidget(prewittButton, 2, 0)

        logButton = QPushButton("LoG")
        logButton.clicked.connect(self.log)
        layout.addWidget(logButton, 2, 1)

        poissonButton = QPushButton("Poisson")
        poissonButton.clicked.connect(self.poisson)
        layout.addWidget(poissonButton, 2, 2)

        cannyButton = QPushButton("Canny")
        cannyButton.clicked.connect(self.canny)
        layout.addWidget(cannyButton, 2, 3)

        medianLowPassButton = QPushButton("Median Low Pass")
        medianLowPassButton.clicked.connect(self.medianLowPass)
        layout.addWidget(medianLowPassButton, 2, 4)

        saltAndPepperButton = QPushButton("Salt & Pepper")
        saltAndPepperButton.clicked.connect(self.saltAndPepper)
        layout.addWidget(saltAndPepperButton, 3, 2)
        # layout.addWidget(QPushButton("Canny"), 3, 2)
        self.image = QLabel()
        self.image.height = 640
        self.image.width = 480
        self.pixmap = QPixmap(480, 640)
        
        self.image.setPixmap(self.pixmap)
        layout.addWidget(self.image, 4, 0, 4, 4)
        self.setLayout(layout)

    def originalImage(self):
        currImage = self.imgPath
        while currImage == self.imgPath:
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, _ = img.shape        
        convert_to_Qt_format = QtGui.QImage(img, w, h, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def grayscale(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        print(gray_image)

        h, w, _ = img.shape        
        convert_to_Qt_format = QtGui.QImage(gray_image.data, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def roberts(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        #TODO
        roberts_cross_v = np.array( [[0, 0, -1],
                                    [0, 1, 0],
                                    [0, 0, 0]] )
        
        roberts_cross_h = np.array( [[-1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]] )
        
        img = cv.imread(self.imgPath,0)
        vertical = cv.filter2D(img, -1, roberts_cross_v)
        horizontal = cv.filter2D(img, -1, roberts_cross_h)
        edged_img = horizontal + vertical
        h, w = edged_img.shape        
        convert_to_Qt_format = QtGui.QImage(edged_img.data, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))
        # cv.imshow(edged_img)
    
    def canny(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_canny = cv.Canny(gray, 100, 200)
        h, w = img_canny.shape
        convert_to_Qt_format = QtGui.QImage(img_canny, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def prewitt(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()

        img = cv.imread(self.imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gaussian = cv.GaussianBlur(gray,(3,3),0)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

        img_prewittx = cv.filter2D(img_gaussian, -1, kernelx)
        img_prewitty = cv.filter2D(img_gaussian, -1, kernely)
        img_prewitt = img_prewittx + img_prewitty
        h, w = img_prewitt.shape
        convert_to_Qt_format = QtGui.QImage(img_prewitt, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def sobel(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gaussian = cv.GaussianBlur(gray,(3,3),0)
        img_sobelx = cv.Sobel(img_gaussian,cv.CV_8U,1,0,ksize=3)
        img_sobely = cv.Sobel(img_gaussian,cv.CV_8U,0,1,ksize=3)
        img_sobel = img_sobelx + img_sobely
        h, w = img_sobel.shape
        convert_to_Qt_format = QtGui.QImage(img_sobel, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def poisson(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        noisy = np.random.poisson(img)
        h, w, ch = noisy.shape
        convert_to_Qt_format = QtGui.QImage(noisy.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
        p = QtGui.QPixmap(convert_to_Qt_format)
        self.image.setPixmap(p)
    
    def threshold(self):
        parameter = self.firstParameter.text()
        if parameter != '':
            while self.imgPath == '':
                self.imgPath = askopenfilename()
            img = cv.imread(self.imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgToProcess = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            _,thresh = cv.threshold(imgToProcess,255*float(parameter),255,cv.THRESH_BINARY)
            h, w = thresh.shape
            convert_to_Qt_format = QtGui.QImage(thresh, w, h, QtGui.QImage.Format.Format_Grayscale8)
            p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(QPixmap.fromImage(p))

    def basicLowPass(self):
        parameter = self.secondParameter.text()
        if parameter != '':
            parameter = int(parameter)
            while self.imgPath == '':
                self.imgPath = askopenfilename()
            img = cv.imread(self.imgPath)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            kernel = np.ones((parameter,parameter),np.float32)/(parameter*parameter)
            img2 = cv.filter2D(img,ddepth=-1,kernel=kernel)
            h, w, _ = img2.shape
            convert_to_Qt_format = QtGui.QImage(img2, w, h, QtGui.QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
            self.image.setPixmap(QPixmap.fromImage(p))

    def medianLowPass(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)    
        grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img2 = cv.medianBlur(grayImg, 5)
        h, w = img2.shape
        convert_to_Qt_format = QtGui.QImage(img2, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def saltAndPepper(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


        row, col = img.shape
            
        number_of_pixels = random.randint(300, 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 255
                
        number_of_pixels = random.randint(300 , 10000)
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = 0
        
        h, w = img.shape
        convert_to_Qt_format = QtGui.QImage(img, w, h, QtGui.QImage.Format.Format_Grayscale8)
        p = convert_to_Qt_format.scaled(w, h, Qt.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(QPixmap.fromImage(p))

    def log(self):
        while self.imgPath == '':
            self.imgPath = askopenfilename()
        img = cv.imread(self.imgPath)    
        blur = cv.GaussianBlur(img,(3,3),0)
        laplacian = cv.Laplacian(blur,cv.CV_64F)
        laplacian1 = laplacian/laplacian.max()
        imgFinal = cv.imwrite(filename='.\\saida.png', img=laplacian1)
        img = cv.imread('.\\saida.png',0)
        h, w = img.shape
        convert_to_Qt_format = QtGui.QImage(img, w, h, QtGui.QImage.Format.Format_Grayscale8)
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

