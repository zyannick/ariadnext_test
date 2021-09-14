# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Qt5Agg')

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.resize(1336, 655)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 621, 591))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_ref = QtWidgets.QLabel(self.frame)
        self.label_ref.setGeometry(QtCore.QRect(20, 20, 221, 17))
        self.label_ref.setObjectName("label_ref")
        self.image_person1 = QtWidgets.QLabel(self.frame)
        self.image_person1.setGeometry(QtCore.QRect(20, 70, 151, 101))
        self.image_person1.setObjectName("image_person1")
        self.nom1 = QtWidgets.QLineEdit(self.frame)
        self.nom1.setGeometry(QtCore.QRect(220, 100, 113, 25))
        self.nom1.setObjectName("nom1")
        self.aj1 = QtWidgets.QPushButton(self.frame)
        self.aj1.setGeometry(QtCore.QRect(390, 100, 88, 25))
        self.aj1.setObjectName("aj1")
        self.image_person2 = QtWidgets.QLabel(self.frame)
        self.image_person2.setGeometry(QtCore.QRect(20, 220, 151, 101))
        self.image_person2.setObjectName("image_person2")
        self.nom2 = QtWidgets.QLineEdit(self.frame)
        self.nom2.setGeometry(QtCore.QRect(220, 240, 113, 25))
        self.nom2.setObjectName("nom2")
        self.aj2 = QtWidgets.QPushButton(self.frame)
        self.aj2.setGeometry(QtCore.QRect(390, 240, 88, 25))
        self.aj2.setObjectName("aj2")
        self.image_person3 = QtWidgets.QLabel(self.frame)
        self.image_person3.setGeometry(QtCore.QRect(20, 380, 151, 101))
        self.image_person3.setObjectName("image_person3")
        self.nom3 = QtWidgets.QLineEdit(self.frame)
        self.nom3.setGeometry(QtCore.QRect(220, 410, 113, 25))
        self.nom3.setObjectName("nom3")
        self.aj3 = QtWidgets.QPushButton(self.frame)
        self.aj3.setGeometry(QtCore.QRect(390, 410, 88, 25))
        self.aj3.setObjectName("aj3")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(640, 10, 681, 591))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.lab_vid_img = QtWidgets.QLabel(self.frame_2)
        self.lab_vid_img.setGeometry(QtCore.QRect(30, 20, 141, 17))
        self.lab_vid_img.setObjectName("lab_vid_img")
        self.vid_img = QtWidgets.QFrame(self.frame_2)
        self.vid_img.setGeometry(QtCore.QRect(70, 80, 521, 431))
        self.vid_img.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.vid_img.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vid_img.setObjectName("vid_img")
        self.up_vid_img = QtWidgets.QPushButton(self.frame_2)
        self.up_vid_img.setGeometry(QtCore.QRect(70, 550, 201, 25))
        self.up_vid_img.setObjectName("up_vid_img")
        self.webcam_button = QtWidgets.QPushButton(self.frame_2)
        self.webcam_button.setGeometry(QtCore.QRect(447, 550, 141, 25))
        self.webcam_button.setObjectName("webcam_button")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1336, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.aj1.clicked.connect(lambda: self.get_cropped(1))
        self.aj2.clicked.connect(lambda: self.get_cropped(2))
        self.aj3.clicked.connect(lambda: self.get_cropped(3))



        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.show()

    @pyqtSlot(np.ndarray)
    def update_spectre(self, image):
        qimage = QImage(image, image.shape[1], image.shape[0],
                        QImage.Format_RGB888)
        self.label_spectre.setPixmap(QPixmap.fromImage(qimage))

    @pyqtSlot(np.ndarray)
    def update_spectre(self, image):
        qimage = QImage(image, image.shape[1], image.shape[0],
                        QImage.Format_RGB888)
        self.label_spectre.setPixmap(QPixmap.fromImage(qimage))

    def afficher_images(self):

        self.thread = VideoThread(self.selected_file)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.vid_img)
        # start the thread
        self.thread.start()


    def get_video(self, id_button):
        self.selected_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                 '.', "Videos files")

        self.selected_file, _ = self.selected_file

    def get_cropped(self, id_button):
        self.selected_image = QFileDialog.getOpenFileName(self, 'Open file',
                                                 '.', "Cropped faces (*)")

        self.selected_image, _ = self.selected_image
        image = cv2.imread(self.selected_image, 1)
        image = cv2.resize(image, dsize=(100, 100))
        if id_button==1:
            qimage = QImage(image, image.shape[1], image.shape[0],
                            QImage.Format_RGB888)
            self.image_person1.setPixmap(QPixmap.fromImage(qimage))


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_ref.setText(_translate("MainWindow", "Listes references images"))
        self.image_person1.setText(_translate("MainWindow", "image person 1"))
        self.nom1.setText(_translate("MainWindow", "Person1"))
        self.aj1.setText(_translate("MainWindow", "Ajouter"))
        self.image_person2.setText(_translate("MainWindow", "image person 2"))
        self.nom2.setText(_translate("MainWindow", "Person2"))
        self.aj2.setText(_translate("MainWindow", "Ajouter"))
        self.image_person3.setText(_translate("MainWindow", "image person 3"))
        self.nom3.setText(_translate("MainWindow", "Person3"))
        self.aj3.setText(_translate("MainWindow", "Ajouter"))
        self.lab_vid_img.setText(_translate("MainWindow", "Video/images"))
        self.up_vid_img.setText(_translate("MainWindow", "Upload Video/Image"))
        self.webcam_button.setText(_translate("MainWindow", "Play"))
