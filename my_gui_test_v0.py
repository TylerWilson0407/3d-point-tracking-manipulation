# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'my_gui_test_v0.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1406, 752)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_camfeed1 = QtWidgets.QLabel(self.centralwidget)
        self.label_camfeed1.setGeometry(QtCore.QRect(0, 30, 640, 480))
        self.label_camfeed1.setAutoFillBackground(False)
        self.label_camfeed1.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_camfeed1.setFrameShape(QtWidgets.QFrame.Box)
        self.label_camfeed1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_camfeed1.setLineWidth(2)
        self.label_camfeed1.setMidLineWidth(0)
        self.label_camfeed1.setText("")
        self.label_camfeed1.setObjectName("label_camfeed1")
        self.label_camfeed2 = QtWidgets.QLabel(self.centralwidget)
        self.label_camfeed2.setGeometry(QtCore.QRect(640, 30, 640, 480))
        self.label_camfeed2.setAutoFillBackground(False)
        self.label_camfeed2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_camfeed2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_camfeed2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_camfeed2.setLineWidth(2)
        self.label_camfeed2.setMidLineWidth(0)
        self.label_camfeed2.setText("")
        self.label_camfeed2.setObjectName("label_camfeed2")
        self.button_start_streams = QtWidgets.QPushButton(self.centralwidget)
        self.button_start_streams.setGeometry(QtCore.QRect(0, 0, 88, 29))
        self.button_start_streams.setObjectName("button_start_streams")
        self.framerate_cam1 = QtWidgets.QLabel(self.centralwidget)
        self.framerate_cam1.setGeometry(QtCore.QRect(0, 510, 101, 21))
        self.framerate_cam1.setObjectName("framerate_cam1")
        self.framerate_cam2 = QtWidgets.QLabel(self.centralwidget)
        self.framerate_cam2.setGeometry(QtCore.QRect(640, 510, 101, 21))
        self.framerate_cam2.setObjectName("framerate_cam2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setEnabled(True)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1406, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setEnabled(True)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_2.setObjectName("toolBar_2")
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar_2)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_start_streams.setText(_translate("MainWindow", "Start Streams"))
        self.framerate_cam1.setText(_translate("MainWindow", "FPS:"))
        self.framerate_cam2.setText(_translate("MainWindow", "FPS:"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))

