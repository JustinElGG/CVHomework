# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Label.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HW1_GUI(object):
    def setupUi(self, HW1_GUI):
        HW1_GUI.setObjectName("HW1_GUI")
        HW1_GUI.resize(1041, 502)
        self.groupBox = QtWidgets.QGroupBox(HW1_GUI)
        self.groupBox.setGeometry(QtCore.QRect(30, 20, 361, 211))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton_1 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_1.setGeometry(QtCore.QRect(20, 40, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 90, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 140, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(180, 40, 161, 141))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 80, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 40, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setGeometry(QtCore.QRect(110, 40, 41, 31))
        self.comboBox.setObjectName("comboBox")
        self.groupBox_3 = QtWidgets.QGroupBox(HW1_GUI)
        self.groupBox_3.setGeometry(QtCore.QRect(60, 260, 291, 211))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_5.setGeometry(QtCore.QRect(90, 60, 111, 101))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.pushButton_5.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton.setGeometry(QtCore.QRect(110, 170, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_4 = QtWidgets.QGroupBox(HW1_GUI)
        self.groupBox_4.setGeometry(QtCore.QRect(410, 20, 291, 211))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_6.setGeometry(QtCore.QRect(90, 60, 111, 101))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.pushButton_6.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.groupBox_6 = QtWidgets.QGroupBox(HW1_GUI)
        self.groupBox_6.setGeometry(QtCore.QRect(410, 260, 291, 211))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_7.setGeometry(QtCore.QRect(50, 50, 191, 51))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.pushButton_7.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_8.setGeometry(QtCore.QRect(50, 120, 191, 51))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        self.pushButton_8.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox_5 = QtWidgets.QGroupBox(HW1_GUI)
        self.groupBox_5.setGeometry(QtCore.QRect(720, 20, 291, 451))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 50, 231, 41))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 120, 231, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_12.setGeometry(QtCore.QRect(30, 190, 231, 41))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_13.setGeometry(QtCore.QRect(30, 260, 231, 41))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_14.setGeometry(QtCore.QRect(30, 330, 231, 41))
        self.pushButton_14.setObjectName("pushButton_14")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_5)
        self.spinBox.setGeometry(QtCore.QRect(80, 390, 141, 31))
        self.spinBox.setMaximum(9999)
        self.spinBox.setObjectName("spinBox")

        self.retranslateUi(HW1_GUI)
        QtCore.QMetaObject.connectSlotsByName(HW1_GUI)

    def retranslateUi(self, HW1_GUI):
        _translate = QtCore.QCoreApplication.translate
        HW1_GUI.setWindowTitle(_translate("HW1_GUI", "HW1"))
        self.groupBox.setTitle(_translate("HW1_GUI", "1. Calibration"))
        self.pushButton_1.setText(_translate("HW1_GUI", "1.1 Find Corners"))
        self.pushButton_2.setText(_translate("HW1_GUI", "1.2 Find Intrinsic"))
        self.pushButton_4.setText(_translate("HW1_GUI", "1.4 Find Distortion"))
        self.groupBox_2.setTitle(_translate("HW1_GUI", "1.3 Find Extrinsic"))
        self.pushButton_3.setText(_translate("HW1_GUI", "1.3 Find Extrinsic"))
        self.label.setText(_translate("HW1_GUI", "Select img"))
        self.groupBox_3.setTitle(_translate("HW1_GUI", "2. AugmentedReality"))
        self.pushButton_5.setText(_translate("HW1_GUI", "Augmented\n"
"Reality"))
        self.pushButton.setText(_translate("HW1_GUI", "Stop"))
        self.groupBox_4.setTitle(_translate("HW1_GUI", "3. Disparit Map"))
        self.pushButton_6.setText(_translate("HW1_GUI", "Disparity Map"))
        self.groupBox_6.setTitle(_translate("HW1_GUI", "4. SIFT"))
        self.pushButton_7.setText(_translate("HW1_GUI", "4.1 Keypoints"))
        self.pushButton_8.setText(_translate("HW1_GUI", "4.2 Matched keypoints"))
        self.groupBox_5.setTitle(_translate("HW1_GUI", "5. Cifar10 VGG16"))
        self.pushButton_10.setText(_translate("HW1_GUI", "1. Show train images"))
        self.pushButton_11.setText(_translate("HW1_GUI", "2. Show hyperparameters"))
        self.pushButton_12.setText(_translate("HW1_GUI", "3.Show model structure"))
        self.pushButton_13.setText(_translate("HW1_GUI", "4.Show accuracy"))
        self.pushButton_14.setText(_translate("HW1_GUI", "5.Test"))
