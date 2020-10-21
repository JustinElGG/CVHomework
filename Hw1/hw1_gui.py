from PyQt5 import QtWidgets, QtGui, QtCore
from Label import Ui_HW1_GUI
import sys
import hw1_1
import hw1_2
import hw1_3
import hw1_4
import hw1_5


class MainWindow(QtWidgets.QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_HW1_GUI()
        self.ui.setupUi(self)
        self.ui.pushButton_1.clicked.connect(self.buttonClicked_1)
        self.ui.pushButton_2.clicked.connect(self.buttonClicked_2)
        self.ui.pushButton_3.clicked.connect(self.buttonClicked_3)
        self.ui.pushButton_4.clicked.connect(self.buttonClicked_4)
        self.ui.pushButton_5.clicked.connect(self.buttonClicked_5)
        self.ui.pushButton.clicked.connect(self.buttonClicked)
        self.ui.pushButton_6.clicked.connect(self.buttonClicked_6)
        self.ui.pushButton_7.clicked.connect(self.buttonClicked_7)
        self.ui.pushButton_8.clicked.connect(self.buttonClicked_8)
        self.ui.pushButton_10.clicked.connect(self.buttonClicked_10)
        self.ui.pushButton_11.clicked.connect(self.buttonClicked_11)
        self.ui.pushButton_12.clicked.connect(self.buttonClicked_12)
        self.ui.pushButton_13.clicked.connect(self.buttonClicked_13)
        self.ui.pushButton_14.clicked.connect(self.buttonClicked_14)

        for i in range(1, 16):
            self.ui.comboBox.addItems([str(i)])

    def buttonClicked_1(self):
        print('*-----Find corners-----*')
        hw1_1.corner_detection()
        print('Done!')
    def buttonClicked_2(self):
        print('*-----Find intrinsic-----*')
        hw1_1.intrinsic()
        print('Done!')
    def buttonClicked_3(self):
        print('*-----Find extrinsic-----*')
        print('<<<Target is img{}>>>'.format(self.ui.comboBox.currentText()))
        hw1_1.extrinsic(self.ui.comboBox.currentText())
        print('Done!')
    def buttonClicked_4(self):
        print('*-----Find distortion-----*')
        hw1_1.distortion()
        print('Done!')
    def buttonClicked_5(self):
        print('*-----Augmented Reality-----*')
        print('Press any button to exit')
        hw1_2.draw_tri()
        print('Done!')
    def buttonClicked(self):
        print('*-----Find corners-----*')
        hw1_2.inputHandler('exit')
        print('Done!')
    def buttonClicked_6(self):
        print('*-----Disparity Map-----*')
        print('Press any button to exit')
        hw1_3.disparity_map()
        print('Done!')
    def buttonClicked_7(self):
        print('*-----Keypoints-----*')
        print('Press any button to exit')
        hw1_4.SIFT('keypoint')
        print('Done!')
    def buttonClicked_8(self):
        print('*-----Matched keypoints-----*')
        print('Press any button to exit')
        hw1_4.SIFT('match')
        print('Done!')

    def buttonClicked_10(self):
        print('*-----Check training image-----*')
        print('Press any button to exit')
        hw1_5.show_image()
        print('Done!')

    def buttonClicked_11(self):
        print('*-----Model structure-----*')
        print('Press any button to exit')
        hw1_5.parameter()
        print('Done!')

    def buttonClicked_12(self):
        print('*-----Hyperparameter-----*')
        hw1_5.structure()
        print('Done!')

    def buttonClicked_13(self):
        print('*-----Training log-----*')
        print('Press any button to exit')
        hw1_5.log()
        print('Done!')

    def buttonClicked_14(self):
        print('*-----Inference-----*')
        hw1_5.inference(int(self.ui.spinBox.value()))
        print('Done!')

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
