# -*- coding: utf-8 -*-
# embed pyqtgraph in pyqt5: https://www.mfitzp.com/tutorials/embed-pyqtgraph-custom-widgets-qt-app/

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np
import sys
from random import randint

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ScatterPlot(pg.ScatterPlotItem):
    def mouseClickEvent(self, event):
        print("clicked on " + str(self.x))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 850)

        # plot window
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.PlotWidget = PlotWidget(self.centralwidget)
        self.PlotWidget.setGeometry(QtCore.QRect(10, 10, 800, 800))
        self.PlotWidget.setObjectName("PlotWidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Perturbation title
        self.perturbationSlidersLabel = QtWidgets.QLabel(self.centralwidget)
        self.perturbationSlidersLabel.setGeometry(QtCore.QRect(960, 30, 170, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.perturbationSlidersLabel.setFont(font)
        self.perturbationSlidersLabel.setObjectName("perturbation_title")

        # perturbation 1
        self.horizontalSlider1 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider1.setGeometry(QtCore.QRect(920, 80, 270, 20))
        self.horizontalSlider1.setMaximum(100)
        self.horizontalSlider1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider1.setInvertedAppearance(False)
        self.horizontalSlider1.setObjectName("horizontal1Slider")
        self.horizontalSlider1.valueChanged.connect(self.slider1Changed)
        self.checkBox1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox1.setGeometry(QtCore.QRect(830, 80, 90, 20))
        self.checkBox1.setTristate(False)
        self.checkBox1.setObjectName("checkBox1")

        # perturbation 2
        self.horizontalSlider2 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider2.setGeometry(QtCore.QRect(920, 130, 270, 20))
        self.horizontalSlider2.setMaximum(100)
        self.horizontalSlider2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider2.setInvertedAppearance(False)
        self.horizontalSlider2.setObjectName("horizontalSlider2")
        self.horizontalSlider2.valueChanged.connect(self.slider2Changed)
        self.checkBox2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox2.setGeometry(QtCore.QRect(830, 130, 90, 20))
        self.checkBox2.setObjectName("checkBox2")

        # perturbation 3
        self.horizontalSlider3 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider3.setGeometry(QtCore.QRect(920, 180, 270, 20))
        self.horizontalSlider3.setMaximum(100)
        self.horizontalSlider3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider3.setInvertedAppearance(False)
        self.horizontalSlider3.setObjectName("horizontalSlider3")
        self.horizontalSlider3.valueChanged.connect(self.slider3Changed)
        self.checkBox3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox3.setGeometry(QtCore.QRect(830, 180, 90, 20))
        self.checkBox3.setObjectName("checkBox3")

        # perturbation 4
        self.horizontalSlider4 = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider4.setGeometry(QtCore.QRect(920, 230, 270, 20))
        self.horizontalSlider4.setMaximum(100)
        self.horizontalSlider4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider4.setInvertedAppearance(False)
        self.horizontalSlider4.setObjectName("horizontalSlider4")
        self.horizontalSlider4.valueChanged.connect(self.slider4Changed)
        self.checkBox4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox4.setGeometry(QtCore.QRect(830, 230, 90, 20))
        self.checkBox4.setObjectName("checkBox4")

        # Random perturbation button
        self.perturbSelectedButton = QtWidgets.QPushButton(self.centralwidget)
        self.perturbSelectedButton.setGeometry(QtCore.QRect(830, 300, 150, 50))
        self.perturbSelectedButton.setObjectName("perturbSelectedButton")
        self.perturbSelectedButton.clicked.connect(self.perturbSelected)

        self.reset = QtWidgets.QPushButton(self.centralwidget)
        self.reset.setGeometry(QtCore.QRect(1000, 300, 110, 50))
        self.reset.setObjectName("reset_button")
        self.reset.clicked.connect(self.resetSliders)

        # Menubar items
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1285, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuViews = QtWidgets.QMenu(self.menubar)
        self.menuViews.setObjectName("menuViews")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.viewsBasic = QtWidgets.QAction(MainWindow)
        self.viewsBasic.setObjectName("viewsBasic")
        self.viewsTrail = QtWidgets.QAction(MainWindow)
        self.viewsTrail.setObjectName("viewsTrail")
        self.viewsHeatmap = QtWidgets.QAction(MainWindow)
        self.viewsHeatmap.setObjectName("viewsHeatmap")
        self.fileReset = QtWidgets.QAction(MainWindow)
        self.fileReset.setObjectName("fileReset")
        #self.fileReset.action.connect(self.reset)
        self.fileSave = QtWidgets.QAction(MainWindow)
        self.fileSave.setObjectName("fileSave")
        self.menuFile.addAction(self.fileReset)
        self.menuFile.addAction(self.fileSave)
        self.menuViews.addAction(self.viewsBasic)
        self.menuViews.addAction(self.viewsTrail)
        self.menuViews.addAction(self.viewsHeatmap)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuViews.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox1.setText(_translate("MainWindow", "Perturbation1"))
        self.checkBox2.setText(_translate("MainWindow", "Perturbation2"))
        self.checkBox3.setText(_translate("MainWindow", "Perturbation3"))
        self.checkBox4.setText(_translate("MainWindow", "Perturbation4"))
        self.perturbSelectedButton.setText(_translate("MainWindow", "Randomize selected"))
        self.reset.setText(_translate("MainWindow", "Reset"))
        self.perturbationSlidersLabel.setText(_translate("MainWindow", "Perturbation sliders"))

        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuViews.setTitle(_translate("MainWindow", "Views"))

        self.viewsBasic.setText(_translate("MainWindow", "Basic interface"))
        self.viewsBasic.setShortcut(_translate("MainWindow", "Ctrl+1"))
        self.viewsTrail.setText(_translate("MainWindow", "Trail interface"))
        self.viewsTrail.setShortcut(_translate("MainWindow", "Ctrl+2"))
        self.viewsHeatmap.setText(_translate("MainWindow", "Heat map interface"))
        self.viewsHeatmap.setShortcut(_translate("MainWindow", "Ctrl+3"))

        self.fileReset.setText(_translate("MainWindow", "Reset"))
        self.fileReset.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.fileSave.setText(_translate("MainWindow", "Save"))
        self.fileSave.setShortcut(_translate("MainWindow", "Ctrl+S"))

    def slider1Changed(self):
        new_value = self.horizontalSlider1.value()
        self.statusbar.showMessage("Changed value of perturbation slider 1 to " + str(new_value))

    def slider2Changed(self):
        new_value = self.horizontalSlider2.value()
        self.statusbar.showMessage("Changed value of perturbation slider 2 to " + str(new_value))

    def slider3Changed(self):
        new_value = self.horizontalSlider3.value()
        self.statusbar.showMessage("Changed value of perturbation slider 3 to " + str(new_value))

    def slider4Changed(self):
        new_value = self.horizontalSlider4.value()
        self.statusbar.showMessage("Changed value of perturbation slider 4 to " + str(new_value))

    def perturbSelected(self):
        if self.checkBox1.isChecked():
            self.horizontalSlider1.setValue(randint(1, self.horizontalSlider2.maximum()))
        if self.checkBox2.isChecked():
            self.horizontalSlider2.setValue(randint(1, self.horizontalSlider2.maximum()))
        if self.checkBox3.isChecked():
            self.horizontalSlider3.setValue(randint(1, self.horizontalSlider3.maximum()))
        if self.checkBox4.isChecked():
            self.horizontalSlider4.setValue(randint(1, self.horizontalSlider4.maximum()))

    def resetSliders(self):
        if self.checkBox1.isChecked():
            self.horizontalSlider1.setValue(0)
        if self.checkBox2.isChecked():
            self.horizontalSlider2.setValue(0)
        if self.checkBox3.isChecked():
            self.horizontalSlider3.setValue(0)
        if self.checkBox4.isChecked():
            self.horizontalSlider4.setValue(0)

    def loadTestData(self, MainWindow):
    	print("Loading dummy data...")
    	x = np.random.normal(size=1000)
    	y = np.random.normal(size=1000)

    	brush = pg.mkBrush(50, 200, 50, 120)
    	hoverBrush = pg.mkBrush(200, 50, 50, 200)
    	items = pg.ScatterPlotItem(x=x, y=y, pen='w', brush=brush, size=15, hoverable=True, hoverBrush=hoverBrush)
    	self.PlotWidget.addItem(items)

    def loadData(self, MainWindow):
    	print("Loading datasets...")
    	X = np.load('data/X_mnist.npy')
    	y = np.load('data/y_mnist.npy')
    	label = "mnist-full"

    	X_train, X_test, y_train, y_test = train_test_split(X, y,
    	 train_size=10000, test_size=3000, random_state=420, stratify=y)

    	print("Loading NNP model...")
    	model = keras.models.load_model("NNP_model_" + label)

    	pred = model.predict(X_test)

    	items = pg.ScatterPlotItem(x=pred[:,0],  y=pred[:,1], data=y_test,
         pen='w', brush=y_test, size=10, hoverable=True, hoverPen=pg.mkPen(0, 0, 0, 255))

    	self.PlotWidget.addItem(items)

if __name__ == "__main__":
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    ui.loadData(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())
