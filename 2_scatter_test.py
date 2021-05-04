# -*- coding: utf-8 -*-
# embed pyqtgraph in pyqt5: https://www.mfitzp.com/tutorials/embed-pyqtgraph-custom-widgets-qt-app/

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np
import sys

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ScatterPlot(pg.ScatterPlotItem):
    def mouseClickEvent(self, event):
        print("clicked" + str(self.x))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.PlotWidget = PlotWidget(self.centralwidget)
        self.PlotWidget.setGeometry(QtCore.QRect(10, 10, 800, 800))
        self.PlotWidget.setObjectName("PlotWidget")



        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 869, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

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
