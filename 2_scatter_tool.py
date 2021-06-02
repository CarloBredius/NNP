# -*- coding: utf-8 -*-
# embed pyqtgraph in pyqt5: https://www.mfitzp.com/tutorials/embed-pyqtgraph-custom-widgets-qt-app/

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np
import sys
from random import randint

from perturb import *
from visualize import *

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

                # Main window
                self.centralwidget = QWidget(MainWindow)
                self.centralwidget.setObjectName("centralwidget")
                MainWindow.setCentralWidget(self.centralwidget)

                # Plot widget
                self.PlotWidget = PlotWidget(self.centralwidget)
                self.PlotWidget.setObjectName("PlotWidget")

                # Trails view
                self.trailsScene = QGraphicsScene()
                self.trailsView = QGraphicsView(self.trailsScene, self.centralwidget)
                self.trailsView.setObjectName("trailsView")

                # Heatmap view
                self.heatmapScene = QGraphicsScene()
                self.heatmapView = QGraphicsView(self.heatmapScene, self.centralwidget)
                self.heatmapView.setObjectName("heatmapView")

                # OpenGL widget
                self.openGLWidget = OpenGLWidget()
                self.openGLWidget.setObjectName("openGLView")

                # Stacked widget, combining plot widget and graphics view
                self.stackedWidget = QStackedWidget(self.centralwidget)
                self.stackedWidget.setGeometry(QRect(10, 10, 800, 800))
                self.stackedWidget.setObjectName("stackedWidget")
                self.stackedWidget.addWidget(self.PlotWidget)
                self.stackedWidget.addWidget(self.trailsView)
                self.stackedWidget.addWidget(self.heatmapView)
                self.stackedWidget.addWidget(self.openGLWidget)

                # Perturbation title
                self.perturbationSlidersLabel = QLabel(self.centralwidget)
                self.perturbationSlidersLabel.setGeometry(QRect(960, 30, 170, 20))
                font = QFont()
                font.setPointSize(14)
                self.perturbationSlidersLabel.setFont(font)
                self.perturbationSlidersLabel.setObjectName("perturbation_title")

                # Perturbation 1
                self.horizontalSlider1 = QSlider(self.centralwidget)
                self.horizontalSlider1.setGeometry(QRect(920, 80, 270, 20))
                self.horizontalSlider1.setMaximum(100)
                self.horizontalSlider1.setOrientation(Qt.Horizontal)
                self.horizontalSlider1.setInvertedAppearance(False)
                self.horizontalSlider1.setObjectName("horizontal1Slider")
                self.horizontalSlider1.valueChanged.connect(self.slider1Changed)
                self.checkBox1 = QCheckBox(self.centralwidget)
                self.checkBox1.setGeometry(QRect(830, 80, 90, 20))
                self.checkBox1.setTristate(False)
                self.checkBox1.setObjectName("checkBox1")

                # Perturbation 2
                self.horizontalSlider2 = QSlider(self.centralwidget)
                self.horizontalSlider2.setGeometry(QRect(920, 130, 270, 20))
                self.horizontalSlider2.setMaximum(100)
                self.horizontalSlider2.setOrientation(Qt.Horizontal)
                self.horizontalSlider2.setInvertedAppearance(False)
                self.horizontalSlider2.setObjectName("horizontalSlider2")
                self.horizontalSlider2.valueChanged.connect(self.slider2Changed)
                self.checkBox2 = QCheckBox(self.centralwidget)
                self.checkBox2.setGeometry(QRect(830, 130, 90, 20))
                self.checkBox2.setObjectName("checkBox2")

                # Perturbation 3
                self.horizontalSlider3 = QSlider(self.centralwidget)
                self.horizontalSlider3.setGeometry(QRect(920, 180, 270, 20))
                self.horizontalSlider3.setMaximum(100)
                self.horizontalSlider3.setOrientation(Qt.Horizontal)
                self.horizontalSlider3.setInvertedAppearance(False)
                self.horizontalSlider3.setObjectName("horizontalSlider3")
                self.horizontalSlider3.valueChanged.connect(self.slider3Changed)
                self.checkBox3 = QCheckBox(self.centralwidget)
                self.checkBox3.setGeometry(QRect(830, 180, 90, 20))
                self.checkBox3.setObjectName("checkBox3")

                # Perturbation 4
                self.horizontalSlider4 = QSlider(self.centralwidget)
                self.horizontalSlider4.setGeometry(QRect(920, 230, 270, 20))
                self.horizontalSlider4.setMaximum(100)
                self.horizontalSlider4.setOrientation(Qt.Horizontal)
                self.horizontalSlider4.setInvertedAppearance(False)
                self.horizontalSlider4.setObjectName("horizontalSlider4")
                self.horizontalSlider4.valueChanged.connect(self.slider4Changed)
                self.checkBox4 = QCheckBox(self.centralwidget)
                self.checkBox4.setGeometry(QRect(830, 230, 90, 20))
                self.checkBox4.setObjectName("checkBox4")

                # Random perturbation button
                self.perturbSelectedButton = QPushButton(self.centralwidget)
                self.perturbSelectedButton.setGeometry(QRect(830, 300, 150, 50))
                self.perturbSelectedButton.setObjectName("perturbSelectedButton")
                self.perturbSelectedButton.setToolTip("Randomly set all checked perturbations")
                self.perturbSelectedButton.clicked.connect(self.perturbSelected)

                self.reset = QPushButton(self.centralwidget)
                self.reset.setGeometry(QRect(1000, 300, 110, 50))
                self.reset.setObjectName("reset_button")
                self.reset.setToolTip("Set all checked perturbations back to 0")
                self.reset.clicked.connect(self.resetSliders)

                # Menubar items
                self.menubar = QMenuBar(MainWindow)
                self.menubar.setGeometry(QRect(0, 0, 1285, 21))
                self.menubar.setObjectName("menubar")
                self.menuFile = QMenu(self.menubar)
                self.menuFile.setObjectName("menuFile")
                self.menuViews = QMenu(self.menubar)
                self.menuViews.setObjectName("menuViews")
                MainWindow.setMenuBar(self.menubar)
                self.statusbar = QStatusBar(MainWindow)
                self.statusbar.setObjectName("statusbar")
                MainWindow.setStatusBar(self.statusbar)

                self.viewsBasic = QAction(MainWindow)
                self.viewsBasic.setObjectName("viewsBasic")
                self.viewsBasic.triggered.connect(lambda: self.switchWidget(0))
                self.viewsTrail = QAction(MainWindow)
                self.viewsTrail.setObjectName("viewsTrail")
                self.viewsTrail.triggered.connect(lambda: self.switchWidget(1))
                self.viewsHeatmap = QAction(MainWindow)
                self.viewsHeatmap.setObjectName("viewsHeatmap")
                self.viewsHeatmap.triggered.connect(lambda: self.switchWidget(2))
                self.viewsOpenGL = QAction(MainWindow)
                self.viewsOpenGL.setObjectName("viewsOpenGL")
                self.viewsOpenGL.triggered.connect(lambda: self.switchWidget(3))

                self.fileReset = QAction(MainWindow)
                self.fileReset.setObjectName("fileReset")
                # self.fileReset.action.connect(self.reset)
                self.fileSave = QAction(MainWindow)
                self.fileSave.setObjectName("fileSave")
                self.menuFile.addAction(self.fileReset)
                self.menuFile.addAction(self.fileSave)
                self.menuViews.addAction(self.viewsBasic)
                self.menuViews.addAction(self.viewsTrail)
                self.menuViews.addAction(self.viewsHeatmap)
                self.menuViews.addAction(self.viewsOpenGL)

                self.menubar.addAction(self.menuFile.menuAction())
                self.menubar.addAction(self.menuViews.menuAction())

                self.retranslateUi(MainWindow)
                QMetaObject.connectSlotsByName(MainWindow)

        def retranslateUi(self, MainWindow):
                _translate = QCoreApplication.translate
                MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
                self.checkBox1.setText(_translate("MainWindow", "Add constant"))
                self.checkBox2.setText(_translate("MainWindow", "Dim. removal"))
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
                self.viewsOpenGL.setText(_translate("MainWindow", "OpenGL interface"))
                self.viewsOpenGL.setShortcut(_translate("MainWindow", "Ctrl+4"))

                self.fileReset.setText(_translate("MainWindow", "Reset"))
                self.fileReset.setShortcut(_translate("MainWindow", "Ctrl+R"))
                self.fileSave.setText(_translate("MainWindow", "Save"))
                self.fileSave.setShortcut(_translate("MainWindow", "Ctrl+S"))

        def switchWidget(self, index):
            self.stackedWidget.setCurrentIndex(index)
            print("Going to stacked widget index: " + str(index))
            # TODO: only compute intermediateDatasets if not already done so
            if index == 1:
                self.statusbar.showMessage("Computing and predicting intermediate datasets per increment...")
                self.computeIntermediateDatasets()
                self.statusbar.showMessage("Drawing trail map...")
                self.projectTrailMap()
                self.statusbar.showMessage("Trail map projected.")
            if index == 2:
                self.statusbar.showMessage("Computing and predicting intermediate datasets per increment...")
                self.computeIntermediateDatasets()
                self.statusbar.showMessage("Drawing heat map...")
                self.projectHeatMap()
                self.statusbar.showMessage("Heat map projected.")
            if index == 3:
                self.computeIntermediateDatasets()
                if self.predList:
                    self.openGLWidget.paintTrailMapGL(self.predList, self.y_test, self.class_colors)
                else:
                    print("No data to create trail map with.")
                    self.statusbar.showMessage("No data to create trail map with.")

        def computeIntermediateDatasets(self):
            self.predList = []
            # If noise perturbation is selected
            if self.checkBox1.isChecked():
                maxValue = self.horizontalSlider1.value()
                # Perturb the dataset per increment
                self.dataset.interDataOfPerturb(1, maxValue)

                # Predict every dataset and save to predList
                for i in range(1, maxValue):
                    #QpointF(self.trailsView.mapToScene(self.dataset.interDataset[i]))
                    self.predList.append(self.model.predict(self.dataset.interDataset[i]))

            # If dimension removal perturbation is selected
            if self.checkBox2.isChecked():
                maxValue = self.horizontalSlider2.value()
                # Perturb the dataset per increment
                self.dataset.interDataOfPerturb(2, maxValue)
                pred = self.model.predict(self.dataset.interDataset)

        # Project a trail map using the data in predList
        def projectTrailMap(self):
            for j in range(len(self.predList[0])):
                pen = pg.mkPen(color = self.y_test[j], width = 1)
                for i in range(len(self.predList) - 1):
                    x1 = self.predList[i][j][0]
                    y1 = self.predList[i][j][1]
                    x2 = self.predList[i + 1][j][0]
                    y2 = self.predList[i + 1][j][1]
                    self.trailsScene.addLine(self.predList[i][j][0], self.predList[i][j][1], self.predList[i + 1][j][0], self.predList[i + 1][j][1], pen)

            # Fit trail map to screen
            # self.trailsScene.itemsBoundingRect()
            self.trailsView.fitInView(0, 0, 1, 1, Qt.KeepAspectRatio)

        def projectHeatMap(selfs):
            pass

        def replot(self):
                pred = self.model.predict(self.dataset.perturbed)
                items = pg.ScatterPlotItem(x=pred[:,0],  y=pred[:,1], data=np.arange(len(self.dataset.perturbed)),
                 pen='w', brush=self.brushes, size=10, hoverable=True, hoverPen=pg.mkPen(0, 0, 0, 255))
                self.PlotWidget.clear()
                self.PlotWidget.addItem(items)

        def slider1Changed(self):
                new_value = self.horizontalSlider1.value()
                self.statusbar.showMessage("Changed value of perturbation slider 1 to " + str(new_value))
                self.dataset.addConstantNoise(new_value)
                self.replot()

        def slider2Changed(self):
                new_value = self.horizontalSlider2.value()
                self.statusbar.showMessage("Changed value of perturbation slider 2 to " + str(new_value))
                self.dataset.removeRandomDimensions(new_value)
                self.replot()

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
            self.y_test = y_test

            # Class colors
            self.class_colors = [(0.651, 0.808, 0.890),
                                 (0.122, 0.471, 0.706),
                                 (0.698, 0.875, 0.541),
                                 (0.200, 0.627, 0.173),
                                 (0.984, 0.604, 0.600),
                                 (0.890, 0.102, 0.110),
                                 (0.992, 0.749, 0.435),
                                 (1.000, 0.498, 0.000),
                                 (0.792, 0.698, 0.839),
                                 (0.416, 0.239, 0.604),
                                 (1.000, 1.000, 0.600),
                                 (0.694, 0.349, 0.147)]

            self.dataset = Dataset(X_test)

            print("Loading NNP model...")
            self.model = keras.models.load_model("NNP_model_" + label)

            pred = self.model.predict(X_test)
            self.brushes = []
            for label in y_test:
                color = self.class_colors[label]
                self.brushes.append(QBrush(QColor.fromRgbF(color[0], color[1], color[2])))

            items = pg.ScatterPlotItem(x=pred[:,0],  y=pred[:,1], data=np.arange(len(X_test)),
                 pen='w', brush=self.brushes, size=10, hoverable=True, hoverPen=pg.mkPen(0, 0, 0, 255))

            self.PlotWidget.addItem(items)


if __name__ == "__main__":
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)

        ui.loadData(MainWindow)

        MainWindow.show()
        sys.exit(app.exec_())
