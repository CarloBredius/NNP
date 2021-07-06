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
from trails import *
from heat import *
from star import *

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

        self.lastPerturbation = None

        # Main window
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Plot widget
        self.plotWidget = PlotWidget(self.centralwidget)
        self.plotWidget.setObjectName("PlotWidget")

        # trails widget using OpenGL
        self.trailsGLWidget = TrailsGLWidget()
        self.trailsGLWidget.setObjectName("trailsGLView")

        # Heatmap view
        self.heatGLWidget = HeatGLWidget()
        self.heatGLWidget.setObjectName("heatGLView")

        # Star map
        self.starMapGLWidget = StarMapGLWidget()
        self.starMapGLWidget.setObjectName("starMapGLWidget")

        # Stacked widget, combining plot widget and graphics view
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QRect(10, 10, 800, 800))
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidget.addWidget(self.plotWidget)
        self.stackedWidget.addWidget(self.trailsGLWidget)
        self.stackedWidget.addWidget(self.heatGLWidget)
        self.stackedWidget.addWidget(self.starMapGLWidget)

        # Widget Title
        self.WidgetTitle = QLabel(self.centralwidget)
        self.WidgetTitle.setGeometry(QRect(920, 30, 200, 35))
        font = QFont()
        font.setPointSize(20)
        self.WidgetTitle.setFont(font)
        self.WidgetTitle.setObjectName("WidgetTitle")

        # Perturbation label
        self.perturbationSlidersLabel = QLabel(self.centralwidget)
        self.perturbationSlidersLabel.setGeometry(QRect(930, 80, 170, 25))
        font = QFont()
        font.setPointSize(14)
        self.perturbationSlidersLabel.setFont(font)
        self.perturbationSlidersLabel.setObjectName("perturbation_title")

        # Configuration label
        self.configLabel = QLabel(self.centralwidget)
        self.configLabel.setGeometry(QRect(930, 350, 170, 25))
        font = QFont()
        font.setPointSize(14)
        self.configLabel.setFont(font)
        self.configLabel.setObjectName("ConfigLabel")
        self.configLabel.setVisible(False)

        # Noise
        self.horizontalSlider1 = QSlider(self.centralwidget)
        self.horizontalSlider1.setGeometry(QRect(920, 130, 270, 20))
        self.horizontalSlider1.setMaximum(100)
        self.horizontalSlider1.setOrientation(Qt.Horizontal)
        self.horizontalSlider1.setInvertedAppearance(False)
        self.horizontalSlider1.setObjectName("horizontal1Slider")
        self.horizontalSlider1.valueChanged.connect(self.slider1Changed)
        self.checkBox1 = QCheckBox(self.centralwidget)
        self.checkBox1.setGeometry(QRect(830, 130, 90, 20))
        self.checkBox1.setTristate(False)
        self.checkBox1.setObjectName("checkBox1")
        self.radioButton1 = QRadioButton(self.centralwidget)
        self.radioButton1.setGeometry(QRect(830, 130, 90, 20))
        self.radioButton1.setObjectName("radioButton1")
        self.radioButton1.setVisible(False)

        # Dimension removal
        self.horizontalSlider2 = QSlider(self.centralwidget)
        self.horizontalSlider2.setGeometry(QRect(920, 180, 270, 20))
        self.horizontalSlider2.setMaximum(100)
        self.horizontalSlider2.setOrientation(Qt.Horizontal)
        self.horizontalSlider2.setInvertedAppearance(False)
        self.horizontalSlider2.setObjectName("horizontalSlider2")
        self.horizontalSlider2.valueChanged.connect(self.slider2Changed)
        self.checkBox2 = QCheckBox(self.centralwidget)
        self.checkBox2.setGeometry(QRect(830, 180, 90, 20))
        self.checkBox2.setObjectName("checkBox2")
        self.radioButton2 = QRadioButton(self.centralwidget)
        self.radioButton2.setGeometry(QRect(830, 180, 90, 20))
        self.radioButton2.setObjectName("radioButton2")
        self.radioButton2.setVisible(False)

        # Jitter
        self.horizontalSlider3 = QSlider(self.centralwidget)
        self.horizontalSlider3.setGeometry(QRect(920, 230, 270, 20))
        self.horizontalSlider3.setMaximum(100)
        self.horizontalSlider3.setOrientation(Qt.Horizontal)
        self.horizontalSlider3.setInvertedAppearance(False)
        self.horizontalSlider3.setObjectName("horizontalSlider3")
        self.horizontalSlider3.valueChanged.connect(self.slider3Changed)
        self.checkBox3 = QCheckBox(self.centralwidget)
        self.checkBox3.setGeometry(QRect(830, 230, 90, 20))
        self.checkBox3.setObjectName("checkBox3")
        self.radioButton3 = QRadioButton(self.centralwidget)
        self.radioButton3.setGeometry(QRect(830, 230, 90, 20))
        self.radioButton3.setObjectName("radioButton3")
        self.radioButton3.setVisible(False)

        # Perturbation 4
        self.horizontalSlider4 = QSlider(self.centralwidget)
        self.horizontalSlider4.setGeometry(QRect(920, 280, 270, 20))
        self.horizontalSlider4.setMaximum(100)
        self.horizontalSlider4.setOrientation(Qt.Horizontal)
        self.horizontalSlider4.setInvertedAppearance(False)
        self.horizontalSlider4.setObjectName("horizontalSlider4")
        self.horizontalSlider4.valueChanged.connect(self.slider4Changed)
        self.checkBox4 = QCheckBox(self.centralwidget)
        self.checkBox4.setGeometry(QRect(830, 280, 90, 20))
        self.checkBox4.setObjectName("checkBox4")
        self.radioButton4 = QRadioButton(self.centralwidget)
        self.radioButton4.setGeometry(QRect(830, 280, 90, 20))
        self.radioButton4.setObjectName("radioButton4")
        self.radioButton4.setVisible(False)

        # Random perturbation and reset button
        self.perturbSelectedButton = QPushButton(self.centralwidget)
        self.perturbSelectedButton.setGeometry(QRect(830, 350, 150, 50))
        self.perturbSelectedButton.setObjectName("perturbSelectedButton")
        self.perturbSelectedButton.setToolTip("Randomly set all checked perturbations")
        self.perturbSelectedButton.clicked.connect(self.randChangeSelected)
        self.resetButton = QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QRect(1000, 350, 110, 50))
        self.resetButton.setObjectName("reset_button")
        self.resetButton.setToolTip("Set all checked perturbations back to 0")
        self.resetButton.clicked.connect(self.resetSelected)

        # Compute current configuration button
        self.computeButton = QPushButton(self.centralwidget)
        self.computeButton.setGeometry(QRect(935, 725, 150, 50))
        self.computeButton.setObjectName("ComputeButton")
        self.computeButton.setToolTip("Compute visualization for the current configuration")
        self.computeButton.clicked.connect(self.computeVisualization)
        self.computeButton.setVisible(False)

        # Trail map options
        self.lineThicknessLabel = QLabel(self.centralwidget)
        self.lineThicknessLabel.setGeometry(QRect(920, 390, 170, 20))
        font = QFont()
        font.setPointSize(10)
        self.lineThicknessLabel.setFont(font)
        self.lineThicknessLabel.setObjectName("lineThicknessSliderLabel")
        self.lineThicknessLabel.setVisible(False)

        self.lineThicknessSlider = QSlider(self.centralwidget)
        self.lineThicknessSlider.setGeometry(860, 420, 270, 20)
        self.lineThicknessSlider.setRange(1, 10)
        self.lineThicknessSlider.setValue(5)
        self.lineThicknessSlider.setOrientation(Qt.Horizontal)
        self.lineThicknessSlider.setInvertedAppearance(False)
        self.lineThicknessSlider.setObjectName("lineThicknessSlider")
        self.lineThicknessSlider.valueChanged.connect(self.lineThicknessSliderChanged)
        self.lineThicknessSlider.setVisible(False)

        # Heat map options
        self.heatmapInterpSliderLabel = QLabel(self.centralwidget)
        self.heatmapInterpSliderLabel.setGeometry(QRect(920, 390, 170, 20))
        font = QFont()
        font.setPointSize(10)
        self.heatmapInterpSliderLabel.setFont(font)
        self.heatmapInterpSliderLabel.setObjectName("Interpolate threshold")
        self.heatmapInterpSliderLabel.setVisible(False)

        self.heatmapInterpSlider = QSlider(self. centralwidget)
        self.heatmapInterpSlider.setGeometry(QRect(860, 420, 270, 20))
        self.heatmapInterpSlider.setMaximum(99)
        self.heatmapInterpSlider.setOrientation(Qt.Horizontal)
        self.heatmapInterpSlider.setInvertedAppearance(False)
        self.heatmapInterpSlider.setObjectName("heatmapInterpSlider")
        self.heatmapInterpSlider.valueChanged.connect(self.heatmapInterpSliderChanged)
        self.heatmapInterpSlider.setVisible(False)

        # Star map options
        self.convexHullCheckbox = QCheckBox(self.centralwidget)
        self.convexHullCheckbox.setGeometry(QRect(830, 380, 120, 30))
        self.convexHullCheckbox.setTristate(False)
        self.convexHullCheckbox.setObjectName("convexHullCheckbox")
        self.convexHullCheckbox.setVisible(False)
        self.convexHullCheckbox.stateChanged.connect(self.convexHullCheckboxChanged)

        self.angularColorCheckbox = QCheckBox(self.centralwidget)
        self.angularColorCheckbox.setGeometry(QRect(830, 400, 120, 30))
        self.angularColorCheckbox.setTristate(False)
        self.angularColorCheckbox.setObjectName("angularColorCheckbox")
        self.angularColorCheckbox.setVisible(False)
        self.angularColorCheckbox.setChecked(True)
        self.angularColorCheckbox.stateChanged.connect(self.angularColorChanged)

        self.interpolateColorCheckbox = QCheckBox(self.centralwidget)
        self.interpolateColorCheckbox.setGeometry(QRect(830, 420, 120, 30))
        self.interpolateColorCheckbox.setTristate(False)
        self.interpolateColorCheckbox.setObjectName("interpolateColorCheckbox")
        self.interpolateColorCheckbox.setVisible(False)
        self.interpolateColorCheckbox.setChecked(True)
        self.interpolateColorCheckbox.stateChanged.connect(self.interpolateColorChanged)

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
        self.viewsOpenGL = QAction(MainWindow)
        self.viewsOpenGL.setObjectName("viewsTrailGL")
        self.viewsOpenGL.triggered.connect(lambda: self.switchWidget(1))
        self.viewsHeatmap = QAction(MainWindow)
        self.viewsHeatmap.setObjectName("viewsHeatGL")
        self.viewsHeatmap.triggered.connect(lambda: self.switchWidget(2))
        self.viewsStarMap = QAction(MainWindow)
        self.viewsStarMap.setObjectName("viewsStarMap")
        self.viewsStarMap.triggered.connect(lambda: self.switchWidget(3))

        self.fileReset = QAction(MainWindow)
        self.fileReset.setObjectName("fileReset")
        # self.fileReset.action.connect(self.reset)
        self.fileSave = QAction(MainWindow)
        self.fileSave.setObjectName("fileSave")
        self.menuFile.addAction(self.fileReset)
        self.menuFile.addAction(self.fileSave)
        self.menuViews.addAction(self.viewsBasic)
        self.menuViews.addAction(self.viewsOpenGL)
        self.menuViews.addAction(self.viewsHeatmap)
        self.menuViews.addAction(self.viewsStarMap)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuViews.menuAction())

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.WidgetTitle.setText(_translate("MainWindow", "Scatter plot"))

        self.checkBox1.setText(_translate("MainWindow", "Add constant"))
        self.checkBox2.setText(_translate("MainWindow", "Dim. removal"))
        self.checkBox3.setText(_translate("MainWindow", "Jitter (unord.)"))
        self.checkBox4.setText(_translate("MainWindow", "Perturbation4"))
        self.radioButton1.setText(_translate("MainWindow", "Add constant"))
        self.radioButton2.setText(_translate("MainWindow", "Dim. removal"))
        self.radioButton3.setText(_translate("MainWindow", "Jitter (unord.)"))
        self.radioButton4.setText(_translate("MainWindow", "Perturbation4"))

        self.perturbSelectedButton.setText(_translate("MainWindow", "Randomize selected"))
        self.resetButton.setText(_translate("MainWindow", "Reset"))
        self.computeButton.setText(_translate("MainWindow", "Compute visualization"))
        self.perturbationSlidersLabel.setText(_translate("MainWindow", "Perturbations"))

        # Configuration labels
        self.configLabel.setText(_translate("MainWindow", "Configuration"))
        self.lineThicknessLabel.setText(_translate("MainWindow", "Line thickness"))
        self.heatmapInterpSliderLabel.setText(_translate("MainWindow", "Interpolate threshold"))
        self.convexHullCheckbox.setText(_translate("MainWindow", "Convex hull"))
        self.angularColorCheckbox.setText(_translate("MainWindow", "Angular color"))
        self.interpolateColorCheckbox.setText(_translate("MainWindow", "Interpolate color"))

        # Menu labels
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuViews.setTitle(_translate("MainWindow", "Views"))

        self.viewsBasic.setText(_translate("MainWindow", "Basic interface"))
        self.viewsBasic.setShortcut(_translate("MainWindow", "Ctrl+1"))
        self.viewsOpenGL.setText(_translate("MainWindow", "Trails interface"))
        self.viewsOpenGL.setShortcut(_translate("MainWindow", "Ctrl+2"))
        self.viewsHeatmap.setText(_translate("MainWindow", "Heat map interface"))
        self.viewsHeatmap.setShortcut(_translate("MainWindow", "Ctrl+3"))
        self.viewsStarMap.setText(_translate("MainWindow", "Star map interface"))
        self.viewsStarMap.setShortcut(_translate("MainWindow", "Ctrl+4"))

        self.fileReset.setText(_translate("MainWindow", "Reset"))
        self.fileReset.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.fileSave.setText(_translate("MainWindow", "Save"))
        self.fileSave.setShortcut(_translate("MainWindow", "Ctrl+S"))

    def differentPerturbation(self):
        if self.lastPerturbation is None:
            return True
        else:
            return not self.currentPerturbation() == self.lastPerturbation

    def currentPerturbation(self):
        if self.radioButton1.isChecked():
            return 1, self.horizontalSlider1.value()
        elif self.radioButton2.isChecked():
            return 2, self.horizontalSlider2.value()
        elif self.radioButton3.isChecked():
            return 3, self.horizontalSlider3.value()
        else:  # self.radioButton4.isChecked():
            return 4, self.horizontalSlider4.value()

    def computeVisualization(self):
        # If no radio button is checked
        if not self.radioButton1.isChecked() and not self.radioButton2.isChecked() and \
                not self.radioButton3.isChecked() and not self.radioButton4.isChecked():
            self.statusbar.showMessage("No perturbation is selected. Please choose one of them.")
            print("No perturbation is selected. Please choose one of them.")
            return

        currentWidgetIndex = self.stackedWidget.currentIndex()
        assert currentWidgetIndex > 0, "Error: There should be no computation button while in the scatter plot screen"

        if not self.differentPerturbation():
            self.statusbar.showMessage("Same configuration, skip computing intermediate data sets")
            print("Same configuration")
        else:
            self.statusbar.showMessage("Computing and predicting intermediate data sets per increment...")
            self.computeIntermediateDatasets()
            self.lastPerturbation = self.currentPerturbation()

        if currentWidgetIndex == 1:
            self.statusbar.showMessage("Drawing trail map...")
            print("Drawing trail map...")
            self.trailsGLWidget.paintTrailMapGL(self.predList, self.y_test, self.class_colors)
            self.trailsGLWidget.update()

        if currentWidgetIndex == 2:
            self.statusbar.showMessage("Drawing heat map...")
            print("Drawing heat map...")
            self.heatGLWidget.computeHeatMap(self.predList)
            self.heatGLWidget.update()

        if currentWidgetIndex == 3:
            self.statusbar.showMessage("Drawing star map...")
            print("Drawing star map...")
            if self.convexHullCheckbox.isChecked():
                self.starMapGLWidget.paintConvexStarMapGL(self.predList, self.y_test, self.class_colors)
            else:
                self.starMapGLWidget.paintStarMapGL(self.predList, self.y_test, self.class_colors)
            self.starMapGLWidget.update()

    def switchWidget(self, index):
        self.stackedWidget.setCurrentIndex(index)
        print("Going to stacked widget index: " + str(index))
        if index == 0:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Scatter plot"))
            self.statusbar.showMessage("Showing scatter plot.")
            self.computeButton.setVisible(False)
            self.resetButton.setVisible(True)
            self.perturbSelectedButton.setVisible(True)

            # Replace radio buttons with checkboxes
            self.radioButton1.setVisible(False)
            self.radioButton2.setVisible(False)
            self.radioButton3.setVisible(False)
            self.radioButton4.setVisible(False)
            self.checkBox1.setVisible(True)
            self.checkBox2.setVisible(True)
            self.checkBox3.setVisible(True)
            self.checkBox4.setVisible(True)

            self.configLabel.setVisible(False)
        else:
            self.resetButton.setVisible(False)
            self.perturbSelectedButton.setVisible(False)

            # Replace checkboxes with radio buttons
            self.radioButton1.setVisible(True)
            self.radioButton2.setVisible(True)
            self.radioButton3.setVisible(True)
            self.radioButton4.setVisible(True)
            self.checkBox1.setVisible(False)
            self.checkBox2.setVisible(False)
            self.checkBox3.setVisible(False)
            self.checkBox4.setVisible(False)

            self.configLabel.setVisible(True)

        if index == 1:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Trail map"))
            self.computeButton.setVisible(True)
            self.lineThicknessLabel.setVisible(True)
            self.lineThicknessSlider.setVisible(True)
        else:
            self.lineThicknessLabel.setVisible(False)
            self.lineThicknessSlider.setVisible(False)
        if index == 2:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Heat map"))
            self.computeButton.setVisible(True)
            self.heatmapInterpSliderLabel.setVisible(True)
            self.heatmapInterpSlider.setVisible(True)
        else:
            self.heatmapInterpSliderLabel.setVisible(False)
            self.heatmapInterpSlider.setVisible(False)
        if index == 3:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Star map"))
            self.computeButton.setVisible(True)
            self.convexHullCheckbox.setVisible(True)
            if not self.convexHullCheckbox.isChecked():
                self.angularColorCheckbox.setVisible(True)
                self.interpolateColorCheckbox.setVisible(True)
            else:
                self.angularColorCheckbox.setVisible(False)
                self.interpolateColorCheckbox.setVisible(False)
        else:
            self.convexHullCheckbox.setVisible(False)
            self.angularColorCheckbox.setVisible(False)
            self.interpolateColorCheckbox.setVisible(False)

    def computeIntermediateDatasets(self):
        print("Computing intermediate datasets")
        # Perturb using the checked perturbation
        if self.radioButton1.isChecked():
            max_value = self.horizontalSlider1.value()
            self.dataset.interDataOfPerturb(1, max_value)
        if self.radioButton2.isChecked():
            max_value = self.horizontalSlider2.value()
            self.dataset.interDataOfPerturb(2, max_value)
        if self.radioButton3.isChecked():
            max_value = self.horizontalSlider3.value()
            self.dataset.interDataOfPerturb(3, max_value)

        # Predict every dataset and save to predList
        self.predList = []
        for i in range(0, len(self.dataset.interDataset)):
            self.predList.append(self.model.predict(self.dataset.interDataset[i]))

    def replot(self):
        pred = self.model.predict(self.dataset.perturbed)
        items = pg.ScatterPlotItem(x=pred[:,0], y=pred[:,1], data=np.arange(len(self.dataset.perturbed)),
        pen='w', brush=self.brushes, size=10, hoverable=True, hoverPen=pg.mkPen(0, 0, 0, 255))
        self.plotWidget.clear()
        self.plotWidget.addItem(items)
        # Disable range adjustments
        self.plotWidget.setRange(None, (0, 1),(0, 1))

        self.plotWidget.setXRange(0, 1)

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
        self.dataset.jitterNoise(new_value * 0.01)
        self.replot()

    def slider4Changed(self):
        new_value = self.horizontalSlider4.value()
        self.statusbar.showMessage("Changed value of perturbation slider 4 to " + str(new_value))

    def lineThicknessSliderChanged(self):
        new_value = self.lineThicknessSlider.value()
        self.statusbar.showMessage("Interpolate value of heat map changed to " + str(new_value))
        self.trailsGLWidget.max_line_thickness = new_value

    def heatmapInterpSliderChanged(self):
        new_value = self.heatmapInterpSlider.value()
        self.statusbar.showMessage("Interpolate value of heat map changed to " + str(new_value))
        self.heatGLWidget.maxInterpValue = 1 - new_value * 0.01

    def convexHullCheckboxChanged(self, state):
        state = (state == Qt.Checked)
        self.interpolateColorCheckbox.setVisible(not state)
        self.angularColorCheckbox.setVisible(not state)
        self.starMapGLWidget.convex_hull = state

    def angularColorChanged(self, state):
        self.starMapGLWidget.angular_color = (state == Qt.Checked)

    def interpolateColorChanged(self, state):
        self.starMapGLWidget.interpolate_rays = (state == Qt.Checked)

    def randChangeSelected(self):
        if self.checkBox1.isChecked():
            self.horizontalSlider1.setValue(randint(1, self.horizontalSlider2.maximum()))
        if self.checkBox2.isChecked():
            self.horizontalSlider2.setValue(randint(1, self.horizontalSlider2.maximum()))
        if self.checkBox3.isChecked():
            self.horizontalSlider3.setValue(randint(1, self.horizontalSlider3.maximum()))
        if self.checkBox4.isChecked():
            self.horizontalSlider4.setValue(randint(1, self.horizontalSlider4.maximum()))

    def resetSelected(self):
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
        self.plotWidget.addItem(items)

    def loadData(self, MainWindow):
        print("Loading datasets...")
        X = np.load('data/X_mnist.npy')
        y = np.load('data/y_mnist.npy')
        label = "mnist-full"

        X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=10000, test_size=3000, random_state=420, stratify=y)
        self.y_test = y_test

        # Class colors
        self.class_colors = [(0.890, 0.102, 0.110),
                             (0.122, 0.471, 0.706),
                             (0.698, 0.875, 0.541),
                             (0.200, 0.627, 0.173),
                             (0.984, 0.604, 0.600),
                             (0.651, 0.808, 0.890),
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

        items = pg.ScatterPlotItem(x=pred[:,0], y=pred[:,1], data=np.arange(len(X_test)),
                                   pen='w', brush=self.brushes, size=10, hoverable=True, hoverPen=pg.mkPen(0, 0, 0, 255))

        self.plotWidget.addItem(items)

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
