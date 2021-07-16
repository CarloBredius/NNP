# -*- coding: utf-8 -*-
# embed pyqtgraph in pyqt5: https://www.mfitzp.com/tutorials/embed-pyqtgraph-custom-widgets-qt-app/
# TODO: All tooltips

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

        # Stacked widget, combining all visualizations
        self.stackedWidget = QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QRect(10, 10, 800, 800))
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidget.addWidget(self.plotWidget)
        self.stackedWidget.addWidget(self.trailsGLWidget)
        self.stackedWidget.addWidget(self.heatGLWidget)
        self.stackedWidget.addWidget(self.starMapGLWidget)

        # For checking if intermediate results can be reused or not
        self.recompute = True

        # Widget Title
        self.WidgetTitle = QLabel(self.centralwidget)
        self.WidgetTitle.setGeometry(QRect(920, 30, 200, 35))
        font = QFont()
        font.setPointSize(20)
        self.WidgetTitle.setFont(font)
        self.WidgetTitle.setObjectName("WidgetTitle")

        # Perturbations label
        self.perturbationSlidersLabel = QLabel(self.centralwidget)
        self.perturbationSlidersLabel.setGeometry(QRect(930, 80, 170, 25))
        font = QFont()
        font.setPointSize(14)
        self.perturbationSlidersLabel.setFont(font)
        self.perturbationSlidersLabel.setObjectName("perturbation_title")

        # Translation
        self.translationLabel = QLabel(self.centralwidget)
        self.translationLabel.setGeometry(QRect(950, 105, 170, 25))
        font = QFont()
        font.setPointSize(10)
        self.translationLabel.setFont(font)
        self.translationLabel.setObjectName("translationLabel")
        self.translationAmountSlider = QSlider(self.centralwidget)
        self.translationAmountSlider.setGeometry(QRect(920, 130, 270, 20))
        self.translationAmountSlider.setMaximum(100)
        self.translationAmountSlider.setOrientation(Qt.Horizontal)
        self.translationAmountSlider.setInvertedAppearance(False)
        self.translationAmountSlider.setObjectName("translationSlider")
        self.translationAmountSlider.valueChanged.connect(self.translationAmountChanged)
        self.translationAmountCheckbox = QCheckBox(self.centralwidget)
        self.translationAmountCheckbox.setGeometry(QRect(830, 130, 90, 20))
        self.translationAmountCheckbox.setTristate(False)
        self.translationAmountCheckbox.setObjectName("translationAmountCheckbox")
        self.translationAmountRadioButton = QRadioButton(self.centralwidget)
        self.translationAmountRadioButton.setGeometry(QRect(830, 130, 90, 20))
        self.translationAmountRadioButton.setObjectName("translationAmountRadioButton")
        self.translationAmountRadioButton.setVisible(False)
        self.translationAmountRadioButton.setChecked(True)

        self.translationDimensionSlider = QSlider(self.centralwidget)
        self.translationDimensionSlider.setGeometry(QRect(920, 150, 270, 20))
        self.translationDimensionSlider.setMaximum(100)
        self.translationDimensionSlider.setOrientation(Qt.Horizontal)
        self.translationDimensionSlider.setInvertedAppearance(False)
        self.translationDimensionSlider.setObjectName("translationDimensionSlider")
        self.translationDimensionSlider.valueChanged.connect(self.translationDimentionsChanged)
        self.translationDimensionCheckbox = QCheckBox(self.centralwidget)
        self.translationDimensionCheckbox.setGeometry(QRect(830, 150, 90, 20))
        self.translationDimensionCheckbox.setTristate(False)
        self.translationDimensionCheckbox.setObjectName("translationDimensionCheckbox")
        self.translationDimensionRadioButton = QRadioButton(self.centralwidget)
        self.translationDimensionRadioButton.setGeometry(QRect(830, 150, 90, 20))
        self.translationDimensionRadioButton.setObjectName("translationDimensionRadioButton")
        self.translationDimensionRadioButton.setVisible(False)

        # Scale
        self.scaleLabel = QLabel(self.centralwidget)
        self.scaleLabel.setGeometry(QRect(970, 165, 170, 25))
        font = QFont()
        font.setPointSize(10)
        self.scaleLabel.setFont(font)
        self.scaleLabel.setObjectName("scaleLabel")
        self.scaleAmountSlider = QSlider(self.centralwidget)
        self.scaleAmountSlider.setGeometry(QRect(920, 190, 270, 20))
        self.scaleAmountSlider.setMaximum(100)
        self.scaleAmountSlider.setSliderPosition(100)
        self.scaleAmountSlider.setOrientation(Qt.Horizontal)
        self.scaleAmountSlider.setInvertedAppearance(False)
        self.scaleAmountSlider.setObjectName("scaleAmountSlider")
        self.scaleAmountSlider.valueChanged.connect(self.scaleAmountChanged)
        self.scaleAmountCheckbox = QCheckBox(self.centralwidget)
        self.scaleAmountCheckbox.setGeometry(QRect(830, 190, 90, 20))
        self.scaleAmountCheckbox.setTristate(False)
        self.scaleAmountCheckbox.setObjectName("scaleAmountCheckbox")
        self.scaleAmountRadioButton = QRadioButton(self.centralwidget)
        self.scaleAmountRadioButton.setGeometry(QRect(830, 190, 90, 20))
        self.scaleAmountRadioButton.setObjectName("scaleAmountRadioButton")
        self.scaleAmountRadioButton.setVisible(False)

        self.scaleDimensionSlider = QSlider(self.centralwidget)
        self.scaleDimensionSlider.setGeometry(QRect(920, 210, 270, 20))
        self.scaleDimensionSlider.setMaximum(100)
        self.scaleDimensionSlider.setOrientation(Qt.Horizontal)
        self.scaleDimensionSlider.setInvertedAppearance(False)
        self.scaleDimensionSlider.setObjectName("scaleDimensionSlider")
        self.scaleDimensionSlider.valueChanged.connect(self.scaleDimensionChanged)
        self.scaleDimensionCheckbox = QCheckBox(self.centralwidget)
        self.scaleDimensionCheckbox.setGeometry(QRect(830, 210, 90, 20))
        self.scaleDimensionCheckbox.setTristate(False)
        self.scaleDimensionCheckbox.setObjectName("scaleDimensionCheckbox")
        self.scaleDimensionRadioButton = QRadioButton(self.centralwidget)
        self.scaleDimensionRadioButton.setGeometry(QRect(830, 210, 90, 20))
        self.scaleDimensionRadioButton.setObjectName("scaleDimensionRadioButton")
        self.scaleDimensionRadioButton.setVisible(False)

        # Jitter
        self.jitterSlider = QSlider(self.centralwidget)
        self.jitterSlider.setGeometry(QRect(920, 250, 270, 20))
        self.jitterSlider.setMaximum(100)
        self.jitterSlider.setOrientation(Qt.Horizontal)
        self.jitterSlider.setInvertedAppearance(False)
        self.jitterSlider.setObjectName("jitterSlider")
        self.jitterSlider.valueChanged.connect(self.jitterSliderChanged)
        self.jitterCheckbox = QCheckBox(self.centralwidget)
        self.jitterCheckbox.setGeometry(QRect(830, 250, 90, 20))
        self.jitterCheckbox.setObjectName("jitterCheckbox")
        self.jitterRadioButton = QRadioButton(self.centralwidget)
        self.jitterRadioButton.setGeometry(QRect(830, 250, 90, 20))
        self.jitterRadioButton.setObjectName("jitterRadioButton")
        self.jitterRadioButton.setVisible(False)

        # Global scale
        self.globalScaleSlider = QSlider(self.centralwidget)
        self.globalScaleSlider.setGeometry(QRect(920, 280, 270, 20))
        self.globalScaleSlider.setMaximum(100)
        self.globalScaleSlider.setSliderPosition(100)
        self.globalScaleSlider.setOrientation(Qt.Horizontal)
        self.globalScaleSlider.setInvertedAppearance(False)
        self.globalScaleSlider.setObjectName("globalScaleSlider")
        self.globalScaleSlider.valueChanged.connect(self.globalScaleSliderChanged)
        self.globalScaleCheckbox = QCheckBox(self.centralwidget)
        self.globalScaleCheckbox.setGeometry(QRect(830, 280, 90, 20))
        self.globalScaleCheckbox.setObjectName("globalScaleCheckbox")
        self.globalScaleRadioButton = QRadioButton(self.centralwidget)
        self.globalScaleRadioButton.setGeometry(QRect(830, 280, 90, 20))
        self.globalScaleRadioButton.setObjectName("globalScaleRadioButton")
        self.globalScaleRadioButton.setVisible(False)

        # Permute
        self.permuteSlider = QSlider(self.centralwidget)
        self.permuteSlider.setGeometry(QRect(920, 310, 270, 20))
        self.permuteSlider.setMaximum(100)
        self.permuteSlider.setOrientation(Qt.Horizontal)
        self.permuteSlider.setInvertedAppearance(False)
        self.permuteSlider.setObjectName("permuteSlider")
        self.permuteSlider.valueChanged.connect(self.permuteSliderChanged)
        self.permuteCheckbox = QCheckBox(self.centralwidget)
        self.permuteCheckbox.setGeometry(QRect(830, 310, 90, 20))
        self.permuteCheckbox.setObjectName("permuteCheckbox")
        self.permuteRadioButton = QRadioButton(self.centralwidget)
        self.permuteRadioButton.setGeometry(QRect(830, 310, 90, 20))
        self.permuteRadioButton.setObjectName("permuteRadioButton")
        self.permuteRadioButton.setVisible(False)

        # Dimension removal
        self.dimRemovalSlider = QSlider(self.centralwidget)
        self.dimRemovalSlider.setGeometry(QRect(920, 340, 270, 20))
        self.dimRemovalSlider.setMaximum(100)
        self.dimRemovalSlider.setOrientation(Qt.Horizontal)
        self.dimRemovalSlider.setInvertedAppearance(False)
        self.dimRemovalSlider.setObjectName("dimRemovalSlider")
        self.dimRemovalSlider.valueChanged.connect(self.dimRemovalSliderChanged)
        self.dimRemovalCheckbox = QCheckBox(self.centralwidget)
        self.dimRemovalCheckbox.setGeometry(QRect(830, 340, 90, 20))
        self.dimRemovalCheckbox.setObjectName("dimRemovalCheckbox")
        self.dimRemovalRadioButton = QRadioButton(self.centralwidget)
        self.dimRemovalRadioButton.setGeometry(QRect(830, 340, 90, 20))
        self.dimRemovalRadioButton.setObjectName("dimRemovalRadioButton")
        self.dimRemovalRadioButton.setVisible(False)

        # Group the perturbations radio buttons
        self.radioButtons = QButtonGroup(self.centralwidget)
        self.radioButtons.addButton(self.translationAmountRadioButton, 1)
        self.radioButtons.addButton(self.translationDimensionRadioButton, 2)
        self.radioButtons.addButton(self.scaleAmountRadioButton, 3)
        self.radioButtons.addButton(self.scaleDimensionRadioButton, 4)
        self.radioButtons.addButton(self.jitterRadioButton, 5)
        self.radioButtons.addButton(self.globalScaleRadioButton, 6)
        self.radioButtons.addButton(self.permuteRadioButton, 7)
        self.radioButtons.addButton(self.dimRemovalRadioButton, 8)

        # Random perturbation and reset button
        self.perturbSelectedButton = QPushButton(self.centralwidget)
        self.perturbSelectedButton.setGeometry(QRect(850, 370, 150, 50))
        self.perturbSelectedButton.setObjectName("perturbSelectedButton")
        self.perturbSelectedButton.setToolTip("Randomly set all checked perturbations")
        self.perturbSelectedButton.clicked.connect(self.randChangeSelected)
        self.resetButton = QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QRect(1020, 370, 110, 50))
        self.resetButton.setObjectName("reset_button")
        self.resetButton.setToolTip("Set all checked perturbations back to 0")
        self.resetButton.clicked.connect(self.resetSelected)

        # Configuration label
        self.configLabel = QLabel(self.centralwidget)
        self.configLabel.setGeometry(QRect(930, 370, 170, 25))
        font = QFont()
        font.setPointSize(14)
        self.configLabel.setFont(font)
        self.configLabel.setObjectName("ConfigLabel")
        self.configLabel.setVisible(False)

        # General configuration options
        self.globalOpacitySliderLabel = QLabel(self.centralwidget)
        self.globalOpacitySliderLabel.setGeometry(QRect(960, 650, 170, 20))
        font = QFont()
        font.setPointSize(10)
        self.globalOpacitySliderLabel.setFont(font)
        self.globalOpacitySliderLabel.setObjectName("globalOpacitySliderLabel")
        self.globalOpacitySliderLabel.setVisible(False)

        self.globalOpacitySlider = QSlider(self.centralwidget)
        self.globalOpacitySlider.setGeometry(QRect(870, 680, 270, 20))
        self.globalOpacitySlider.setMaximum(100)
        self.globalOpacitySlider.setSliderPosition(100)
        self.globalOpacitySlider.setOrientation(Qt.Horizontal)
        self.globalOpacitySlider.setInvertedAppearance(False)
        self.globalOpacitySlider.setObjectName("globalOpacitySlider")
        self.globalOpacitySlider.valueChanged.connect(self.globalOpacitySliderChanged)
        self.globalOpacitySlider.setVisible(False)

        # Trail map options
        self.trailAngularColorCheckbox = QCheckBox(self.centralwidget)
        self.trailAngularColorCheckbox.setGeometry(QRect(830, 410, 120, 30))
        self.trailAngularColorCheckbox.setTristate(False)
        self.trailAngularColorCheckbox.setObjectName("trailAngularColorCheckbox")
        self.trailAngularColorCheckbox.setVisible(False)
        self.trailAngularColorCheckbox.stateChanged.connect(self.trailAngularColorCheckboxChanged)

        self.lineThicknessLabel = QLabel(self.centralwidget)
        self.lineThicknessLabel.setGeometry(QRect(950, 440, 170, 20))
        font = QFont()
        font.setPointSize(10)
        self.lineThicknessLabel.setFont(font)
        self.lineThicknessLabel.setObjectName("lineThicknessSliderLabel")
        self.lineThicknessLabel.setVisible(False)

        self.lineThicknessSlider = QSlider(self.centralwidget)
        self.lineThicknessSlider.setGeometry(860, 470, 270, 20)
        self.lineThicknessSlider.setRange(1, 10)
        self.lineThicknessSlider.setPageStep(2)
        self.lineThicknessSlider.setValue(5)
        self.lineThicknessSlider.setOrientation(Qt.Horizontal)
        self.lineThicknessSlider.setInvertedAppearance(False)
        self.lineThicknessSlider.setObjectName("lineThicknessSlider")
        self.lineThicknessSlider.valueChanged.connect(self.lineThicknessSliderChanged)
        self.lineThicknessSlider.setVisible(False)

        # Heat map options
        self.heatmapInterpSliderLabel = QLabel(self.centralwidget)
        self.heatmapInterpSliderLabel.setGeometry(QRect(930, 410, 170, 20))
        font = QFont()
        font.setPointSize(10)
        self.heatmapInterpSliderLabel.setFont(font)
        self.heatmapInterpSliderLabel.setObjectName("heatmapInterpSliderLabel")
        self.heatmapInterpSliderLabel.setVisible(False)

        self.heatmapInterpSlider = QSlider(self. centralwidget)
        self.heatmapInterpSlider.setGeometry(QRect(870, 440, 270, 20))
        self.heatmapInterpSlider.setMaximum(99)
        self.heatmapInterpSlider.setOrientation(Qt.Horizontal)
        self.heatmapInterpSlider.setInvertedAppearance(False)
        self.heatmapInterpSlider.setObjectName("heatmapInterpSlider")
        self.heatmapInterpSlider.valueChanged.connect(self.heatmapInterpSliderChanged)
        self.heatmapInterpSlider.setVisible(False)

        # Star map options
        self.convexHullCheckbox = QCheckBox(self.centralwidget)
        self.convexHullCheckbox.setGeometry(QRect(830, 400, 120, 30))
        self.convexHullCheckbox.setTristate(False)
        self.convexHullCheckbox.setObjectName("convexHullCheckbox")
        self.convexHullCheckbox.setVisible(False)
        self.convexHullCheckbox.stateChanged.connect(self.convexHullCheckboxChanged)

        self.starAngularColorCheckbox = QCheckBox(self.centralwidget)
        self.starAngularColorCheckbox.setGeometry(QRect(830, 420, 120, 30))
        self.starAngularColorCheckbox.setTristate(False)
        self.starAngularColorCheckbox.setObjectName("angularColorCheckbox")
        self.starAngularColorCheckbox.setVisible(False)
        self.starAngularColorCheckbox.setDisabled(False)
        self.starAngularColorCheckbox.setChecked(True)
        self.starAngularColorCheckbox.stateChanged.connect(self.starAngularColorCheckboxChanged)

        self.interpolateColorCheckbox = QCheckBox(self.centralwidget)
        self.interpolateColorCheckbox.setGeometry(QRect(830, 440, 120, 30))
        self.interpolateColorCheckbox.setTristate(False)
        self.interpolateColorCheckbox.setObjectName("interpolateColorCheckbox")
        self.interpolateColorCheckbox.setVisible(False)
        self.interpolateColorCheckbox.setDisabled(False)
        self.interpolateColorCheckbox.setChecked(True)
        self.interpolateColorCheckbox.stateChanged.connect(self.interpolateColorCheckboxChanged)

        # Compute current configuration button
        self.computeButton = QPushButton(self.centralwidget)
        self.computeButton.setGeometry(QRect(935, 725, 150, 50))
        self.computeButton.setObjectName("ComputeButton")
        self.computeButton.setToolTip("Compute visualization for the current configuration")
        self.computeButton.clicked.connect(self.computeVisualization)
        self.computeButton.setVisible(False)

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

        self.perturbationSlidersLabel.setText(_translate("MainWindow", "Perturbations"))
        self.translationLabel.setText(_translate("MainWindow", "Translation"))
        self.translationAmountCheckbox.setText(_translate("MainWindow", "Amount"))
        self.translationAmountRadioButton.setText(_translate("MainWindow", "Amount"))
        self.translationDimensionRadioButton.setText(_translate("MainWindow", "Dimensions"))
        self.translationDimensionCheckbox.setText(_translate("MainWindow", "Dimensions"))

        self.scaleLabel.setText(_translate("MainWindow", "Scale"))
        self.scaleAmountCheckbox.setText(_translate("MainWindow", "Amount"))
        self.scaleAmountRadioButton.setText(_translate("MainWindow", "Amount"))
        self.scaleDimensionCheckbox.setText(_translate("MainWindow", "Dimensions"))
        self.scaleDimensionRadioButton.setText(_translate("MainWindow", "Dimensions"))

        self.jitterCheckbox.setText(_translate("MainWindow", "Jitter (unord)"))
        self.jitterRadioButton.setText(_translate("MainWindow", "Jitter (unord)"))

        self.globalScaleCheckbox.setText(_translate("MainWindow", "Global scale"))
        self.globalScaleRadioButton.setText(_translate("MainWindow", "Global scale"))

        self.permuteCheckbox.setText(_translate("MainWindow", "Permute"))
        self.permuteRadioButton.setText(_translate("MainWindow", "Permute"))

        self.dimRemovalCheckbox.setText(_translate("MainWindow", "Dim. removal"))
        self.dimRemovalRadioButton.setText(_translate("MainWindow", "Dim. removal"))

        self.perturbSelectedButton.setText(_translate("MainWindow", "Randomize selected"))
        self.resetButton.setText(_translate("MainWindow", "Reset"))
        self.computeButton.setText(_translate("MainWindow", "Compute visualization"))

        # Configuration labels
        self.configLabel.setText(_translate("MainWindow", "Configuration"))
        self.trailAngularColorCheckbox.setText(_translate("MainWindow", "Angular color"))
        self.lineThicknessLabel.setText(_translate("MainWindow", "Line thickness"))
        self.heatmapInterpSliderLabel.setText(_translate("MainWindow", "Interpolate threshold"))
        self.convexHullCheckbox.setText(_translate("MainWindow", "Convex hull"))
        self.starAngularColorCheckbox.setText(_translate("MainWindow", "Angular color"))
        self.interpolateColorCheckbox.setText(_translate("MainWindow", "Interpolate color"))
        self.globalOpacitySliderLabel.setText(_translate("MainWindow", "Global opacity"))

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

    def switchWidget(self, index):
        self.stackedWidget.setCurrentIndex(index)
        print("Going to stacked widget index: " + str(index))
        # When a config is used in multiple widgets, set to False should only be done at first code appearance
        if index == 0:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Scatter plot"))
            self.statusbar.showMessage("Showing scatter plot.")
            self.computeButton.setVisible(False)
            self.resetButton.setVisible(True)
            self.perturbSelectedButton.setVisible(True)

            # Replace radio buttons with checkboxes
            self.translationAmountCheckbox.setVisible(True)
            self.translationAmountRadioButton.setVisible(False)
            self.translationDimensionCheckbox.setVisible(True)
            self.translationDimensionRadioButton.setVisible(False)

            self.scaleAmountCheckbox.setVisible(True)
            self.scaleAmountRadioButton.setVisible(False)
            self.scaleDimensionCheckbox.setVisible(True)
            self.scaleDimensionRadioButton.setVisible(False)

            self.jitterCheckbox.setVisible(True)
            self.jitterRadioButton.setVisible(False)

            self.globalScaleCheckbox.setVisible(True)
            self.globalScaleRadioButton.setVisible(False)

            self.permuteCheckbox.setVisible(True)
            self.permuteRadioButton.setVisible(False)

            self.dimRemovalCheckbox.setVisible(True)
            self.dimRemovalRadioButton.setVisible(False)

            self.configLabel.setVisible(False)
        else:
            self.resetButton.setVisible(False)
            self.perturbSelectedButton.setVisible(False)

            # Replace checkboxes with radio buttons
            self.translationAmountCheckbox.setVisible(False)
            self.translationAmountRadioButton.setVisible(True)
            self.translationDimensionCheckbox.setVisible(False)
            self.translationDimensionRadioButton.setVisible(True)

            self.scaleAmountRadioButton.setVisible(True)
            self.scaleAmountCheckbox.setVisible(False)
            self.scaleDimensionRadioButton.setVisible(True)
            self.scaleDimensionCheckbox.setVisible(False)

            self.jitterCheckbox.setVisible(False)
            self.jitterRadioButton.setVisible(True)

            self.globalScaleCheckbox.setVisible(False)
            self.globalScaleRadioButton.setVisible(True)

            self.permuteCheckbox.setVisible(False)
            self.permuteRadioButton.setVisible(True)

            self.dimRemovalCheckbox.setVisible(False)
            self.dimRemovalRadioButton.setVisible(True)

            self.configLabel.setVisible(True)
        if index == 1:
            self.WidgetTitle.setText(QCoreApplication.translate("MainWindow", "Trail map"))
            self.computeButton.setVisible(True)
            self.trailAngularColorCheckbox.setVisible(True)
            self.lineThicknessLabel.setVisible(True)
            self.lineThicknessSlider.setVisible(True)
            self.globalOpacitySliderLabel.setVisible(True)
            self.globalOpacitySlider.setVisible(True)
        else:
            self.trailAngularColorCheckbox.setVisible(False)
            self.lineThicknessLabel.setVisible(False)
            self.lineThicknessSlider.setVisible(False)
            self.globalOpacitySliderLabel.setVisible(False)
            self.globalOpacitySlider.setVisible(False)
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
            self.starAngularColorCheckbox.setVisible(True)
            self.interpolateColorCheckbox.setVisible(True)
            self.globalOpacitySliderLabel.setVisible(True)
            self.globalOpacitySlider.setVisible(True)
            if not self.convexHullCheckbox.isChecked():
                self.starAngularColorCheckbox.setDisabled(False)
                self.interpolateColorCheckbox.setDisabled(False)
            else:
                self.starAngularColorCheckbox.setDisabled(True)
                self.interpolateColorCheckbox.setDisabled(True)
        else:
            self.convexHullCheckbox.setVisible(False)
            self.starAngularColorCheckbox.setVisible(False)
            self.interpolateColorCheckbox.setVisible(False)

    def computeVisualization(self):
        currentWidgetIndex = self.stackedWidget.currentIndex()
        assert currentWidgetIndex > 0, "Error: There should be no computation button while in the scatter plot screen"

        if self.recompute:
            self.statusbar.showMessage("Computing and predicting intermediate data sets per increment...")
            self.computeIntermediateDatasets()
            self.heatGLWidget.heatmapFilled = False
            self.recompute = False
        else:
            self.statusbar.showMessage("Same configuration, skip computing intermediate data sets")
            print("Same configuration, skip computing intermediate data sets")

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

    def globalOpacitySliderChanged(self):
        new_value = self.globalOpacitySlider.value() * 0.01
        self.statusbar.showMessage("Changed value of global opacity slider to " + "{:.2f}".format(new_value))
        self.trailsGLWidget.global_opacity = new_value
        self.starMapGLWidget.global_opacity = new_value

    def computeIntermediateDatasets(self):
        checked_button_index = self.radioButtons.checkedId()
        max_value = 0

        # Perturb using the checked perturbation
        if checked_button_index == 1:
            max_value = self.translationAmountSlider.value()
        elif checked_button_index == 2:
            max_value = self.translationDimensionSlider.value()
        elif checked_button_index == 3:
            max_value = self.scaleAmountSlider.value()
        elif checked_button_index == 4:
            max_value = self.scaleDimensionSlider.value()
        elif checked_button_index == 5:
            max_value = self.jitterSlider.value()
        elif checked_button_index == 6:
            max_value = self.globalScaleSlider.value()
        elif checked_button_index == 7:
            max_value = self.permuteSlider.value()
        elif checked_button_index == 8:
            max_value = self.dimRemovalSlider.value()
        else:
            print(f"Error: Unknown radio button with index {checked_button_index} in computeIntermediateDatasets.")

        print(f"Computing intermediate datasets for perturbation {checked_button_index} with range 0-{max_value}.")
        self.dataset.interDataOfPerturb(checked_button_index, max_value)

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

    def translationAmountChanged(self):
        new_value = self.translationAmountSlider.value()
        self.statusbar.showMessage("Changed translation perturbation: amount to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.translation(new_value, self.translationDimensionSlider.value())
            self.replot()

    def translationDimentionsChanged(self):
        new_value = self.translationDimensionSlider.value()
        self.statusbar.showMessage("Changed translation perturbation: number of random dimensions to " + str(new_value))
        self.recompute = True

        if self.stackedWidget.currentIndex() == 0:
            self.dataset.translation(self.translationAmountSlider.value(), new_value)
            self.replot()

    def scaleAmountChanged(self):
        new_value = self.scaleAmountSlider.value()
        self.statusbar.showMessage("Changed scale perturbation: number of random dimensions to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.scale(new_value, self.scaleDimensionSlider.value())
            self.replot()

    def scaleDimensionChanged(self):
        new_value = self.scaleDimensionSlider.value()
        self.statusbar.showMessage("Changed scale perturbation: amount to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.scale(self.scaleAmountSlider.value(), new_value)
            self.replot()

    def jitterSliderChanged(self):
        new_value = self.jitterSlider.value()
        self.statusbar.showMessage("Changed value of jitter slider to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.jitterNoise(new_value * 0.01)
            self.replot()

    def globalScaleSliderChanged(self):
        new_value = self.globalScaleSlider.value()
        self.statusbar.showMessage("Changed value of global scale slider to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.scaleAllPerturbations(new_value)
            self.replot()

    def permuteSliderChanged(self):
        new_value = self.permuteSlider.value()
        self.statusbar.showMessage("Changed value of permute slider to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.permute(new_value)
            self.replot()

    def dimRemovalSliderChanged(self):
        new_value = self.dimRemovalSlider.value()
        self.statusbar.showMessage("Changed value of dimension removal slider to " + str(new_value))
        self.recompute = True
        if self.stackedWidget.currentIndex() == 0:
            self.dataset.removeRandomDimensions(new_value)
            self.replot()

    def trailAngularColorCheckboxChanged(self, state):
        self.trailsGLWidget.angular_color = (state == Qt.Checked)

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
        self.interpolateColorCheckbox.setDisabled(state)
        self.starAngularColorCheckbox.setDisabled(state)
        self.starMapGLWidget.convex_hull = state

    def starAngularColorCheckboxChanged(self, state):
        self.starMapGLWidget.angular_color = (state == Qt.Checked)

    def interpolateColorCheckboxChanged(self, state):
        self.starMapGLWidget.interpolate_rays = (state == Qt.Checked)

    def randChangeSelected(self):
        if self.translationAmountCheckbox.isChecked():
            self.translationAmountSlider.setValue(randint(0, self.translationAmountSlider.maximum()))
        if self.translationDimensionCheckbox.isChecked():
            self.translationDimensionSlider.setValue(randint(0, self.translationDimensionSlider.maximum()))

        if self.scaleAmountCheckbox.isChecked():
            self.scaleAmountSlider.setValue(randint(0, self.translationAmountSlider.maximum()))
        if self.scaleDimensionCheckbox.isChecked():
            self.scaleDimensionSlider.setValue(randint(0, self.translationAmountSlider.maximum()))

        if self.jitterCheckbox.isChecked():
            self.jitterSlider.setValue(randint(0, self.jitterSlider.maximum()))

        if self.globalScaleCheckbox.isChecked():
            self.globalScaleSlider.setValue(randint(0, self.globalScaleSlider.maximum()))

        if self.permuteCheckbox.isChecked():
            self.permuteSlider.setValue(randint(0, self.permuteSlider.maximum()))

        if self.dimRemovalCheckbox.isChecked():
            self.dimRemovalSlider.setValue(randint(0, self.dimRemovalSlider.maximum()))

    def resetSelected(self):
        if self.translationAmountCheckbox.isChecked():
            self.translationAmountSlider.setValue(0)
        if self.translationDimensionCheckbox.isChecked():
            self.translationDimensionSlider.setValue(0)

        if self.scaleAmountCheckbox.isChecked():
            self.scaleAmountSlider.setValue(100)
        if self.scaleDimensionCheckbox.isChecked():
            self.scaleDimensionSlider.setValue(0)

        if self.jitterCheckbox.isChecked():
            self.jitterSlider.setValue(0)

        if self.globalScaleCheckbox.isChecked():
            self.globalScaleSlider.setValue(100)

        if self.permuteCheckbox.isChecked():
            self.permuteSlider.setValue(0)

        if self.dimRemovalCheckbox.isChecked():
            self.dimRemovalSlider.setValue(0)

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

        items = pg.ScatterPlotItem(x=pred[:,0], y=pred[:,1], data=y_test,
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
