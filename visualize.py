import OpenGL.GL as GL
import sys
from PyQt5.QtWidgets import *
import numpy as np


class OpenGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.1, 2.1, -0.1, 1.1, -1, 1)

    def paintTrailMapGL(self, pred_list, labels, class_colors):
        print("Painting trail map")
        #GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glBegin(GL.GL_LINES)
        for j in range(len(pred_list[0])):
            GL.glColor(class_colors[labels[j]])
            # color_switch.get(labels[j], "No color for index " + str(labels[j]) + " found!")
            for i in range(len(pred_list) - 1):
                GL.glVertex2f(pred_list[i][j][0], pred_list[i][j][1])
        GL.glEnd()
        #GL.glFlush()