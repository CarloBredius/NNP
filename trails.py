import sys
from PyQt5.QtWidgets import *
import numpy as np

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class TrailsGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.zoom = 1.0
        GL.glShadeModel(GL.GL_FLAT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(1.0 - self.zoom, self.zoom, 1.0 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        self.rotX = 0;
        self.rotY = 0;
        self.zoomFlag = False

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        # TODO: check which button is pressed
        self.rotX = (event.x() - self.lastPos.x()) * 0.0001
        self.rotY = (event.y() - self.lastPos.y()) * 0.0001
        self.update()

    def mouseReleaseEvent(self, event):
        pass

    def wheelEvent(self, event):
        scroll = event.angleDelta()
        self.zoomFlag = True
        if scroll.y() > 0:  # up
            print("Scrolling up")
            self.zoom += 0.1
            self.update()
        else:  # down
            print("Scrolling down")
            self.zoom -= 0.1
            self.update()

    def paintGL(self):
        self.paintTrailMapGL(self.pred_list, self.labels, self.class_colors)

    def paintTrailMapGL(self, pred_list, labels, class_colors):
        print("Painting trail map")
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Loop over every spot
        for j in range(len(pred_list[0])):
            GL.glColor(class_colors[labels[j]])
            GL.glBegin(GL.GL_LINES)
            # Loop over every location of the spot
            for i in range(len(pred_list) - 1):
                # OpenGL needs a start and an endpoint, hence why some points will be added twice
                GL.glVertex2f(pred_list[i][j][0], pred_list[i][j][1])
                GL.glVertex2f(pred_list[i + 1][j][0], pred_list[i + 1][j][1])
            GL.glEnd()

        # Handle translation
        if self.rotX != 0 or self.rotY != 0:
            GL.glTranslated(self.rotX, -self.rotY, 0)
            self.rotX = 0
            self.rotY = 0
        # Handle scaling
        if self.zoomFlag:
            print("Zooming in or out")
            GL.glScalef(self.zoom, self.zoom, 0)
            self.zoomFlag = False

        GL.glFlush()
