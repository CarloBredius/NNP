import sys
from PyQt5.QtWidgets import *
import numpy as np

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class TrailsGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for trail map")
        # enable the use of transparency
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.pred_list = None
        self.max_line_thickness = 1
        self.rotX = 0
        self.rotY = 0
        self.zoomFlag = False
        self.zoom = 1.0
        GL.glShadeModel(GL.GL_FLAT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(1.0 - self.zoom, self.zoom, 1.0 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

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
        if self.pred_list:
            self.paintTrailMapGL(self.pred_list, self.labels, self.class_colors)
        else:
            self.emptyScreen()

    def emptyScreen(self):
        print("Display empty trail map screen")

    def paintTrailMapGL(self, pred_list, labels, class_colors):
        print("Painting trail map")
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        opacity_stepsize = 1 / len(pred_list)
        thickness_interval = int(len(pred_list) / self.max_line_thickness)

        # Loop over every spot
        for j in range(len(pred_list[0])):
            # first iteration will put thickness at 1
            line_thickness = 0
            color = class_colors[labels[j]]
            counter = 0
            # Loop over every location of the spot
            for i in range(len(pred_list) - 1):
                if counter % thickness_interval == 0:
                    line_thickness += 1
                    GL.glLineWidth(line_thickness)
                GL.glColor4f(color[0],  color[1], color[2], counter * opacity_stepsize)
                # OpenGL needs a start and an endpoint, hence why some points will be added twice
                GL.glBegin(GL.GL_LINES)
                GL.glVertex2f(pred_list[i][j][0], pred_list[i][j][1])
                GL.glVertex2f(pred_list[i + 1][j][0], pred_list[i + 1][j][1])
                GL.glEnd()
                counter += 1

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
