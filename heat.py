from PyQt5.QtWidgets import *
import numpy as np

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class HeatGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for heat map")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.heat_array = None
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
        if self.heat_array is not None:
            square_size = 1 / self.array_size
            for j in range(self.array_size):
                for i in range(self.array_size):
                    # TODO: instead of only the points, the lines need to influence the heatmap
                    # TODO: maybe use a third of the max_value
                    interpolate_value = self.heat_array[i, j] / self.max_value
                    GL.glColor(1, 1 - interpolate_value, 1 - interpolate_value)
                    GL.glBegin(GL.GL_POLYGON)
                    GL.glVertex(i * square_size, j * square_size)
                    GL.glVertex((i + 1) * square_size, j * square_size)
                    GL.glVertex((i + 1) * square_size, (j + 1) * square_size)
                    GL.glVertex(i * square_size, (j + 1) * square_size)
                    GL.glEnd()
            GL.glFlush()
        else:
            print("No array to paint")

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

    def createHeatMatrix(self, pred_list):
        print("Creating heat matrix...")
        self.array_size = 100
        self.heat_array = np.zeros((self.array_size, self.array_size))
        for j in range(len(pred_list[0])):
            # Loop over every location of the spot
            for i in range(len(pred_list)):
                point = pred_list[i][j]
                self.heat_array[int(point[0] * self.array_size), int(point[1] * self.array_size)] += 1
        self.max_value = np.amax(self.heat_array)

    def printarray(self, array):
        print("filled array:")
        for j in range(len(array[0])):
            print(array[j])
