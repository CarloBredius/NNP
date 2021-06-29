from PyQt5.QtWidgets import *

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class StarMapGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for star map")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glShadeModel(GL.GL_FLAT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(1.0 - self.zoom, self.zoom, 1.0 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def computeStarMap(self, pred_list):
        pass

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
