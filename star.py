import math

from PyQt5.QtWidgets import *

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class StarMapGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for star map")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        self.zoom = 1.0
        GL.glOrtho(1.0 - self.zoom, self.zoom, 1.0 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def computeStarMap(self, pred_list):
        pass

    def paintGL(self):
        self.paintStarMapGL(self.pred_list, self.labels, self.class_colors)

    def euclidean(self, p1, p2):
        return math.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def paintStarMapGL(self, pred_list, labels, class_colors):
        print("Painting star map")
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Loop over every spot
        for j in range(len(pred_list[0])):
            color = class_colors[labels[j]]
            base_point = pred_list[0][j]
            # Loop over every location of the spot, skip the first step
            for i in range(1, len(pred_list) - 1):
                star_edge = pred_list[i][j][0], pred_list[i][j][1]
                # TODO: use theta to get a color from hsv color wheel
                dx, dy = base_point[0] - star_edge[0], base_point[1] - star_edge[1]
                theta = math.atan2(dy, dx)

                GL.glBegin(GL.GL_LINES)
                GL.glColor3f(1, 1, 1)
                GL.glVertex2f(base_point[0], base_point[1])
                GL.glColor3f(color[0], color[1], color[2])
                GL.glVertex2f(star_edge[0], star_edge[1])
                GL.glEnd()

        GL.glFlush()