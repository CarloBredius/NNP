import colorsys
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

        # Configuration options
        self.angular_color = True
        self.interpolate_rays = True

        self.zoom = 1.0
        self.pred_list = None
        GL.glLoadIdentity()
        GL.glOrtho(1.0 - self.zoom, self.zoom, 1.0 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self.pred_list:
            self.paintStarMapGL(self.pred_list, self.labels, self.class_colors)
        else:
            self.emptyScreen()

    def euclidean(self, p1, p2):
        return math.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def emptyScreen(self):
        print("Display empty star map screen")

    def paintStarMapGL(self, pred_list, labels, class_colors):
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors

        # Loop over every spot
        for j in range(len(pred_list[0])):
            brush_color = class_colors[labels[j]]
            base_point = pred_list[0][j]
            # Loop over every location of the spot, skip the first step
            for i in range(1, len(pred_list) - 1):
                ray_edge = pred_list[i][j][0], pred_list[i][j][1]

                if self.angular_color:
                    dx, dy = base_point[0] - ray_edge[0], base_point[1] - ray_edge[1]
                    theta = math.atan2(dy, dx)
                    # Normalize theta with 1/2π
                    brush_color = colorsys.hsv_to_rgb(theta * 0.15915494309189533576888376337251, 1, 1)

                GL.glBegin(GL.GL_LINES)
                if self.interpolate_rays:
                    GL.glColor3f(1, 1, 1)
                else:
                    GL.glColor3f(brush_color[0], brush_color[1], brush_color[2])
                GL.glVertex2f(base_point[0], base_point[1])
                GL.glColor3f(brush_color[0], brush_color[1], brush_color[2])
                GL.glVertex2f(ray_edge[0], ray_edge[1])
                GL.glEnd()

        GL.glFlush()
        print("Done drawing")