import colorsys
import math
import numpy as np
from PyQt5.QtWidgets import *

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class TrailsGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for trail map")
        # Enable the use of transparency
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.recompute = True
        self.pred_list = None
        self.diff_list = None
        self.min_diff = 999999
        self.max_diff = 0

        # Configuration options
        self.twodnd = False
        self.angular_color = False
        self.max_line_thickness = 5
        self.global_opacity = 1
        self.rotX = 0
        self.rotY = 0
        self.zoomFlag = False
        self.zoom = 1

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
            if self.twodnd:
                self.paintDifferenceMapGL(self.pred_list, self.inter_dataset)
            else:
                self.paintTrailMapGL(self.pred_list, self.labels, self.class_colors)
        else:
            self.emptyScreen()

    def emptyScreen(self):
        print("Display empty trail map screen")

    def euclidean(self, p1, p2):
        return math.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def euclideanNd(self, p1, p2):
        sum = 0
        for i in range(len(p1)):
            sum += pow(p2[i] - p1[i], 2)
        return math.sqrt(sum)

    def computeDistanceList(self, pred_list, inter_dataset):
        assert len(inter_dataset) == len(pred_list)
        assert len(inter_dataset[0]) == len(pred_list[0])

        diff_list = np.zeros((len(pred_list), len(pred_list[0])))

        # Compute difference between distances and keep track of max difference
        for j in range(len(inter_dataset[0])):
            for i in range(len(inter_dataset) - 1):
                dist_2d = self.euclidean(pred_list[i][j], pred_list[i + 1][j])
                dist_nd = self.euclideanNd(inter_dataset[i][j], inter_dataset[i + 1][j])
                diff = dist_2d / dist_nd
                diff_list[i][j] = diff

                if diff < self.min_diff:
                    self.min_diff = diff
                if diff > self.max_diff:
                    self.max_diff = diff

        print("Computed distance list")
        self.diff_list = diff_list

    def paintDifferenceMapGL(self, pred_list, inter_dataset):
        assert len(pred_list) == len(inter_dataset)

        print("Painting difference trail map")
        self.pred_list = pred_list
        self.inter_dataset = inter_dataset
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        thickness_interval = int(len(pred_list) / self.max_line_thickness)

        if self.recompute:
            print("Computing difference list")
            self.computeDistanceList(pred_list, inter_dataset)
            self.recompute = False

        # Loop over every spot
        for j in range(len(pred_list[0])):
            # First iteration will put thickness at 1
            line_thickness = 0

            # Loop over every location of the spot
            for i in range(len(pred_list) - 1):
                if i % thickness_interval == 0:
                    line_thickness += 1
                    GL.glLineWidth(line_thickness)

                p1, p2 = pred_list[i][j], pred_list[i + 1][j]

                # Determine color by using the normalized difference
                dist_diff = self.diff_list[i][j]

                interp_color = 1 - ((dist_diff - self.min_diff) / (self.max_diff - self.min_diff))
                brush_color = (1, interp_color, interp_color, 1)

                GL.glColor4f(brush_color[0],  brush_color[1], brush_color[2], brush_color[3] * self.global_opacity)
                # OpenGL needs a start and an endpoint, hence why some points will be added twice
                GL.glBegin(GL.GL_LINES)
                GL.glVertex2f(p1[0], p1[1])
                GL.glVertex2f(p2[0], p2[1])
                GL.glEnd()

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
            # First iteration will put thickness at 1
            line_thickness = 0
            brush_color = class_colors[labels[j]]
            # Loop over every location of the spot
            for i in range(len(pred_list) - 1):
                if i % thickness_interval == 0:
                    line_thickness += 1
                    GL.glLineWidth(line_thickness)

                p1, p2 = pred_list[i][j], pred_list[i + 1][j]
                if self.angular_color:
                    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
                    theta = math.atan2(dy, dx)
                    # Normalize theta with 1/2π
                    brush_color = colorsys.hsv_to_rgb(theta * 0.15915494309189533576888376337251, 1, 1)

                # OpenGL needs a start and an endpoint, hence why some points will be added twice
                GL.glBegin(GL.GL_LINES)
                GL.glColor4f(brush_color[0],  brush_color[1], brush_color[2], i * opacity_stepsize * self.global_opacity)
                GL.glVertex2f(p1[0], p1[1])
                GL.glColor4f(brush_color[0],  brush_color[1], brush_color[2], (i + 1) * opacity_stepsize * self.global_opacity)
                GL.glVertex2f(p2[0], p2[1])
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
