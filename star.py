import colorsys
import math
import numpy as np
from scipy.spatial import ConvexHull

from PyQt5.QtWidgets import *

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class StarMapGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for star map")
        # enable the use of transparency
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.pred_list = None

        # Configuration options
        self.convex_hull = False
        self.angular_color = False
        self.eigen_color = True
        self.interpolate_rays = True
        self.global_opacity = 1
        self.zoom = 1

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(1 - self.zoom, self.zoom, 1 - self.zoom, self.zoom, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self.pred_list:
            if self.convex_hull:
                self.paintConvexStarMapGL(self.pred_list, self.labels, self.class_colors)
            else:
                self.paintStarMapGL(self.pred_list, self.labels, self.class_colors)
        else:
            self.emptyScreen()

    def euclidean(self, p1, p2):
        return math.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def emptyScreen(self):
        print("Display empty star map screen")

    # Determine color with hue and saturation computed through Eigen values and eccentricity
    def pcaColors(self, point_cloud):
        # Convert point cloud to covariance matrix
        cov_matrix = np.cov(np.array(point_cloud).T)

        # Compute Eigen values en vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Combine and order on eigen values
        eigen_list = [(x, y) for x, y in sorted(zip(eigenvalues, eigenvectors), reverse=True)]

        # Determine hue using the longest Eigenvector
        longest_eigenvector = eigen_list[0][1]
        theta = math.atan2(longest_eigenvector[0], longest_eigenvector[1]) + math.pi
        # Interpolate over pi for half the color wheel (mirrored is not used)
        hue = theta * 0.31830988618379067153776752674503

        # Determine saturation using eccentricity (Longest eigenvalue/ 2nd longest eigenvalue)
        # TODO: normalize saturation
        saturation = eigen_list[0][0] / eigen_list[1][0]
        return colorsys.hsv_to_rgb(hue, 1, 1)

    def paintConvexStarMapGL(self, pred_list, labels, class_colors):
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors

        # Loop over every spot
        for j in range(len(pred_list[0])):
            # Fill list for point cloud
            points = []
            for i in range(len(pred_list) - 1):
                points.append((pred_list[i][j][0], pred_list[i][j][1]))

            if self.eigen_color:
                brush_color = self.pcaColors(points)
            else:
                brush_color = class_colors[labels[j]]

            GL.glColor4f(brush_color[0], brush_color[1], brush_color[2], self.global_opacity)

            # Create list of convex hull indices
            hull_indices = []
            hull = ConvexHull(points)
            for index in hull.vertices:
                hull_indices.append(index)
            # Complete convex hull by adding the first index to the end
            hull_indices.append(hull.vertices[0])

            # Draw lines through the convex hull points
            GL.glBegin(GL.GL_LINES)
            for i in range(len(hull_indices) - 1):
                point = points[hull_indices[i]]
                GL.glVertex2f(point[0], point[1])
                point2 = points[hull_indices[i + 1]]
                GL.glVertex2f(point2[0], point2[1])
            GL.glEnd()
        GL.glFlush()

    def paintStarMapGL(self, pred_list, labels, class_colors):
        self.pred_list = pred_list
        self.labels = labels
        self.class_colors = class_colors

        # Loop over every spot
        for j in range(len(pred_list[0])):
            brush_color = class_colors[labels[j]]
            if self.eigen_color:
                # Fill list for point cloud
                points = []
                for i in range(len(pred_list) - 1):
                    points.append((pred_list[i][j][0], pred_list[i][j][1]))
                brush_color = self.pcaColors(points)


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
                    GL.glColor4f(1, 1, 1, self.global_opacity)
                else:
                    GL.glColor4f(brush_color[0], brush_color[1], brush_color[2], self.global_opacity)
                GL.glVertex2f(base_point[0], base_point[1])
                GL.glColor4f(brush_color[0], brush_color[1], brush_color[2], self.global_opacity)
                GL.glVertex2f(ray_edge[0], ray_edge[1])
                GL.glEnd()

        GL.glFlush()
        print("Done drawing")