import cmath
import concurrent.futures
import numpy as np

from PyQt5.QtWidgets import *

try:
    import OpenGL.GL as GL
except ImportError:
    print("OpenGL must be installed to run this program.")

class HeatGLWidget(QOpenGLWidget):
    def initializeGL(self):
        print("Initalize openGL for heat map")
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.data = None
        self.maxInterpValue = 1.0
        self.max_heat = 0
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

    def euclidean(self, p1, p2):
        return cmath.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))

    def findnearestneighbours(self, point, radius):
        amount = 0
        # Do not look outside of range
        for i in range(max(0, point[0] - radius), min(self.width(), point[0] + radius)):
            for j in range(max(0, point[1] - radius), min(self.height(), point[1] + radius)):
                value = self.points_array[i][j]
                # if point is 0, early out
                if value == 0:
                    continue
                eucl_dist = self.euclidean(point, (i, j))
                #print(f"first: {point}, second: {i, j}, Dist: {eucl_dist.real}")
                if eucl_dist.real <= radius:
                    amount += value
        return amount

    def computeHeat(self, point):
        x = point[0]
        y = point[1]
        amount = self.findnearestneighbours((x, y), 20)
        if amount > self.max_heat:
            self.max_heat = amount
        return point, amount

    def computeHeatMap(self, pred_list):
        self.points_array = np.zeros((self.width(), self.height()))
        for j in range(len(pred_list[0])):
            # Loop over every location of the spot
            for i in range(len(pred_list)):
                point = int(pred_list[i][j][0] * self.width()), int(pred_list[i][j][1] * self.height())
                self.points_array[point] += 1

        # Determine which points need to be processed
        pointlist = []
        for i in range(self.width()):
            for j in range(self.height()):
                pointlist.append((i, j))

        # With a concurrent thread pool
        self.heat_map = np.zeros((self.width(), self.height()))
        print("Start concurrently computing heat for each pixel...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for point, amount in executor.map(self.computeHeat, pointlist):
                self.heat_map[point] = amount

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        self.fillHeatBuffer()
        GL.glDrawPixels(self.width(), self.height(), GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                        (GL.GLubyte * len(self.data))(*self.data))

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

    def fillHeatBuffer(self):
        # Create 1D data buffer with with white color as base
        self.data = [255 for _ in range(0, self.height() * self.width() * 3)]
        heat_per_amount = 255 / self.max_heat
        for i in range(self.width()):
            for j in range(self.height()):
                if self.heat_map[i][j] > 0:
                    loc = int(3 * (i + j * self.width()))
                    interp_heat = 255 - int(self.heat_map[i][j] * heat_per_amount)
                    #self.data[loc] = 255
                    self.data[loc + 1] = interp_heat
                    self.data[loc + 2] = interp_heat


    def rayTrace(self, p1, p2):
        # Shoot a ray from p1 to p2, add every cell it traversed to a list
        traversed = [p1]
        if p1 == p2:
            # Same location, so no need to trace a ray
            return traversed
        xA, yA = p1
        xB, yB = p2
        dx, dy = xB - xA, yB - yA

        x, y = p1
        tIx = dy * (x + dx - xA) if dx != 0 else float("+inf")
        tIy = dx * (y + dy - yA) if dy != 0 else float("+inf")

        while (x, y) != p2:
            movx, movy = tIx <= tIy, tIy <= tIx
            if movx:
                x += dx
                tIx = dy * (x + dx - xA)

            if movy:
                y += dy
                tIy = dx * (y + dy - yA)

            traversed.append((x, y))
        return traversed

    def printarray(self, array):
        print("Filled array:")
        for j in range(len(array[0])):
            print(str(j) + ": " + str(array[j]))
