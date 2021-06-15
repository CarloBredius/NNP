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
        self.maxInterpValue = 1.0
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
        if self.heat_array is None:
            print("No array to paint")
            return

        square_size = 1 / self.array_size
        for j in range(self.array_size):
            for i in range(self.array_size):
                interpolate_value = self.heat_array[i, j] / (self.maxInterpValue * self.max_value)

                GL.glColor(1, 1 - interpolate_value, 1 - interpolate_value)
                GL.glBegin(GL.GL_POLYGON)
                GL.glVertex(i * square_size, j * square_size)
                GL.glVertex((i + 1) * square_size, j * square_size)
                GL.glVertex((i + 1) * square_size, (j + 1) * square_size)
                GL.glVertex(i * square_size, (j + 1) * square_size)
                GL.glEnd()
        GL.glFlush()

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

    def createHeatMatrix(self, pred_list):
        print("Creating heat matrix...")
        self.array_size = 100
        self.heat_array = np.zeros((self.array_size, self.array_size))
        for j in range(len(pred_list[0])):
            # Loop over every location of the spot
            for i in range(len(pred_list)):
                p1 = int(pred_list[i][j][0] * self.array_size), int(pred_list[i][j][1] * self.array_size)
                #p2 = int(pred_list[i + 1][j][0] * self.array_size), int(pred_list[i + 1][j][1] * self.array_size)
                #traversed = self.rayTrace(p1, p2)
                #for cell in traversed:
                #    self.heat_array[cell] += 1
                #self.printarray(self.heat_array)
                self.heat_array[p1] += 1
        self.max_value = np.amax(self.heat_array)

    def printarray(self, array):
        print("Filled array:")
        for j in range(len(array[0])):
            print(str(j) + ": " + str(array[j]))
