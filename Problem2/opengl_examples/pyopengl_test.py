import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


vertices=(
    (1,-1,-1),
    (1,1,-1),
    (-1,1,-1),
    (-1,-1,-1),
    (0,0,1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (1,4),
    (1,2),
    (2,4),
    (2,3), # (2,3)
    (3,4)
)


def Pyramid():
    glLineWidth(5)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
            glColor3f(0,1,0)
    glEnd()


def main():
    pygame.init()
    display = (400, 400)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50)

    glTranslatef(0,0,-5)

    clock = pygame.time.Clock()
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(2, 1, 1, 3)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Pyramid()
        pygame.display.flip()


'''
import sys
import random #for random numbers
from OpenGL.GL import * #for definition of points
from OpenGL.GLU import *
from OpenGL.GLUT import * #for visualization in a window
import numpy as np

AMOUNT = 10
DIMENSION = 3

def changePoints(points):
    for i in range(len(points)):
        points[i] = random.uniform(-1.0, 1.0)
    print(points)
    return points

def displayPoints(points):
    vbo=GLuint(0) # init the Buffer in Python!
    glGenBuffers(1, vbo) # generate a buffer for the vertices
    glBindBuffer(GL_ARRAY_BUFFER, vbo) #bind the vertex buffer
    glBufferData(GL_ARRAY_BUFFER,sys.getsizeof(points), points, GL_STREAM_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, vbo) #bind the vertex buffer

    glEnableClientState(GL_VERTEX_ARRAY) # enable Vertex Array
    glVertexPointer(DIMENSION, GL_FLOAT,0, ctypes.cast(0, ctypes.c_void_p))
    glBindBuffer(GL_ARRAY_BUFFER, vbo) #bind the vertex buffer
    glDrawArrays(GL_POINTS, 0, AMOUNT)
    glDisableClientState(GL_VERTEX_ARRAY) # disable the Vertex Array
    glDeleteBuffers(1, vbo)

##creates Points
def Point():

    points = np.arange(AMOUNT*3, dtype = np.float32)
    points = changePoints(points)

    #Visualization
    displayPoints(points)


##clears the color and depth Buffer, call Point() and swap the buffers of the current window
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    Point()
    glutSwapBuffers()

def main():
    ##initials GLUT
    glutInit(sys.argv)
    #sets the initial display mode (selects a RGBA mode window; selects a double buffered window; selects a window with a depth buffer)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    #defines the size of the Window
    glutInitWindowSize(800, 800)
    #creates a window with title
    glutCreateWindow(b'Points') #!string as title is causing a error, because underneath the PyOpenGL call is an old-school C function expecting ASCII text. Solution: pass the string in byte format.
    glutDisplayFunc(display) #sets the display callback for the current window.
    glutMainLoop() #enters the GLUT event processing loop.
'''