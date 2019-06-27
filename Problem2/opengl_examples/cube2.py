import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube():
    '''glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()'''
    NPOINTS = len(verticies)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glPointSize(5.0)
    for i in range(NPOINTS):
        glBegin(GL_POINTS)
        point = verticies[i]
        glVertex3fv(point)
        glEnd()


def point_cloud(pts: np.ndarray, p_size: float = 2.0) -> None:
    NPOINTS = len(pts)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glPointSize(p_size)
    for i in range(NPOINTS):
        glBegin(GL_POINTS)
        point = tuple(pts[i][:3])
        glVertex3fv(point)
        glEnd()


def main_cube2(pts: np.ndarray):
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    #gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gluPerspective(45, (1.0 * display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # Cube()
        point_cloud(pts)
        pygame.display.flip()
        pygame.time.wait(10)