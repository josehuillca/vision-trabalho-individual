import numpy as np
import pygame

from OpenGL.GL import *
from OpenGL.GLU import *


def point_cloud(pts: np.ndarray, p_size: float = 2.0) -> None:
    NPOINTS = len(pts)
    glEnable(GL_PROGRAM_POINT_SIZE)
    glPointSize(p_size)
    for i in range(NPOINTS):
        glBegin(GL_POINTS)
        point = tuple(pts[i][:3])
        glVertex3fv(point)
        glEnd()


def IdentityMat44():
    return np.matrix(np.identity(4), copy=False, dtype='float32')


def main_cube2(pts):
    pygame.init()
    display = (600,600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF|pygame.OPENGL)

    tx = 0
    ty = 0
    tz = 0
    ry = 0

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    view_mat = IdentityMat44()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -0.4)
    glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
    glLoadIdentity()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_a:
                    tx = 0.01
                elif event.key == pygame.K_d:
                    tx = -0.01
                elif event.key == pygame.K_w:
                    tz = 0.01
                elif event.key == pygame.K_s:
                    tz = -0.01
                elif event.key == pygame.K_RIGHT:
                    ry = 0.3
                elif event.key == pygame.K_LEFT:
                    ry = -0.3
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a and tx > 0:
                    tx = 0
                elif event.key == pygame.K_d and tx < 0:
                    tx = 0
                elif event.key == pygame.K_w and tz > 0:
                    tz = 0
                elif event.key == pygame.K_s and tz < 0:
                    tz = 0
                elif event.key == pygame.K_RIGHT and ry > 0:
                    ry = 0.0
                elif event.key == pygame.K_LEFT and ry < 0:
                    ry = 0.0

        glPushMatrix()
        glLoadIdentity()
        glTranslatef(tx, ty, tz)
        glRotatef(ry, 0, 1, 0)
        glMultMatrixf(view_mat)

        glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        point_cloud(pts)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)
