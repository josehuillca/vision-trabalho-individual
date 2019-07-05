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

# Exemplo de puntos [[X, Y, Z, W]]
# [[-0.03291531 -0.01312066 -0.04070807  1.        ]
#  [ 0.00281686  0.06297159 -0.02050089  1.        ]
#  [-0.00248463  0.06254201 -0.02561252  1.        ]]
def main_cube2(pts):
    pygame.init()
    display = (600,600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF|pygame.OPENGL)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    view_mat = IdentityMat44()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -0.5)
    glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)
    glLoadIdentity()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glPushMatrix()
        glLoadIdentity()
        glMultMatrixf(view_mat)

        glGetFloatv(GL_MODELVIEW_MATRIX, view_mat)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        point_cloud(pts)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)
