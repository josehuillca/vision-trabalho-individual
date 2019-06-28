import pygame
from pygame.locals import *
import numpy as np
import cv2

from OpenGL.GL import *
from OpenGL.GLU import *

from sys import platform as _platform
if _platform == "darwin":
   # MAC OS X
   import matplotlib
   matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import cart2hom, interest_points
from typing import Tuple, List

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
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


def point_cloud(pts):
    num_pts = len(pts)
    glPointSize(2);
    glBegin(GL_POINTS)
    for i in range(0, num_pts):
        glVertex3fv(pts[i][:3])
    glEnd()


def load_parameters(file: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    :param file: Path + name file(i.e. *_par.txt)
    :return: matrix P, intrinsic parameters, images name
    """
    imgs_name = []  # images names
    Ps = []         # List of matrices P
    Ks = []         # List of intrinsic parameters
    with open(file, 'r+') as fo:
        num_imgs = int(fo.readline().rstrip('\n'))
        for i in range(0, num_imgs):
            line = fo.readline().rstrip('\n').split(' ')
            imgs_name.append(line[0])
            K = np.array([float(k) for k in line[1:10]]).reshape(3, 3)      # matrix shape = 3*3
            R = np.array([float(r) for r in line[10:19]]).reshape(3, 3)     # matrix shape = 3*3
            t = np.array([float(t) for t in line[19:]])                     # matrix shape = 3*1
            I_t = np.hstack((np.identity(3), np.transpose([t])))
            P = np.dot(K, np.dot(R, I_t))
            Ps.append(P)
            Ks.append(K)
    # all intrinsic parameters(Ks) are equal
    return Ps, Ks, imgs_name


def triangulate_point(pt1: np.ndarray, pt2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    :param pt1: [x, y, w=1]
    :param pt2: [x', y', w=1]
    :param P1:  matrix P 3x4
    :param P2:  matrix P 3x4
    :return:    [X, Y, Z, W=1]
    """
    x, y = pt1[0], pt1[1]
    x_, y_ = pt2[0], pt2[1]

    A = np.asarray([
        [x*P1[2][0]-P1[0][0], x*P1[2][1]-P1[0][1], x*P1[2][2]-P1[0][2], x*P1[2][3]-P1[0][3]],
        [y*P1[2][0]-P1[1][0], y*P1[2][1]-P1[1][1], y*P1[2][2]-P1[1][2], y*P1[2][3]-P1[1][3]],
        [x_*P2[2][0]-P2[0][0], x_*P2[2][1]-P2[0][1], x_*P2[2][2]-P2[0][2], x_*P2[2][3]-P2[0][3]],
        [y_*P2[2][0]-P2[1][0], y_*P2[2][1]-P2[1][1], y_*P2[2][2]-P2[1][2], y_*P2[2][3]-P2[1][3]]
    ])

    u, s, vh = np.linalg.svd(A)
    X = vh[-1, :4]
    return X / X[3]


def linear_triangulation(pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Linear triangulation (Hartley ch 12.2 pg 312) to find the 3D point X
    where p1 = m1 * X and p2 = m2 * X. Solve AX = 0.
    :param pts1: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param pts2: 2D points in homo. or catesian coordinates. Shape (3 x n)
    :param P1: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :param P2: Camera matrices associated with p1 and p2. Shape (3 x 4)
    :returns: 4 x n homogenous 3d triangulated points [[X, Y, Z, W=1], ...]
    """
    num_points = pts1.shape[1]
    res = np.ones((4, num_points))
    # We need the points to be in this format: [[x1,y1,w1], [x2, y2,w2], ...]
    pts1, pts2 = pts1.T, pts2.T

    for i in range(0, num_points):
        res[:, i] = triangulate_point(pts1[i], pts2[i], P1, P2)

    return res


def reconstruction_3d(path: str, par_file_name: str):
    """
    :param path:          donde se encuentra el archivo de los parametros y las imagenes
    :param par_file_name:
    :return:
    """
    # Loading matrix P and name images
    Ps, Ks, imgs_name = load_parameters(path + par_file_name)
    pts3D = [[],[],[],[]]
    for i in range(0, len(imgs_name)-1):
        j = i + 1     # imagenes que usaremos
        img1_gray = cv2.imread(path + imgs_name[i], 0)
        img2_gray = cv2.imread(path + imgs_name[j], 0)
        # 'interest_points' return formatted points: [[x1, y1], [x2, y2], ...[xn, yn]]
        pts1, pts2 = interest_points(img1_gray, img2_gray, ratio=0.8, num_max=500, display_matches='None')
        # we required formatted point: [[x1, x2, ...xn], [y1, y2, ...yn]]
        pts1, pts2 = pts1.T, pts2.T
        # homogeneous coordinates: [[x1, x2, ...xn], [y1, y2, ...yn], [1, 1, ...1]]
        points1 = cart2hom(pts1)
        points2 = cart2hom(pts2)
        # normalizing coordinates
        points1n = np.dot(np.linalg.inv(Ks[i]), points1)
        points2n = np.dot(np.linalg.inv(Ks[j]), points2)

        # ------------
        P1 = Ps[i]  #np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P2 = Ps[j]

        tripoints3d = linear_triangulation(points1n, points2n, P1, P2)

        # automatically center 3d object
        x3d = np.array([(max(tripoints3d[0]) + min(tripoints3d[0])) / 2.] * len(tripoints3d[0]))
        y3d = np.array([(max(tripoints3d[1]) + min(tripoints3d[1])) / 2.] * len(tripoints3d[1]))
        z3d = np.array([(max(tripoints3d[2]) + min(tripoints3d[2])) / 2.] * len(tripoints3d[2]))
        tripoints3d[0] = tripoints3d[0]-x3d
        tripoints3d[1] = tripoints3d[1]-y3d
        tripoints3d[2] = tripoints3d[2]-z3d

        pts3D = np.hstack([pts3D, tripoints3d])
        #print("Min: ", min(tripoints3d.T[2]),)
        #print("Max: ", max(tripoints3d.T[2]),)
    return pts3D


def IdentityMat44():
    return np.matrix(np.identity(4), copy=False, dtype='float32')


def main(pts):
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
    glTranslatef(0, 0, -0.001)
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
                    ry = 0.1
                elif event.key == pygame.K_LEFT:
                    ry = -0.1
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
        #Cube()
        point_cloud(pts)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":
    path = "Problem2/data/dinoSparseRing/"
    parameters_file = 'dinoSR_par.txt'
    pts3D = reconstruction_3d(path, parameters_file)

    print(pts3D.shape)
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')

    ax.plot(pts3D[0], pts3D[1], pts3D[2], 'b.')
    # glTranslatef(-(max.x + min.x)/2.0f,-(max.y + min.y)/2.0f,-(max.z + min.z)/2.0f);
    '''x3d = np.array([(max(pts3D[0][62:111]) + min(pts3D[0][62:111]))/2.]*len(pts3D[0][62:111]))
    y3d = np.array([(max(pts3D[1][62:111]) + min(pts3D[1][62:111]))/2.]*len(pts3D[1][62:111]))
    z3d = np.array([(max(pts3D[2][62:111]) + min(pts3D[2][62:111]))/2.]*len(pts3D[2][62:111]))
    print(len(x3d), len(pts3D[0][62:111]))
    x = (pts3D[0][62:111] - x3d)
    y = (pts3D[1][62:111] - y3d)
    z = (pts3D[2][62:111] - z3d)
    ax.plot(x, y, z, 'b.')
    x3d = np.array([(max(pts3D[0][:60]) + min(pts3D[0][:60])) / 2.]*len(pts3D[2][:60]))
    y3d = np.array([(max(pts3D[1][:60]) + min(pts3D[1][:60])) / 2.]*len(pts3D[2][:60]))
    z3d = np.array([(max(pts3D[2][:60]) + min(pts3D[2][:60])) / 2.]*len(pts3D[2][:60]))
    ax.plot(pts3D[0][:60]-x3d, pts3D[1][:60]-y3d, pts3D[2][:60]-z3d, 'r.')
    x3d = np.array([(max(pts3D[0][131:160]) + min(pts3D[0][131:160])) / 2.] * len(pts3D[2][131:160]))
    y3d = np.array([(max(pts3D[1][131:160]) + min(pts3D[1][131:160])) / 2.] * len(pts3D[2][131:160]))
    z3d = np.array([(max(pts3D[2][131:160]) + min(pts3D[2][131:160])) / 2.] * len(pts3D[2][131:160]))
    ax.plot(pts3D[0][131:160]-x3d, pts3D[1][131:160]-y3d, pts3D[2][131:160]-z3d, 'k.')'''
    # ax.plot_trisurf(tripoints3d[0], tripoints3d[1], tripoints3d[2], linewidth=0.2, antialiased=True)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    #ax.view_init(elev=135, azim=90)
    plt.show()

    #main(pts3D.T)
    #print(pts3D[0][:62], pts3D[1][:62], pts3D[2][:62])
