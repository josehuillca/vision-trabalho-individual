import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from sys import platform as _platform
if _platform == "darwin":
   # MAC OS X
   import matplotlib
   matplotlib.use('MacOSX')
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
from Problem2.utils import print_matrix, interest_points, cart2hom, load_parameters, select_inliers_points_within_img, get_all_points_matching
from Problem2.opengl_examples.cube2 import main_cube2


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
    :returns: 4 x n homogenous 3d triangulated points: [[X1, X2, ...], [Y1, Y2, ...], [Z1, Z2, ...], [W1=1, W2=1, ...]]
    """
    num_points = pts1.shape[1]
    res = np.ones((4, num_points))
    # We need the points to be in this format: [[x1,y1,w1], [x2, y2,w2], ...]
    pts1, pts2 = pts1.T, pts2.T

    for i in range(0, num_points):
        res[:, i] = triangulate_point(pts1[i], pts2[i], P1, P2)

    return res


def reconstruction_3d(path: str, Ps: List[np.ndarray], Ks: List[np.ndarray], imgs_name: List[str]):
    """
    :param path:          donde se encuentra el archivo de los parametros y las imagenes
    :param Ps:
    :param Ks:
    :param imgs_name:
    :return:
    """
    num_imgs = len(imgs_name)
    pts3D = [[], [], [], []]

    # points1, points2 = [], []
    # P1, P2 = [], []
    for i in range(0, num_imgs-1, 1):
        j = i + 1  # imagenes que usaremos
        img1_gray = cv2.imread(path + imgs_name[i], 0)
        img2_gray = cv2.imread(path + imgs_name[j], 0)

        # Preprocesing
        p = 0.75  # Porcentaje del umbral a tomar en cuenta
        th1, _ = cv2.threshold(img1_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2, _ = cv2.threshold(img2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th1, th2 = th1 * p, th2 * p
        img1_gray[img1_gray < th1] = 0
        img2_gray[img2_gray < th2] = 0

        all_points = False
        if all_points:
            pts1, pts2 = get_all_points_matching(img1_gray, img2_gray)
        else:
            # 'interest_points' return formatted points: [[x1, y1], [x2, y2], ...[xn, yn]]
            pts1, pts2, _ = interest_points(img1_gray, img2_gray, ratio=0.65, num_max=500, display_matches='0')
            #
            # pts1, pts2 = select_inliers_points_within_img(img1_gray, img2_gray, pts1, pts2)
        # Verificamos que haya puntos
        if len(pts1) != 0:
            # we required formatted point: [[x1, x2, ...xn], [y1, y2, ...yn]]
            pts1, pts2 = pts1.T, pts2.T
            # homogeneous coordinates: [[x1, x2, ...xn], [y1, y2, ...yn], [1, 1, ...1]]
            points1 = cart2hom(pts1)
            points2 = cart2hom(pts2)
            # normalizing coordinates
            normalize = False
            if normalize:
                points1n = np.dot(np.linalg.inv(Ks[i]), points1)
                points2n = np.dot(np.linalg.inv(Ks[j]), points2)
            else:
                points1n, points2n = points1, points2
            # ------------
            P1 = Ps[i]  # np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
            P2 = Ps[j]

            tripoints3d = linear_triangulation(points1n, points2n, P1, P2)

            # automatically center 3d object
            '''x3d = np.array([(max(tripoints3d[0]) + min(tripoints3d[0])) / 2.] * len(tripoints3d[0]))
            y3d = np.array([(max(tripoints3d[1]) + min(tripoints3d[1])) / 2.] * len(tripoints3d[1]))
            z3d = np.array([(max(tripoints3d[2]) + min(tripoints3d[2])) / 2.] * len(tripoints3d[2]))
            tripoints3d[0] = tripoints3d[0] - x3d
            tripoints3d[1] = tripoints3d[1] - y3d
            tripoints3d[2] = tripoints3d[2] - z3d'''

            pts3D = np.hstack([pts3D, tripoints3d])
            # print("Min: ", min(tripoints3d.T[2]),)
            # print("Max: ", max(tripoints3d.T[2]),)


    # Pasando Test de calidad x*P*X=0 e x'*P'X=0
    '''print("x*PX=0 ")
    a = np.dot(P1, np.transpose([pts3D.T[0]]))
    r = np.cross(points1.T[0], np.array([a[0][0], a[1][0], a[2][0]]))
    print(r)'''

    # PLOT 3D POINTS
    '''fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(pts3D[0], pts3D[1], pts3D[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()'''

    return pts3D

