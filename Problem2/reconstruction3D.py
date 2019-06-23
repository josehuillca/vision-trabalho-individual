import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
from Problem2.utils import print_matrix, interest_points, cart2hom


def load_images_names(file: str) -> List:
    """
    :param file: Path + name file(i.e. *_good_silhouette_images.txt)
    :return:
    """
    lines = [line.rstrip('\n') for line in open(file, 'r+')]
    return lines


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

    i, j = 2, 1     # imagenes que usaremos
    img1_gray = cv2.imread(path + imgs_name[i], 0)
    img2_gray = cv2.imread(path + imgs_name[j], 0)
    # 'interest_points' return formatted points: [[x1, y1], [x2, y2], ...[xn, yn]]
    pts1, pts2 = interest_points(img1_gray, img2_gray, ratio=0.8, num_max=500, display_matches=False)
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

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(tripoints3d[0], tripoints3d[1], tripoints3d[2], 'b.')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()

    return None

