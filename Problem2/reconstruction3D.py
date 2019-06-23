import numpy as np
import cv2
from typing import List, Tuple
from Problem2.utils import print_matrix, interest_points


def load_images_names(file: str) -> List:
    """
    :param file: Path + name file(i.e. *_good_silhouette_images.txt)
    :return:
    """
    lines = [line.rstrip('\n') for line in open(file, 'r+')]
    return lines


def load_matrix_P(file: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    :param file: Path + name file(i.e. *_par.txt)
    :return:
    """
    imgs_name = []
    Ps = []
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

    print_matrix(Ps[0], title="Ps[0]")
    return Ps, imgs_name


def triangulate_point(pt1: np.ndarray, pt2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    :param pt1: [x, y]
    :param pt2: [x', y']
    :param P1:  matrix P 3x4
    :param P2:  matrix P 3x4
    :return:    [X, Y, Z]
    """
    x, y = pt1[0], pt1[1]
    x_, y_ = pt2[0], pt2[1]

    A = [[], [], [], []]
    A[0] = [x*P1[2][0]-P1[0][0], x*P1[2][1]-P1[0][1], x*P1[2][2]-P1[0][2], x*P1[2][3]-P1[0][3]]
    A[1] = [y*P1[2][0]-P1[1][0], y*P1[2][1]-P1[1][1], y*P1[2][2]-P1[1][2], y*P1[2][3]-P1[1][3]]
    A[2] = [x_*P2[2][0]-P2[0][0], x_*P2[2][1]-P2[0][1], x_*P2[2][2]-P2[0][2], x_*P2[2][3]-P2[0][3]]
    A[3] = [y_*P2[2][0]-P2[1][0], y_*P2[2][1]-P2[1][1], y_*P2[2][2]-P2[1][2], y_*P2[2][3]-P2[1][3]]

    u, s, vh = np.linalg.svd(A)
    X = vh[-1]
    X = X / X[3]
    print(X)
    return np.array(X[:3])


def triangulate(pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    :param pts1:  [[x1, y1], [x2, y2], ...[xn, yn]]
    :param pts2:  [[x'1, y'1], [x'2, y'2], ...[x'n, y'n]]
    :param P1:
    :param P2:
    :return:
    """
    n = len(pts1)
    points3D = []
    for i in range(0, n):
        X = triangulate_point(pts1[i], pts2[i], P1, P2)
        points3D.append(X)
    print(len(points3D))

    return np.array(points3D)


def reconstruction_3d(path: str, par_file_name: str):
    """
    :param path:          donde se encuentra el archivo de los parametros y las imagenes
    :param par_file_name:
    :return:
    """
    # Loading matrix P and name images
    Ps, imgs_name = load_matrix_P(path + par_file_name)

    i, j = 0, 1     # imagenes que usaremos

    img1 = cv2.imread(path + imgs_name[i], 0)
    img2 = cv2.imread(path + imgs_name[j], 0)
    pts1, pts2 = interest_points(img1, img2, ratio=0.8, num=500, display_matches=False)
    pts3D = triangulate(pts1, pts2, Ps[i], Ps[j])

    return pts1, pts2, Ps, pts3D

