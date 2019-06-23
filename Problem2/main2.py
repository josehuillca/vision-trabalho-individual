import cv2
import numpy as np
from Problem2.fundamentalMatrix import compute_f
from Problem2.reconstruction3D import reconstruction_3d
from Problem2.pyopengl_test import main
from Problem2.opengl_examples.cube2 import main_cube2
from Problem2.opengl_examples.drawing_sierpinski_gasket import main_sierpinski
from Problem2.utils import print_matrix


def execute_problem2():
    path = "Problem2/data/dinoSparseRing/"   # "Problem2/data/templeSparseRing/"
    img1_name = "dinoSR0002.png"
    img2_name = "dinoSR0003.png"

    # img1_gray = cv2.imread(path + img1_name, cv2.IMREAD_GRAYSCALE)
    # img2_gray = cv2.imread(path + img2_name, cv2.IMREAD_GRAYSCALE)
    # compute_f(img1_gray, img2_gray, alg='RANSAC')

    parameters_file = 'dinoSR_par.txt'
    pts1, pts2, Ps, pts3D = reconstruction_3d(path, parameters_file)
    # ------------------------------

    P1 = np.eye(3, 4, dtype=np.float32)
    P2 = np.eye(3, 4, dtype=np.float32)
    P2[0, 3] = -1
    N = 5
    points3d = np.empty((4, N), np.float32)
    points3d[:3, :] = np.random.randn(3, N)
    points3d[3, :] = 1

    points1 = P1 @ points3d
    points1 = points1[:2, :] / points1[2, :]
    points1[:2, :] += np.random.randn(2, N) * 1e-2

    points2 = P2 @ points3d
    points2 = points2[:2, :] / points2[2, :]
    points2[:2, :] += np.random.randn(2, N) * 1e-2

    print(pts1.T.shape)
    points3d_reconstr = cv2.triangulatePoints(Ps[0], Ps[1], pts1.T, pts2.T)
    #points3d_reconstr /= points3d_reconstr[3, :]
    print(points3d_reconstr[3])
    #main_cube2(points3d_reconstr[:3].T)
    print("Finish Problem 2! ...")

