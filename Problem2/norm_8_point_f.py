# ############################################################################################## #
# Algorithm to                                                                         #
# ############################################################################################## #

import numpy as np
import cv2
from typing import Tuple, Any
from Problem2.utils import display_img, print_matrix, draw_lines, interest_points, draw_epipole_lines


def normalize_coord(coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param coord: [[x1, x2, ... xn], [y1, y2, ... yn], [w1, w2, ... wn]]
    :return:      normalized coordinates
    """
    mean = np.mean(coord[:2], axis=1)
    s = np.sqrt(2) / np.std(coord[:2])
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]])
    return np.dot(T, coord), T


def compute_f_8point(pts1_o: np.ndarray, pts2_o: np.ndarray) -> np.ndarray:
    """
    :param pts1_o: Interest points [[x1, y1], ... [xn, yn]]
    :param pts2_o: Interest points [[x'1, y'1], ... [x'n, y'n]]
    :return:       Fundamental Matrix
    """
    # get 8 interest points, Format = [[x1, y1, w1], [x2, y2, w2], ...[xn, yn, wn]]
    # The normal thing is that the points are (x/w, y/w, w/w), where w = 1
    pts1 = np.array([[x, y, 1] for x, y in pts1_o])
    pts2 = np.array([[x, y, 1] for x, y in pts2_o])
    # Format required to improve the matrix process = [[x1, x2, ... xn], [y1, y2, ... yn], [w1, w2, ... wn]]
    pts1, pts2 = pts1.T, pts2.T

    # Normalization: Transform the image coordinates
    pts1_, T1 = normalize_coord(pts1)
    pts2_, T2 = normalize_coord(pts2)
    print("Interest Points: ", pts1_.shape[1])

    # Find the fundamental matrix
    A = [[], [], [], [], [], [], [], []]
    n = pts1_.shape[1]
    for i in range(0, n):
        x, y, w = pts1_[0][i], pts1_[1][i], pts1_[2][i]
        x_, y_, w_ = pts2_[0][i], pts2_[1][i], pts2_[2][i]
        A[i] = [x_*x, x_*y, x_*w, y_*x, y_*y, y_*w, w_*x, w_*y, w_*w]

    # print_matrix(np.array(A), title="Matrix A")
    u, s, vh = np.linalg.svd(A)
    # print_matrix(np.array(vh), title="Matrix vh")

    # Let F be the last column of vh
    # Reshape into  a 3x3 matrix F_shapeu
    F_s = vh[-1].reshape(3, 3)

    # Normalize
    F_s = F_s/np.linalg.norm(F_s, ord=np.inf)
    # Constraint enforcement SVD descomposition
    u_, s_, vh_ = np.linalg.svd(F_s)
    s_n = np.zeros((3, 3))
    s_n[0][0], s_n[1][1] = s_[0], s_[1]
    F_ = np.dot(u_, np.dot(s_n, vh_))

    F_ = F_/F_[2, 2]

    # Denormalize
    F = np.dot(np.transpose(T2), np.dot(F_, T1))
    F = F/F[2, 2]

    return F
