# ############################################################################################## #
# Algorithm to                                                                         #
# ############################################################################################## #

import numpy as np


def compute_f_8point(pts1_o: np.ndarray, pts2_o: np.ndarray) -> np.ndarray:
    """
    :param pts1_o: Interest points [[x1, y1], ... [xn, yn]]
    :param pts2_o: Interest points [[x'1, y'1], ... [x'n, y'n]]
    :return:       Fundamental Matrix
    """

    n = pts1_o.shape[1]
    # normalize image coordinates
    x1 = pts1_o / pts1_o[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = np.dot(T1, x1)

    x2 = pts2_o / pts2_o[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = np.dot(T2, x2)

    # compute F with the normalized coordinates
    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    F = F / F[2, 2]

    # reverse normalization
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]
