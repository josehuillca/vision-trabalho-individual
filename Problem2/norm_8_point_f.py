# ############################################################################################## #
# Algorithm to automatically estimate the fundamental matrix between two images using RANSAC.    #
# (i) Interest points                                                                            #
# (ii) Putative correspondences                                                                  #
# (iii) RANSAC robust estimation                                                                 #
# (iv) Non-linear estimation                                                                     #
# (v) Guided matching                                                                            #
# ############################################################################################## #

import numpy as np
import cv2
from typing import Tuple, Any, List
from Problem2.utils import display_img


def normalize_coord(shape: Tuple[int, int], coord: Tuple[int, int]) -> List:
    """
    :param shape: (height, width)
    :param coord: (x, y)
    :return:      (x, y) normalized
    """
    h, w = shape[0], shape[1]
    T = [[w+h, 0, w/2],
         [0, w+h, h/2],
         [0, 0, 1]]
    T = np.linalg.inv(np.array(T))
    x = np.transpose(np.array([[coord[0], coord[1], 1]]))
    x_ = np.dot(T, x)
    x_ = x_.flatten().tolist()
    return [x_[1], x_[0]]


def interest_points(img1: np.ndarray, img2: np.ndarray, num: int = 8, display_matches: bool = False) -> Tuple[Any, Any]:
    """ Asumimos que sift nos dara mas de 'num=8' puntos de interes
    :param img1:            Input image  gray-scale
    :param img2:            Input image  gray-scale
    :param num:             cantidad de puntos de interes a ser extraida
    :param display_matches: Draw matches
    :return:                [[x1,y1], ...[xn,yn]], [[x1,y1], ...[xn,yn]]
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    k = 0
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.4 * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            if k > num:
                break
            k = k + 1

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Draw top matches
    if display_matches:
        im_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        display_img(im_matches, "Matches", (600 * 2, 600))
    return pts1, pts2


def compute_f_8point(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    :param img1: Input image  gray-scale
    :param img2: Input image  gray-scale
    :return:
    """
    # get 8 interest points
    pts1, pts2 = interest_points(img1, img2, num=8, display_matches=True)
    print(len(pts1), len(pts2))

    return None
