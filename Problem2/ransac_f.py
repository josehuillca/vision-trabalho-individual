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
from typing import Tuple, Any
from Problem2.utils import display_img, draw_points


def interest_points(img1: np.ndarray, img2: np.ndarray, display_matches: bool = False) -> Tuple[Any, Any]:
    """
    :param img1: Input image  gray-scale
    :param img2: Input image  gray-scale
    :param display_matches:
    :return:
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # putative_correspondences
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    if display_matches:
        imMatches = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
        display_img(imMatches, "Matches", (600 * 2, 600))
    return pts1, pts2


def putative_correspondences() -> None:
    return None


def compute_f_8point(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    :param img1: Input image  gray-scale
    :param img2: Input image  gray-scale
    :return:
    """
    pts1, pts2 = interest_points(img1, img2, display_matches=True)

    return None
