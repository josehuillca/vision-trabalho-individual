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
from Problem2.utils import display_img, draw_points, interest_points


def putative_correspondences() -> None:
    return None


def compute_f_8point(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    :param img1: Input image  gray-scale
    :param img2: Input image  gray-scale
    :return:
    """
    pts1, pts2 = interest_points(img1, img2, ratio=0.7, num=300, display_matches=True)

    return None
