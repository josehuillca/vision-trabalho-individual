import cv2
import numpy as np
from Problem2.utils import interest_points, draw_epipole_lines, print_matrix
from Problem2.norm_8_point_f import compute_f_8point
    

def compute_f(img1: np.ndarray, img2: np.ndarray, alg: str, draw_epilines: bool = True) -> None:
    """
    :param img1: Input image  gray-scale
    :param img2: Input image  gray-scale
    :param alg:  Algorithm used to calculate fundamental matrix ('8_POINT', 'RANSAC')
    :return:
    """
    pts1 = []
    pts2 = []
    F = []

    if alg == '8_POINT':
        pts1, pts2 = interest_points(img1, img2, ratio=0.4, num=8, display_matches=True)
        F = compute_f_8point(pts1, pts2)
    elif alg == 'RANSAC':
        pts1, pts2 = interest_points(img1, img2, ratio=0.7, num=300, display_matches=True)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
    else:
        print("WARNING: Algorithm('alg') can only be '8_POINT' or 'RANSAC'.")
        return None

    print_matrix(F, title="Fundamental Matrix")
    if draw_epilines:
        draw_epipole_lines(img1, img2, pts1, pts2, F)
