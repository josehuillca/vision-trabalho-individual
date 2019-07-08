import cv2
import numpy as np
from Problem2.utils import interest_points, draw_epipole_lines, print_matrix, cart2hom
from Problem2.norm_8_point_f import compute_f_8point
from Problem2.ransac_f import compute_f_ransac
    

def compute_f(img1_file: str, img2_file: str, alg: str, T: np.ndarray = None, draw_epilines: bool = True) -> np.ndarray:
    """
    :param img1_file:   Input image  name
    :param img2_file:   Input image  name
    :param alg:         Algorithm used to calculate fundamental matrix ('8_POINT', 'RANSAC')
    :param T:           Matrix de parametros intrinsecos
    :param draw_epilines:
    :return:
    """
    img1 = cv2.imread(img1_file, 0)     # gray-scale
    img2 = cv2.imread(img2_file, 0)     # gray-scale

    if alg == '8_POINT':
        pts1, pts2, _ = interest_points(img1, img2, ratio=0.4, num_max=8, display_matches='cv2')
        pts1_ = cart2hom(pts1.T)
        pts2_ = cart2hom(pts2.T)
        F = compute_f_8point(pts1_, pts2_)

    elif alg == 'RANSAC':
        pts1, pts2, _ = interest_points(img1, img2, ratio=0.7, num_max=300, display_matches='cv2')
        F, _ = compute_f_ransac(pts1.T, pts2.T)

    else:
        print("WARNING: Algorithm('alg') can only be '8_POINT' or 'RANSAC'.")
        return np.array([])

    print_matrix(F, title="Fundamental Matrix", c_type='e')
    if draw_epilines:
        draw_epipole_lines(img1, img2, pts1, pts2, F)

    return F