from Problem2.fundamentalMatrix import compute_f
from Problem2.norm_8_point_f import compute_f_8point
import cv2


def execute_problem2():
    path = "Problem2/data/templeSparseRing/"
    img1_name = "templeSR0001.png"
    img2_name = "templeSR0002.png"

    img1_gray = cv2.imread(path + img1_name, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(path + img2_name, cv2.IMREAD_GRAYSCALE)
    # compute_f(img1_gray, img2_gray)
    compute_f_8point(img1_gray, img2_gray)
    print("Finish Problem 2! ...")

