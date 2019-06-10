import cv2
from Problem1.utils import display_img, crop_img
import numpy as np
from typing import Tuple
from math import floor
import progressbar


def ssd(a: np.ndarray, b: np.ndarray) -> float:
    """ Sum Square Difference ('a' and 'b' have to be the same size)
    :param a: Array numpy
    :param b: Array numpy
    :return: sum square difference
    """
    return ((a-b)**2).sum()


def sad(a: np.ndarray, b: np.ndarray) -> float:
    """ Sum of Added Difference
    :param a:
    :param b:
    :return:
    """
    return abs(a-b).sum()


def disparity_map_gray_scale(img1: np.ndarray, img2: np.ndarray, janela: Tuple[int, int] = (3,3), measure: str = 'SSD') -> None:
    """ Given a pair of rectified stereo images, use the matching window to calculate the
        disparity map of the second image relative to the first.
    :param img1: Input image BGR (source)
    :param img2: Input image BGR (reference)
    :param janela: Janela de busca (x,y), es mejor que sea cuadratica(x=y)
    :param measure: Similarity/Dissimilarity Measures
    :return:
    """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Display two images into same window
    images_gray = np.hstack((img1_gray, img2_gray))
    display_img(images_gray, "img1_gray and img2_gray", (800, 400))

    h, w = img1_gray.shape                              # height and width
    jx, jy = floor(janela[0]/2.), floor(janela[1]/2.)   # half-janela
    disparity_map = np.zeros(img1_gray.shape, dtype=img1_gray.dtype)

    max_val_bar = 100
    bar = progressbar.ProgressBar(maxval=max_val_bar, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # iy, ix, they are the beginning of the coordinates to make the crop
    for y_ in range(jy, h-jy):
        iy = y_-jy
        for x_ in range(jx, w-jx):                      # To avoid conflicts
            a = crop_img(img1_gray, (iy, x_ - jx), janela)
            r_ssd, x_min = 9999, 0                                # infinite
            for k in range(jx, w-jx):                   # Percorrer vertically in the second image
                ix = k-jx
                b = crop_img(img2_gray, (iy, ix), janela)
                if measure == 'SSD':
                    s = ssd(a, b)
                if measure == 'SAD':
                    s = sad(a, b)
                if s < r_ssd:
                    r_ssd, x_min = s, k
            d = x_min - x_
            disparity_map[y_][x_] = d
        bar.update((y_ / w) * max_val_bar)
    bar.finish()

    display_img(disparity_map, "Disparity Map", (400, 400))


def execute_problem1():
    path = "Problem1/data/teddy/"
    img1_name = "im2.png"
    img2_name = "im6.png"

    img1 = cv2.imread(path + img1_name)
    img2 = cv2.imread(path + img2_name)

    # Display two images into same window
    images_bgr = np.hstack((img1, img2))
    display_img(images_bgr, "img1 and img2", (800, 400))

    disparity_map_gray_scale(img1, img2, (9, 9), 'SAD')

    print("Finished Problem 1...")


