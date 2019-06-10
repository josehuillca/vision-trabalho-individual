import cv2
from Problem1.utils import display_img, crop_img
import numpy as np
from typing import Tuple
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
    jx, jy = janela[0]//2, janela[1]//2   # half-janela
    disparity_map = np.zeros(img1_gray.shape, dtype=img1_gray.dtype)

    max_val_bar = 100
    bar = progressbar.ProgressBar(maxval=max_val_bar, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # iy, ix, they are the beginning of the coordinates to make the crop
    for y_ in range(jy, h-jy):
        iy = y_-jy
        for x_ in range(jx, w-jx):                      # To avoid conflicts
            ix_0 = x_-jy
            a = crop_img(img1_gray, (iy, ix_0), janela)
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
            d = abs(x_min - x_)
            disparity_map[y_][x_] = d
        bar.update((y_ / w) * max_val_bar)
    bar.finish()

    display_img(disparity_map, "Disparity Map", (400, 400))


def disparity_map(img1: np.ndarray, img2: np.ndarray, janela: Tuple[int, int] = (3,3)) -> None:
    """ Given a pair of rectified stereo images, use the matching window to calculate the
        disparity map of the second image relative to the first.
    :param img1: Input image BGR (source)
    :param img2: Input image BGR (reference)
    :param janela: Janela de busca (x,y), es mejor que sea cuadratica(x=y)
    :return:
    """
    # convert BGR to gray-scale (RGB[A] to Gray:  Y <- 0.299*R + 0.587*G + 0.114*B)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = img1_gray.shape  # height and width
    jx, jy = janela[0] // 2, janela[1] // 2  # half-janela
    disparityMap = np.empty(img1_gray.shape, dtype=img1_gray.dtype)
    iterator = min(jx, 5)   # tamanho para percorrer as janelas

    # ProgressBar
    max_val_bar = 100
    bar = progressbar.ProgressBar(maxval=max_val_bar,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # iy, ix, they are the beginning of the coordinates to make the crop
    for y_ in range(0, h, iterator):
        iy = y_ - jy
        for x_ in range(0, w, iterator):  # To avoid conflicts
            ix_0 = x_ - jy
            a = crop_img(img1_gray, (iy, ix_0), janela)
            r_ssd, x_min = 9999, 0  # infinite
            for k in range(0, w, iterator):  # Percorrer vertically in the second image
                ix = k - jx
                b = crop_img(img2_gray, (iy, ix), janela)

            #    s = sad(a, b)
            #    if s < r_ssd:
            #        r_ssd, x_min = s, k
            #        if s == 0:
            #            break       # the closest match is found
            #d = abs(x_min - x_)
                i,j =0,0
                for y_p in range(y_, min(y_+janela[1], h)):
                    i = 0
                    for x_p in range(k, min(k+janela[0], w)):
                        if j<b.shape[0] and i<b.shape[1]:
                            disparityMap[y_p][x_p] = b[j][i]
                        i = i+1
                    j = j+1
            break
        bar.update((y_ / w) * max_val_bar)
    bar.finish()

    display_img(disparityMap, "Disparity Map", (400, 400))


def execute_problem1():
    path = "Problem1/data/teddy/"
    img1_name = "im2.png"
    img2_name = "im6.png"

    img1 = cv2.imread(path + img1_name)
    img2 = cv2.imread(path + img2_name)

    # Display three images into same window
    img1_2 = cv2.addWeighted(img1, 0.6, img2, 0.5, 0)
    images_bgr = np.hstack((img1, img2, img1_2))
    display_img(images_bgr, "[ %s, %s and addWeight]" % (img1_name, img2_name), (1200, 400))

    disparity_map(img1, img2, (15, 15))

    print("Finished Problem 1...")


