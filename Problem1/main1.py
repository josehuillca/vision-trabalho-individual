import cv2
from Problem1.utils import display_img, crop_img
import numpy as np
from typing import Tuple
import progressbar
import matplotlib.pyplot as plt

GOOD_MATCH_PERCENT = 0.25   # porcentaje de los buenos matches a tomar en cuenta


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


def get_distance(img1: np.ndarray, img2: np.ndarray, display: bool=False) -> int:
    """ Usando SIFT, calcularemos la distance de sus keypoints
    :param img1: Input image gray-scale
    :param img2: Input image gray-scale
    :param display: display matches
    :return: distancia en pixeles
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    if display:
        matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        display_img(matching_result, "Matches", (400 * 2, 400))
    print(len(matches))
    return int(matches[len(matches)-1].distance)


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
    iterator = min(jx, 5)
    disparity_map = np.zeros(img1_gray.shape, dtype=img1_gray.dtype)

    max_val_bar = 100
    bar = progressbar.ProgressBar(maxval=max_val_bar, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # iy, ix, they are the beginning of the coordinates to make the crop
    for y_ in range(jy, h-jy, iterator):
        iy = y_-jy
        for x_ in range(jx, w-jx, iterator):                      # To avoid conflicts
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

    # ---
    d_j = get_distance(img1_gray, img2_gray, True)

    h, w = img2_gray.shape  # height and width
    jx, jy = janela[0] // 2, janela[1] // 2  # half-janela
    iterator = min(jx, 5)   # tamanho para percorrer as janelas

    img1_gray_mod = np.zeros((img1_gray.shape[0] + 2 * jy, img1_gray.shape[1] + 2 * jx), dtype=img1_gray.dtype)
    img1_gray_mod[jy:jy + h, jx:jx + w] = img1_gray

    img2_gray_mod = np.zeros((img2_gray.shape[0] + 2*jy, img2_gray.shape[1] + 2*jx), dtype=img2_gray.dtype)
    img2_gray_mod[jy:jy+h, jx:jx+w] = img2_gray

    disparityMap = np.empty(img1_gray_mod.shape, dtype=img1_gray.dtype)
    h_, w_ = img1_gray_mod.shape
    s_list = []
    # ProgressBar
    max_val_bar = 100
    bar = progressbar.ProgressBar(maxval=max_val_bar,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # iy, ix, they are the beginning of the coordinates to make the crop
    for y_ in range(jy, h_-jy, iterator):
        iy = y_ - jy
        for x_ in range(jx, w_-jx, iterator):  # To avoid conflicts
            ix_0 = x_ - jx
            a = crop_img(img1_gray_mod, (iy, ix_0), janela)
            r_ssd, x_min = 9999, 0  # infinite
            for k in range(max(jx, x_ - d_j * 2), min(w_-jx, x_ + d_j * 2), iterator):  # Percorrer vertically in the second image
                ix = k - jx
                b = crop_img(img2_gray_mod, (iy, ix), janela)

                s = ssd(a, b)
                if s < r_ssd:
                    r_ssd, x_min = s, k
                if s == 0:
                    break

            d = abs(x_min - x_)
            for y_p in range(iy, min(iy+janela[1], h_)):
                for x_p in range(ix_0, min(ix_0+janela[0], w_)):
                    disparityMap[y_][x_] = d

        bar.update((y_ / w) * max_val_bar)
    bar.finish()

    #plt.plot(s_list)
    #plt.savefig("Problem1/result/temp/temp.jpg")
    display_img(disparityMap, "Disparity Map", (400, 400))


def disparity_map_v2(img1: np.ndarray, img2: np.ndarray, ndisp: int, janela: Tuple[int, int] = (3,3)) -> None:
    """ mapa de disparidade da segunda imagem em relação à primeira. (busco en la primera imagen...)
    :param img1: input image Left BGR
    :param img2: input image Right BGR
    :param ndisp:
    :param janela: (h_y, w_x)
    :return:
    """
    img_l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)      # Image Left
    img_r = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)      # Image Right

    jy, jx = janela[0], janela[1]
    h, w = img_r.shape
    disp_map = np.zeros(img_r.shape, dtype=img_r.dtype)
    iter = 5

    maxv = 100
    bar = progressbar.ProgressBar(maxval=maxv, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for y in range(0, h-jy+1, min(iter, jy)):
        for x in range(0, w-jx+1, min(iter, jx)):
            window_r = crop_img(img_r, (y, x), janela)
            SM = 9999  # init value of Similarity Measure
            disp_val = 0
            for x_ in range(x + jx, min(w - jx, x + ndisp)):
                window_l = crop_img(img_l, (y, x_), janela)
                result_SM = ssd(window_r, window_l)
                if result_SM < SM:
                    disp_val = abs(x_ - x)
                    SM = result_SM
                    if SM == 0:
                        break

            for yi in range(y, y + jy):
                for xi in range(x, x + jx):
                    disp_map[yi][xi] = disp_val

            bar.update((y / w) * maxv)
    bar.finish()

    display_img(disp_map, "Disparity Map", (600, 600))
    return None


def execute_problem1():
    path = "Problem1/data/Motorcycle-imperfect/"   # Motorcycle-imperfect
    img1_name = "im0.png"
    img2_name = "im1.png"

    img1 = cv2.imread(path + img1_name)
    img2 = cv2.imread(path + img2_name)

    # Display three images into same window
    img1_2 = cv2.addWeighted(img1, 0.9, img2, 0.5, 0)
    images_bgr = np.hstack((img1, img2, img1_2))
    display_img(images_bgr, "[ %s, %s and addWeight]" % (img1_name, img2_name), (1200, 400))

    disparity_map_v2(img1, img2, ndisp=280, janela=(15, 15))
    # disparity_map_gray_scale(img1,img2, (3, 3))
    print("Finished Problem 1...")


