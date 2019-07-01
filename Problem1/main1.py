import cv2
import time
import numpy as np
import progressbar

from math import sqrt
from typing import Tuple
from Problem1.utils import display_img, crop_img, WriteFilePFM, save_pfm
from skimage.measure import compare_ssim as ssim

ITER_MIN = 3    # iteracion minima de la janelaen la imagen fuente/source


def SSD(a: np.ndarray, b: np.ndarray) -> float:
    """ Sum Square Difference ('a' and 'b' have to be the same size)
    :param a: Array numpy
    :param b: Array numpy
    :return: sum square difference
    """
    return ((a-b)**2).sum()


def SAD(a: np.ndarray, b: np.ndarray) -> float:
    """ Sum Add Difference ('a' and 'b' have to be the same size)
    :param a: Array numpy
    :param b: Array numpy
    :return: sum square difference
    """
    return (abs(a-b)).sum()


def SSIM(a: np.ndarray, b: np.ndarray) -> float:
    """ Structural Similarity Measure (janela 7x7 a mas)
    :param a:
    :param b:
    :return: 'ssim'  results in 1 if it is very similar, so we subtract 1,
            because we say it is more similar when it is closer to 0.
    """
    return 1.0 - ssim(a, b)


def NCC(a: np.ndarray, b: np.ndarray) -> float:
    """ Normalized Cross Correlation ('a' and 'b' have to be the same size)
    :param a: Array numpy normalized
    :param b: Array numpy normalized
    :return: 'NCC' returns a value of -1 to 1, where one is very similar,
            we convert it to a range of 0 to 1, where 0 is very similar
    """
    ncc_ = (a*b).sum()/sqrt(((a**2).sum())*((b**2).sum()))

    if ncc_ >= 0:
        res = (1 - ncc_)/2.
    else:
        res = (ncc_ + 1)/2.
    return res


def SM_algorithm(a: np.ndarray, b: np.ndarray, alg: str) -> float:
    """ Similarity Measure Algorithm
    :param a: Array numpy
    :param b: Array numpy
    :param alg: Algorithm name(SSD, NC)
    :return:
    """
    if alg == "SSD":
        return SSD(a, b)
    if alg == "SAD":
        return SAD(a, b)
    if alg == "SSIM":
        return SSIM(a, b)
    if alg == "NCC":
        return NCC(a, b)
    return 0


def disparity_map(img1: np.ndarray, img2: np.ndarray, ndisp: int, janela: Tuple[int, int], SMA: str, norm: bool = False) -> np.ndarray:
    """ mapa de disparidade da segunda imagem em relação à primeira. (busco en la primera imagen...)
    :param img1: input image Left BGR
    :param img2: input image Right BGR
    :param ndisp: a conservative bound on the number of disparity levels
    :param janela: (h_y, w_x)
    :param SMA: Similarity Measure Algorithm (SSD, NC,..)
    :param norm: (BOOL)normalized gray images
    :return:
    """
    img_l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)       # Image Left
    img_r = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)       # Image Right

    jy, jx = janela[0], janela[1]                        # height and width of the janela
    h, w = img_r.shape                                   # height and width of the image Right
    disp_map = np.zeros(img_r.shape, dtype="float32")    # matrix that will contain the disparity map
    # 'iter' number of pixels that the window advances in the img_r; do not confuse with the search window (img_l)
    iter = ITER_MIN

    # To use 'NCC' it is necessary that the images are normalized in order to obtain a result in a range of [-1, 1],
    # and then convert it to [0,1]
    if SMA == "NCC" or norm:
        img_l = (img_l - np.mean(img_l)) / (np.std(img_l) * len(img_l))
        img_r = (img_r - np.mean(img_r)) / (np.std(img_r) * len(img_r))

    # Progress Bar animation
    maxv = 100
    bar = progressbar.ProgressBar(maxval=maxv, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    #
    for y in range(0, h-jy+1, min(iter, jy)):
        for x in range(0, w-jx+1, min(iter, jx)):
            window_r = crop_img(img_r, (y, x), janela)
            SM = 9999       # init value of Similarity Measure
            disp_val = 0    # disparity value
            # Scrolling through the search window
            #for x_ in range(x + jx, min(w - jx, x + ndisp)):
            for x_ in range(x, min(w - jx, x + ndisp)):
                window_l = crop_img(img_l, (y, x_), janela)
                result_SM = SM_algorithm(window_r, window_l, SMA)
                if result_SM < SM:
                    disp_val = abs(x_ - x)
                    SM = result_SM
                    if SM == 0:
                        break

            for yi in range(y, y + jy):
                for xi in range(x, x + jx):
                    disp_map[yi][xi] = disp_val

            bar.update((y / h) * maxv)
    bar.finish()

    return disp_map


def execute_problem1():
    path = "Problem1/data/Motorcycle/"   # Motorcycle-imperfect
    img1_name = "im0.png"
    img2_name = "im1.png"

    img1 = cv2.imread(path + img1_name)
    img2 = cv2.imread(path + img2_name)

    # Display three images into same window
    img1_2 = cv2.addWeighted(img1, 0.9, img2, 0.5, 0)
    images_bgr = np.hstack((img1, img2, img1_2))
    # display_img(images_bgr, "[ %s, %s and addWeight]" % (img1_name, img2_name), (1200, 400))

    # NECESITO Leer el archivo calib.txt para obtener el ndisp automatico

    start = time.time()
    disp_map = disparity_map(img1, img2, ndisp=69, janela=(15, 15), SMA="SSIM", norm=False)
    end = time.time()
    print("Time Taken : ~ %.0f%s%.2f %s" % ((end - start) // 60, ":", (end - start) % 60, "sec"))

    #display_img(disp_map, "Disparity Map", (600, 600))  # si la imagen tiene dtype='float32' no es posible mostrar
    WriteFilePFM(disp_map, disp_map.shape[1], disp_map.shape[0], path + "disp0.pfm")
    print("Finished Problem 1...")


