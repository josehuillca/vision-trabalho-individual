from typing import Tuple
import cv2
import numpy as np


def display_img(img: np.ndarray, title: str, resize: np.ndarray = (600, 600)) -> None:
    """ Display image window
    :param img: Input image
    :param title: Title image
    :param resize: Resize window (width, height)
    :return: None
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, resize[0], resize[1])
    cv2.waitKey()
    cv2.destroyWindow(title)


def crop_img(img: np.ndarray, yx_i: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    """ Crop image
    :param img: Input image Gray-scale
    :param yx_i: top-left(according to OpenCV) coordinate
    :param size: (height, width) sizes
    :return: crop image
    """
    '''h, w = img.shape
    # Constraint 1
    yh = yx_i[0] + size[0]
    yh_ = yh if yh < h else h
    # Constraint 2
    xw = yx_i[1] + size[1]
    xw_ = xw if xw < w else w
    # Constraint 3
    yi = yx_i[0] if yx_i[0] > 0 else 0
    xi = yx_i[1] if yx_i[1] > 0 else 0
    # print(yi, xi, yh_, xw_, yx_i, size)
    return img[yi: yh_, xi: xw_]'''
    return img[yx_i[0]:yx_i[0]+size[0], yx_i[1]:yx_i[1] + size[1]]

