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
    return img[yx_i[0]: yx_i[0] + size[0], yx_i[1]: yx_i[1] + size[1]]