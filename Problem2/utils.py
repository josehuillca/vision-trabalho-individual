import cv2
import numpy as np
from typing import Tuple, List


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
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_coord(shape: Tuple[int, int], coord: Tuple[int, int]) -> List:
    """
    :param shape:
    :param coord:
    :return:
    """
    h, w = shape[0], shape[1]
    T = [[w+h, 0, w/2],
         [0, w+h, h/2],
         [0, 0, 1]]
    T = np.linalg.inv(np.array(T))
    x = np.transpose(np.array([[coord[0], coord[1], 1]]))
    x_ = np.dot(T, x)
    x_ = x_.flatten().tolist()
    return [x_[1], x_[0]]


def draw_points(img: np.ndarray, points: np.ndarray, cors: np.ndarray = None, r: int = 5) -> None:
    """
    :param img:    Input image BGR
    :param points: [[x1,y1], ...[xn,yn]]
    :param cors:   [(b,g,r), ...(b,g,r)]
    :param r:      circle-radio
    :return:
    """
    if cors is None:
        cors = []
        for i in range(0, len(points)):
            c = tuple(np.random.choice(range(256), size=3))
            cors.append((int(c[0]), int(c[1]), int(c[2])))
    k = 0
    for p in points:
        cv2.circle(img, (p[0], p[1]), r, cors[k], -1)
        k = k + 1
