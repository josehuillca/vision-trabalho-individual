import cv2
import numpy as np
from texttable import Texttable
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


def print_matrix(m: np.ndarray, head: np.ndarray = None, title: str = "") -> None:
    """ display matrix
    :param m:
    :param head: head matrix
    :param title: title matrix
    :return:
    """
    cols_align = []
    cols_m = m.shape[1]
    rows_m = m.shape[0]
    for i in range(0, cols_m):
        if i == 0:
            cols_align.append("l")
        else:
            cols_align.append("r")

    content = []
    if head is None:
        head = ['-' for x in range(0, cols_m)]
    content.append(head)
    for i in range(0, rows_m):
        content.append(m[i])

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_header_align(cols_align)
    table.set_cols_dtype(['a'] * cols_m)  # automatic
    table.set_cols_align(cols_align)
    table.add_rows(content)

    if title != "":
        print("**********************  " + title + "  **********************")

    print(table.draw())

