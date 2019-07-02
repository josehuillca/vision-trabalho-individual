import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sys import platform as _platform
if _platform == "darwin":
   # MAC OS X
   import matplotlib
   matplotlib.use('MacOSX')
from texttable import Texttable
from typing import Tuple, Any,List


def display_img(img: np.ndarray, title: str, resize: Tuple[int, int] = (600, 600), use: str = 'cv2') -> None:
    """ Display image window
    :param img:     Input image
    :param title:   Title image
    :param resize:  Resize window (width, height)
    :param use:     display with cv2 or pyplot
    :return:        None
    """
    if use == 'cv2':
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        cv2.resizeWindow(title, resize[0], resize[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif use == 'pyplot':   # matplotlib
        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.show()
    else:
        return None


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


def draw_lines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def print_matrix(m: np.ndarray, head: np.ndarray = None, title: str = "", c_type: str = 'a') -> None:
    """ display matrix
    :param m:
    :param head: head matrix
    :param title: title matrix
    :param c_type:
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
        head = [' ' for x in range(0, cols_m)]
    content.append(head)
    for i in range(0, rows_m):
        content.append(m[i])

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_header_align(cols_align)
    table.set_cols_dtype([c_type] * cols_m)  # automatic
    table.set_cols_align(cols_align)
    table.add_rows(content)

    if title != "":
        print("**********************  " + title + "  **********************")

    print(table.draw())


def interest_points(img1: np.ndarray, img2: np.ndarray, ratio: float, num_max: int = 8, display_matches: str = 'None') -> Tuple[Any, Any]:
    """ Asumimos que sift nos dara mas de 'num=8' puntos de interes
    :param img1:            Input image  gray-scale
    :param img2:            Input image  gray-scale
    :param ratio:           ratio to search good matches 0-1
    :param num_max:             cantidad de puntos de interes a ser extraida
    :param display_matches: Draw matches
    :return:                [[x1,y1], ...[xn,yn]], [[x'1,y'1], ...[x'n,y'n]]
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    k = 0
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            if k >= num_max:
                break
            k = k + 1
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Constrain matches to fit homography
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    # We select only inlier points
    pts1 = pts1[mask == 1]
    pts2 = pts2[mask == 1]

    # Draw top matches
    if display_matches:
        im_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
        display_img(im_matches, "Matches", (600 * 2, 600), use=display_matches)
    print("Number of interest point obtained: ", len(pts1))
    return pts1, pts2


def draw_epipole_lines(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> None:
    """
    :param img1: Input array gray-scale
    :param img2: Input array gray-scale
    :param pts1: [[x1, y1], ...]
    :param pts2: [[x2, y2], ...]
    :param F:    Fundamental matrix
    :return:
    """
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F.T)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_lines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F.T)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_lines(img2, img1, lines2, pts2, pts1)

    # Display base images into same window
    numpy_h = np.hstack((img5, img3))
    display_img(numpy_h, "Epipole lines - Result", (600 * 2, 600))


def cart2hom(arr: np.ndarray) -> np.ndarray:
    """ Convert catesian to homogenous points by appending a row of 1s
    :param arr: array of shape (num_dimension x num_points)
    :returns: array of shape ((num_dimension+1) x num_points)
    """
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))


def load_images_names(file: str) -> List:
    """
    :param file: Path + name file(i.e. *_good_silhouette_images.txt)
    :return:
    """
    lines = [line.rstrip('\n') for line in open(file, 'r+')]
    return lines


def load_parameters(file: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    :param file: Path + name file(i.e. *_par.txt)
    :return: matrix P, intrinsic parameters, images name
    """
    imgs_name = []  # images names
    Ps = []         # List of matrices P
    Ks = []         # List of intrinsic parameters
    with open(file, 'r+') as fo:
        num_imgs = int(fo.readline().rstrip('\n'))
        for i in range(0, num_imgs):
            line = fo.readline().rstrip('\n').split(' ')
            imgs_name.append(line[0])
            K = np.array([float(k) for k in line[1:10]]).reshape(3, 3)      # matrix shape = 3*3
            R = np.array([float(r) for r in line[10:19]]).reshape(3, 3)     # matrix shape = 3*3
            t = np.array([float(t) for t in line[19:]])                     # matrix shape = 3*1
            Rt = np.hstack((R, np.transpose([t])))
            P = np.dot(K, Rt)
            Ps.append(P)
            Ks.append(K)
            '''print_matrix(K, title="K")
            print_matrix(R, title="R")
            print("t:", t)
            print_matrix(Rt, title="Rt")
            break'''
    # all intrinsic parameters(Ks) are equal
    return Ps, Ks, imgs_name


def load_angles(file: str) -> List[Tuple[float, float]]:
    """
    :param file: Path + file name(i.e. *SR_ang.txt)
    :return: [(-lat, long), ...]
    """
    angles = []
    for line in open(file, 'r+'):
        line = line.rstrip('\n').split(' ')
        angles.append((float(line[0]), float(line[1])))
    return angles