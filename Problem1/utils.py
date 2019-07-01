from typing import Tuple
import cv2
import sys
import numpy as np
import cmath


def display_img(img: np.ndarray, title: str, resize: Tuple[int, int] = (600, 600)) -> None:
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


def littleendian() -> bool:
    """
    check whether machine is little endian
    :return:
    """
    return sys.byteorder == 'little'


def WriteFilePFM(data: np.ndarray, width: int, height: int, filename:str, scalefactor: float=1/255.0) -> None:
    """write pfm image (added by DS 10/24/2013)
    1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
    :param data:
    :param width:
    :param height:
    :param filename:
    :param scalefactor:
    :return:
    """
    # Open the file
    stream = open(filename, "wb")
    if stream is None:
        print("WriteFilePFM: could not open ", filename)
        return None

    if data.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    # sign of scalefact indicates endianness, see pfms specs
    if littleendian():
        scalefactor = -scalefactor

    # write the header: 3 lines: Pf, dimensions, scale factor(negative val == little endian)
    mystring = "Pf\n{} {}\n{}\n".format(width, height, scalefactor)
    stream.write(mystring.encode())

    n = width
    # write rows - - pfm stores rows in inverse order!
    for y in range(height-1, -1, -1):
        ptr =data[y]
        # change invalid pixel (which seem to be represented as -10) to INF
        for x in range(0, width):
            if ptr[x] < 0:
                print("ENTRO?")
                ptr[x] = cmath.inf
        stream.write(ptr)
    # close file
    stream.close()


# otra forma de salvar pfm image
def save_pfm(filename, image, scale = 1/255.0):
    file = open(filename, "wb")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    x = '%d %d\n' % (image.shape[1], image.shape[0])
    file.write(x.encode())

    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    z = '%f\n' % scale
    file.write(z.encode())

    image.tofile(file)