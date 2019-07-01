# ############################################################################################## #
# Algorithm to automatically estimate the fundamental matrix between two images using RANSAC.    #
# (i) Interest points                                                                            #
# (ii) Putative correspondences                                                                  #
# (iii) RANSAC robust estimation                                                                 #
# (iv) Non-linear estimation                                                                     #
# (v) Guided matching                                                                            #
# ############################################################################################## #

import numpy as np
import cv2
from typing import Tuple, Any
from Problem2.utils import cart2hom, print_matrix
from Problem2.norm_8_point_f import compute_f_8point


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)

    """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points
        alsoinliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min(), d, len(alsoinliers))
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            print("ENTRO!...")
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                print("ENTRO2...")
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
        ''''if iterations == 6998:
            return bettermodel, []'''
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    # print("n y n_data:", idxs1, idxs2)
    return idxs1, idxs2


def normalize_coord(coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param coord: [[x1, x2, ... xn], [y1, y2, ... yn], [w1, w2, ... wn]]
    :return:      normalized coordinates
    """
    x = coord[0]
    y = coord[1]
    center = coord.mean(axis=1)  # mean of each row
    cx = x - center[0]  # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    T = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1]
    ])
    return np.dot(T, coord), T


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ Estimate fundamental matrix using eight
            selected correspondences. """

        # transpose and split data into the two point sets
        data = data.T
        #print("data: ", data)
        x1 = data[:3, :8]
        x2 = data[3:, :8]
        #print("X1:", x1)
        # estimate fundamental matrix and return
        F = compute_f_8point(x1, x2)
        return F

    def get_error(self, data, F):
        """ Compute x^T F x for all correspondences,
            return error for each transformed point. """

        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # x1, x2 = x1.T, x2.T

        # Normalization: Transform the image coordinates
        x1, _ = normalize_coord(x1)
        x2, _ = normalize_coord(x2)

        # Sampson distance as error measure
        # print(F, x1)
        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


def compute_f_ransac(x1, x2):
    """ Robust estimation of a fundamental matrix F from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).
        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    maxiter = 5000
    match_theshold = 1e-3

    x1 = cart2hom(x1)
    x2 = cart2hom(x2)

    data = np.vstack((x1, x2))
    model = RansacModel()
    print("Dataaaaa:",data.shape)
    # compute F and return with inlier index
    F, ransac_data = ransac(data.T, model, 8, maxiter, match_theshold, 10, debug=False, return_all=True)
    print_matrix(F)
    return F,  []#ransac_data['inliers']
