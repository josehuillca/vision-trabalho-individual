# ############################################################################################## #
# Algorithm to                                                                         #
# ############################################################################################## #

import numpy as np
import cv2
from typing import Tuple, Any
from Problem2.utils import cart2hom
import scipy # use numpy if scipy unavailable
import scipy.linalg


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


def compute_f_8point(pts1_o: np.ndarray, pts2_o: np.ndarray) -> np.ndarray:
    """
    :param pts1_o: Interest points [[x1, y1], ... [xn, yn]]
    :param pts2_o: Interest points [[x'1, y'1], ... [x'n, y'n]]
    :return:       Fundamental Matrix
    """
    #print(pts2_o)
    # get 8 interest points, Format = [[x1, y1, w1], [x2, y2, w2], ...[xn, yn, wn]]
    # The normal thing is that the points are (x/w, y/w, w/w), where w = 1
    pts1 = np.array([[x, y, 1] for x, y, z in pts1_o])
    pts2 = np.array([[x, y, 1] for x, y, z in pts2_o])
    # Format required to improve the matrix process = [[x1, x2, ... xn], [y1, y2, ... yn], [w1, w2, ... wn]]
    pts1, pts2 = pts1.T, pts2.T

    # Normalization: Transform the image coordinates
    pts1_, T1 = normalize_coord(pts1)
    pts2_, T2 = normalize_coord(pts2)
    print("Interest Points: ", pts1_.shape[1])

    # Find the fundamental matrix
    A = [[], [], [], [], [], [], [], []]
    n = pts1_.shape[1]
    for i in range(0, n):
        x, y, w = pts1_[0][i], pts1_[1][i], pts1_[2][i]
        x_, y_, w_ = pts2_[0][i], pts2_[1][i], pts2_[2][i]
        A[i] = [x_*x, x_*y, x_*w, y_*x, y_*y, y_*w, w_*x, w_*y, w_*w]
    #print("A:", A)
    # print_matrix(np.array(A), title="Matrix A")
    u, s, vh = np.linalg.svd(A)
    # print_matrix(np.array(vh), title="Matrix vh")

    # Let F be the last column of vh
    # Reshape into  a 3x3 matrix F_shapeu
    F_s = vh[-1].reshape(3, 3)

    # Normalize
    #F_s = F_s/np.linalg.norm(F_s, ord=np.inf)
    # Constraint enforcement SVD descomposition
    u_, s_, vh_ = np.linalg.svd(F_s)
    s_n = np.zeros((3, 3))
    s_n[0][0], s_n[1][1] = s_[0], s_[1]
    F_ = np.dot(u_, np.dot(s_n, vh_))

    # F_ = F_/F_[2, 2]

    # Denormalize
    F = np.dot(np.transpose(T2), np.dot(F_, T1))
    F = F/F[2, 2]

    return F

# ----------------------------------------------------------------------

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
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
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(alsoinliers)))
        if len(alsoinliers) > d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
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
    print("n y n_data:", idxs1, idxs2)
    return idxs1, idxs2

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
        F = compute_f_8point(x1.T, x2.T)
        return F

    def get_error(self, data, F):
        """ Compute x^T F x for all correspondences,
            return error for each transformed point. """

        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # Sampson distance as error measure
        Fx1 = np.dot(F, x1)
        Fx2 = np.dot(F, x2)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (np.diag(np.dot(x1.T, np.dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


def F_from_ransac(x1, x2):
    """ Robust estimation of a fundamental matrix F from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).
        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    maxiter = 500
    match_theshold = 1e-6

    x1 = cart2hom(x1)
    x2 = cart2hom(x2)

    data = np.vstack((x1, x2))
    model = RansacModel()
    print("Dataaaaa:",data.shape)
    # compute F and return with inlier index
    F, ransac_data = ransac(data.T, model, 8, maxiter, match_theshold, 20, return_all=True)

    return F, ransac_data['inliers']