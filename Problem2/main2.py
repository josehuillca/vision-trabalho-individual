from Problem2.fundamentalMatrix import compute_f
from Problem2.reconstruction3D import reconstruction_3d
from Problem2.utils import load_parameters


def execute_problem2():
    path = "Problem2/data/dinoSparseRing/"   # "Problem2/data/templeSparseRing/"

    # Loading matrix P and name images
    parameters_file = 'dinoSR_par.txt'
    Ps, Ks, imgs_name = load_parameters(path + parameters_file)

    i, j = 0, 1
    _ = compute_f(path + imgs_name[i], path + imgs_name[j], alg='RANSAC')
    #_ = reconstruction_3d(path, Ps, Ks, imgs_name)
    # ------------------------------
    print("Finish Problem 2! ...")

