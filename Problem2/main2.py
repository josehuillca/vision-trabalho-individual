from Problem2.fundamentalMatrix import compute_f
from Problem2.reconstruction3D import reconstruction_3d
from Problem2.utils import load_parameters, print_matrix
from Problem2.opengl_examples.cube2 import main_cube2


def execute_problem2():
    path = "Problem2/data/dinoSparseRing/"   # "Problem2/data/templeSparseRing/"

    # Loading matrix P and name images
    parameters_file = 'dinoSR_par.txt'
    Ps, Ks, imgs_name = load_parameters(path + parameters_file)
    # print_matrix(Ps[0], title="Ps0")

    i, j = 0, 1
    # _ = compute_f(path + imgs_name[i], path + imgs_name[j], alg='RANSAC')
    pts3D = reconstruction_3d(path, Ps, Ks, imgs_name)
    # main_cube2(pts3D.T)
    # ------------------------------
    print("Finish Problem 2! ...")

