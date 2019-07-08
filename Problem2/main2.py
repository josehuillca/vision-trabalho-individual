from Problem2.fundamentalMatrix import compute_f
from Problem2.reconstruction3D import reconstruction_3d
from Problem2.utils import load_parameters, print_matrix, display_img
from Problem2.opengl_examples.cube2 import main_cube2
import cv2, numpy as np


def execute_problem2():
    dataset = 'dino'
    path = "Problem2/data/" + dataset + "Ring/"   # "Problem2/data/templeSparseRing/"

    # Loading matrix P and name images
    parameters_file = dataset + 'R_par.txt'
    Ps, Ks, imgs_name = load_parameters(path + parameters_file)
    # print_matrix(Ps[0], title="Ps0")

    i,j = 0, 1  # indices de las images a usar
    # Calculando a matriz fundamental
    #_ = compute_f(path + imgs_name[i], path + imgs_name[j], alg='RANSAC')
    pts3D = reconstruction_3d(path, Ps, Ks, imgs_name)
    print("Exemplo dos puntos ·D:", pts3D.T[:3])
    main_cube2(pts3D.T)
    # ------------------------------
    print("Finish Problem 2! ...")

