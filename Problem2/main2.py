from Problem2.reconstruction3D import reconstruction_3d


def execute_problem2():
    path = "Problem2/data/dinoSparseRing/"   # "Problem2/data/templeSparseRing/"
    img1_name = "dinoSR0002.png"
    img2_name = "dinoSR0003.png"

    # img1_gray = cv2.imread(path + img1_name, cv2.IMREAD_GRAYSCALE)
    # img2_gray = cv2.imread(path + img2_name, cv2.IMREAD_GRAYSCALE)
    # compute_f(img1_gray, img2_gray, alg='RANSAC')

    parameters_file = 'dinoSR_par.txt'
    _ = reconstruction_3d(path, parameters_file)
    # ------------------------------
    print("Finish Problem 2! ...")

