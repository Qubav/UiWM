import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv
from uczenie_maszynowe_08_03 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5
import time


if __name__ == "__main__":
    img_org = cv.imread("kot.jpg")
    img = img_org.copy()
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank = np.zeros((input_height, input_width, 3), np.float64)
    print(blank.shape)
    pois_number = input_width * input_height
    poisson = np.random.poisson(1024.0, pois_number)

    plt.figure()
    plt.hist(poisson)
    plt.show()

    for i in range(input_height):
        for j in range(input_width):
            blank[i, j, 0] = img[i, j, 0] + poisson[i * input_height + j]
            blank[i, j, 1] = img[i, j, 1] + poisson[i * input_height + j]
            blank[i, j, 2] = img[i, j, 2] + poisson[i * input_height + j]

    output = cv.normalize(blank, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    cv.imshow("kot szum", output)
    cv.imshow("kot", img)
    cv.waitKey(0)
