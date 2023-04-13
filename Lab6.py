import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv
from Lab3 import two_dim_increasing
import time
from scipy.ndimage import gaussian_filter
from Lab5 import poissoning


def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.
      the input   x   is noisy Poisson-distributed data
      the output  fx  has variance approximately equal to 1.
    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    '''
    return 2.0*np.sqrt(x + 3.0/8.0)

def inverse_anscombe(z):
    '''
    Compute the inverse transform using an approximation of the exact
    unbiased inverse.
    Reference: Makitalo, M., & Foi, A. (2011). A closed-form
    approximation of the exact unbiased inverse of the Anscombe
    variance-stabilizing transformation. Image Processing.
    '''
    #return (z/2.0)**2 - 3.0/8.0
    return (1.0/4.0 * np.power(z, 2) +
            1.0/4.0 * np.sqrt(3.0/2.0) * np.power(z, -1.0) -
            11.0/8.0 * np.power(z, -2.0) + 
            5.0/8.0 * np.sqrt(3.0/2.0) * np.power(z, -3.0) - 1.0 / 8.0)


if __name__ == "__main__":
    img_org = cv.imread("kwiaty.jpg")
    lamb = [1, 4, 16, 64, 256, 1024]
    scale = 10

    noised = poissoning(img_org, 1)    
    # img_A = 2 * np.sqrt(noised.copy() + 3/8)
    # img_A2 = np.power(img_A.copy() / 2, 2) - 3 / 8
    img_A = anscombe(noised)
    print(np.max(img_A))
    img_A3 = gaussian_filter(img_A, sigma=2)
    img_A2 = inverse_anscombe(img_A)
    cv.imshow("funkcja gotowa", noised)
    cv.imshow("A", img_A)
    cv.imshow("A2", img_A2)
    cv.imshow("A3", img_A3)
    cv.waitKey(0)
