import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv
from Lab2 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5



if __name__ == "__main__":

    # img = cv.imread("kot.jpg")
    img = cv.imread("szlachcic1.png")
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank_g = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    blank_r = blank_g.copy()
    blank_b = blank_g.copy()
    blank = blank_g.copy()
    for i in range(int(input_height / 2)):
        for j in range(int(input_width / 2)):
            blank_g[2 * i, 2 * j] = [0, img[2 * i, 2 * j][1], 0]
            blank_g[2 * i + 1, 2 * j + 1] = [0, img[2 * i, 2 * j][1], 0]
            blank_b[2 * i + 1, 2 * j] = [img[2 * i + 1, 2 * j][0], 0, 0]
            blank_r[2 * i, 2 * j + 1] = [0, 0, img[2 * i, 2 * j + 1][2]]

    blank = blank_g + blank_b + blank_r
    cv.imshow("green", blank_g)
    cv.imshow("red", blank_r)
    cv.imshow("blue", blank_b)
    cv.imshow("calosc", blank)
    cv.waitKey(0)

    # mask_g = np.array(4, 4, 1)
    mask_g = np.array([[1, 0], [0, 1]],dtype=np.uint8)
    # print(mask_g)
    scale_h = input_height // 2
    scale_w = input_width // 2
    tile = np.tile(mask_g, (scale_h, scale_w))
    # tile.astype(np.uint8)
    # print(tile)
    green = img[:,:,1]

    green = green * tile
    cv.imshow("gren", green)
    cv.waitKey(0) 



