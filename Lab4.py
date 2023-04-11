import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv
from Lab2 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5



if __name__ == "__main__":

    img = cv.imread("kot.jpg")
    # img = cv.imread("szlachcic1.png")
    img = cv.resize(img.copy(), (228, 228), interpolation=cv.INTER_AREA)
    input_height = img.shape[0]
    input_width = img.shape[1]

    Bayer = False
    Fuji = True
    if(Bayer is True):
        #stworzenie podstawowej maski
        mask_g = np.array([[1, 0], [0, 1]],dtype=np.uint8)
        mask_b = np.array([[0, 0], [1, 0]],dtype=np.uint8)
        mask_r = np.array([[0, 1], [0, 0]],dtype=np.uint8)
        scale_h = input_height // 2
        scale_w = input_width // 2

    elif(Fuji is True):
        mask_g = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]],dtype=np.uint8)
        mask_b = np.array([[0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],dtype=np.uint8)
        mask_r = np.array([[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]],dtype=np.uint8)
        scale_h = input_height // 6
        scale_w = input_width // 6
    
    #powielenie podstawowej marki w celu uzyskania rozłożenia jej na całosci obrazu
    
    masked_g = np.tile(mask_g, (scale_h, scale_w))
    masked_b = np.tile(mask_b, (scale_h, scale_w))
    masked_r = np.tile(mask_r, (scale_h, scale_w))

    #wyodrębnienie przestrzeni kolorystrycznych
    green_component = img[:, :, 1]
    blue_component = img[:, :, 0]
    red_component = img[:, :, 2]

    if(masked_b.shape[0] != input_height):
        diff =  input_height - masked_b.shape[0]
        row = np.ones((diff, masked_b.shape[1]), dtype = np.uint8)
        masked_g = np.append(masked_g, row, axis=0)
        masked_r = np.append(masked_r, row, axis=0)
        masked_b = np.append(masked_b, row, axis=0)

    if(masked_b.shape[1] != input_width):
        diff = input_width - masked_b.shape[1]
        row = np.ones((masked_b.shape[0], diff), dtype = np.uint8)
        masked_g = np.append(masked_g, row, axis=1)
        masked_r = np.append(masked_r, row, axis=1)
        masked_b = np.append(masked_b, row, axis=1)

    green = green_component * masked_g
    blue = blue_component * masked_b
    red = red_component * masked_r

    assembled = cv.merge((blue, green, red))

    cv.imshow("gren", green)
    cv.imshow("blue", blue)
    cv.imshow("red", red)
    cv.imshow("assembled", assembled)
    cv.waitKey(0) 



