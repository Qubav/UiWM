import numpy as np
import cv2 as cv
from Lab2 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5

def cmos(img, Bayer = False, Fuji = True, save = False, filename = "obraz"):
    #pobranie wymiarów obrazu wejściowego
    input_height = img.shape[0]
    input_width = img.shape[1]

    #stworzenie podstawowej maski
    if(Bayer is True):
        mask_g = np.array([[1, 0], [0, 1]],dtype=np.uint8)
        mask_b = np.array([[0, 0], [1, 0]],dtype=np.uint8)
        mask_r = np.array([[0, 1], [0, 0]],dtype=np.uint8)
        scale_h = input_height // 2
        scale_w = input_width // 2
        matrix_type = "Bayer"

    elif(Fuji is True):
        mask_g = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]],dtype=np.uint8)
        mask_b = np.array([[0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],dtype=np.uint8)
        mask_r = np.array([[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]],dtype=np.uint8)
        scale_h = input_height // 6
        scale_w = input_width // 6
        matrix_type = "Fuji"

    #powielenie podstawowej marki w celu uzyskania rozłożenia jej na całosci obrazu
    masked_g = np.tile(mask_g, (scale_h, scale_w))
    masked_b = np.tile(mask_b, (scale_h, scale_w))
    masked_r = np.tile(mask_r, (scale_h, scale_w))

    #wyodrębnienie przestrzeni kolorystrycznych
    green_component = img[:, :, 1]
    blue_component = img[:, :, 0]
    red_component = img[:, :, 2]

    #dodanie brakujących wierszy/kolumn wypełnionych 1, jeśli wymiar nie jest podzielny prze 2 bądź 6
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

    # mnożenie odpowiadających sobie elementów arrayów
    green = green_component * masked_g
    blue = blue_component * masked_b
    red = red_component * masked_r

    assembled = cv.merge((blue, green, red))

    #zapisanie obrazów jeśli wartość ustawiona została na True
    if(save is True):
        cv.imwrite(filename + "_" + matrix_type + "_blue.png", blue)
        cv.imwrite(filename + "_" + matrix_type + "_green.png", green)
        cv.imwrite(filename + "_" + matrix_type + "_red.png", red)
        cv.imwrite(filename + "_" + matrix_type + "_assembled.png", assembled)

    # zwrócenie obrazu złożonego z wcześniej przygotowanych macierzy z wartościami pikseli dla poszczególnych kolorów
    return assembled
   

if __name__ == "__main__":

    img = cv.imread("kot.jpg")
    # img = cv.imread("szlachcic1.png")
    img = cv.resize(img.copy(), (228, 228), interpolation=cv.INTER_AREA)
    assembled = cmos(img, Fuji = True, save = True, filename = "kot228x228")
    cv.imshow("assembled", assembled)
    cv.waitKey(0) 



