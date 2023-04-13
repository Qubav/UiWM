import numpy as np
import cv2 as cv
from Lab2 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5
from Lab3 import conv_interp
from scipy.interpolate import interp1d

def cmos(img, Bayer = False, Fuji = True, save = False, filename = "obraz"):
    #pobranie wymiarów obrazu wejściowego
    input_height = img.shape[0]
    input_width = img.shape[1]

    #stworzenie podstawowej maski, wyznaczenie wartości skali oraz przypisanie typu macierzy na podstawie wybranego typu matrycy cmos
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

    img = assembled.copy()
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank = np.zeros((input_height, input_width, 3), dtype=np.uint8)

    y_b = []
    y_r = []
    y_g = []
    x_b = []
    x_g = []
    x_r = []
    x_intp = []

    for i in range(input_width):
        x_intp.append(i)
    
    for i in range(input_height):
        for j in range(input_width):
            if(j == 0 or j == input_width - 1):
                y_b.append(img[i, j, 0])
                x_b.append(j)
                y_g.append(img[i, j, 1])
                x_g.append(j)
                y_r.append(img[i, j, 2])
                x_r.append(j)

            else:
                if(img[i, j, 0] != 0):
                    y_b.append(img[i, j, 0])
                    x_b.append(j)

                if(img[i, j, 1] != 0):
                    y_g.append(img[i, j, 1])
                    x_g.append(j)
                
                if(img[i, j, 2] != 0):
                    y_r.append(img[i, j, 2])
                    x_r.append(j)
    
        f_b = interp1d(x_b, y_b, "cubic")
        y_b_intp = f_b(x_intp)
        f_g = interp1d(x_g, y_g, "cubic")
        y_g_intp = f_g(x_intp)
        f_r = interp1d(x_r, y_r, "cubic")
        y_r_intp = f_r(x_intp)

        for j in range(input_width):
            blank[i, j] = [y_b_intp[j], y_g_intp[j], y_r_intp[j]]
        
        y_b.clear()
        y_r.clear()
        y_g.clear()
        x_b.clear()
        x_g.clear()
        x_r.clear()
        del y_b_intp
        del y_g_intp
        del y_r_intp

    cv.imshow("blank", blank)
    cv.waitKey(0)
