import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv
from uczenie_maszynowe_08_03 import kernel_h1 as k1, kernel_h2 as k2, kernel_h3 as k3, kernel_h4 as k4, kernel_h5 as k5
import time


def normalize(x):
    if(x > 255):
          return 255
    elif(x < 0):
         return 0
    else:
         return x

def conv_interp(y, kernel_grid, kernel = 5):
    kernel_grid = kernel_grid.copy()
    if(kernel == 1):
        for i in range(kernel_grid.shape[0]):
            for j in range(kernel_grid.shape[1]):
                kernel_grid[i, j] = k1(kernel_grid[i, j])
    elif(kernel == 2):
        for i in range(kernel_grid.shape[0]):
            for j in range(kernel_grid.shape[1]):
                kernel_grid[i, j] = k2(kernel_grid[i, j])
    elif(kernel == 3):
        for i in range(kernel_grid.shape[0]):
            for j in range(kernel_grid.shape[1]):
                kernel_grid[i, j] = k3(kernel_grid[i, j])
    elif(kernel == 4):
        for i in range(kernel_grid.shape[0]):
            for j in range(kernel_grid.shape[1]):
                kernel_grid[i, j] = k4(kernel_grid[i, j])
    elif(kernel == 5):
        for i in range(kernel_grid.shape[0]):
            for j in range(kernel_grid.shape[1]):
                kernel_grid[i, j] = k5(kernel_grid[i, j])
    
    yvals = np.dot(y, kernel_grid)
    for i in range(len(yvals)):
         yvals[i] = normalize(yvals[i])

    return yvals

def two_dim_decreasing_grayscale(img, scale):
    img_org = img.copy()

    if not(isinstance(scale, int)):
         scale = int(scale)

    if(scale < 2):
         scale = 2

    img_gray = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    mask = np.zeros((scale, scale))

    for i in range(scale):
         for j in range(scale):
              mask[i, j] = 1 / (scale * scale)

    blank = np.zeros((int(img_org.shape[1] / scale), int(img_org.shape[0] / scale)), dtype=np.uint8)
    
    for i in range(blank.shape[0]):
         for j in range(blank.shape[1]):
            #zebranie sumy wartości pixeli w masce
            sum = 0
            for k in range(mask.shape[0]):
                for l in range(mask.shape[1]):
                    sum = sum + img_gray[i * scale + k, j * scale + l] * mask[k, l]

            blank[i, j] = sum
    print("Rozmiar oryginalnego obrazu wynosi", img_gray.shape, ", a rozmiar zmniejszonego obrazu wynosi", blank.shape)
    cv.imshow("Oryginalny obraz w odcieniach szarości", img_gray)
    cv.imshow("Zmniejszony obraz w odcieniach szarości", blank)
    cv.waitKey(0)

    return blank

def two_dim_increasing(img, scale, BGR = False, kernel = 5):

    img_org = img.copy()
    if(BGR is False):
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)

    if not(isinstance(scale, int)):
         scale = int(scale)

    if(scale < 2):
         scale = 2
    
    input_height = img.shape[0]
    input_width = img.shape[1]
    output_height = int(input_height * scale)
    output_width = int(input_width * scale)
    if(BGR):
        blank = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
         blank = np.zeros((output_height, output_width), dtype=np.uint8)

    increased_img = blank.copy()

    #listy pomocnicze do przechowywania danych odnośnie jednego wiersza / jednej kolumny
    if(BGR):
        y_b = []
        y_r = []
        y_g = []

    else:
        y = []

    # listy pomocnicze przechowujące wartości x wykorzystane w interpolacji 1D
    x = []
    x_intp = []

    # dodanie odpowiednich wartości do vektora wykorzystanego później w interpolacji
    for i in range(scale * input_width):
            x_intp.append(i)

    # stworzenie wektora wykorzystanego w ramach interpolacji do wskazania położenia x dla wartości y_r...
    for i in range(input_width):
        x.append(scale * i)

    # interpolacja w wierzszach - poszerzenie obrazu
    period = x[1] - x[0]
    kernel_grid = np.tile(x_intp, (len(x), 1))
    offset = np.tile(np.expand_dims(x, axis= -1), (1, len(x_intp)))
    kernel_grid = (kernel_grid - offset) / period

    if(BGR):
        for i in range(input_height):
            # pobranie wartości kolorów B, G, R poszczególnych komórek w wierszu
            for j in range(input_width):
                y_b.append(img_org[i, j][0])
                y_r .append(img_org[i, j][2])
                y_g.append(img_org[i, j][1])

            y_b_int = conv_interp(y_b, kernel_grid, kernel = kernel)
            y_g_int = conv_interp(y_g, kernel_grid, kernel = kernel)
            y_r_int = conv_interp(y_r, kernel_grid, kernel = kernel)

            # uzupełnienie płótna interpolowanymi wartościami
            for j in range(len(y_b_int)):
                blank[i, j] = ([y_b_int[j], y_g_int[j], y_r_int[j]])
            
            # wyczyszczenie list pomocniczych dla konkretgeno wiersza
            y_b.clear()
            y_r.clear()
            y_g.clear()
            del y_b_int
            del y_g_int
            del y_r_int
    else:
         for i in range(input_height):
            # pobranie wartości poszczególnych komórek w wierszu
            for j in range(input_width):
                y.append(img_org[i, j])

            y_int = conv_interp(y, kernel_grid, kernel = kernel)

            # uzupełnienie płótna interpolowanymi wartościami
            for j in range(len(y_int)):
                blank[i, j] = (y_int[j])
            
            # wyczyszczenie list pomocniczych dla konkretgeno wiersza
            y.clear()
            del y_int

    # wyczyszcenie list pomocniczych dla interpolacji wierszów
    x.clear()
    x_intp.clear()

    # dodanie odpowiednich wartości do vektora wykorzystanego później w interpolacji
    for i in range(scale * input_height):
            x_intp.append(i)

    # stworzenie wektora wykorzystanego w ramach interpolacji do wskazania położenia x dla wartości y_r...
    for i in range(input_height):
        x.append(scale * i)

    period = x[1] - x[0]
    kernel_grid = np.tile(x_intp, (len(x), 1))
    offset = np.tile(np.expand_dims(x, axis= -1), (1, len(x_intp)))
    kernel_grid = (kernel_grid - offset) / period

    if(BGR):
        for i in range(output_width):
            # pobranie wartości kolorów B, G, R poszczególnych komórek w kolumnie
            for j in range(input_height):
                y_b.append(blank[j, i][0])
                y_r .append(blank[j, i][2])
                y_g.append(blank[j, i][1])

            y_b_int = conv_interp(y_b, kernel_grid, kernel = kernel)
            y_g_int = conv_interp(y_g, kernel_grid, kernel = kernel)
            y_r_int = conv_interp(y_r, kernel_grid, kernel = kernel)

            # uzupełnienie płótna interpolowanymi wartościami
            for j in range(len(y_b_int)):
                increased_img[j, i] = ([y_b_int[j], y_g_int[j], y_r_int[j]])   

            y_b.clear()
            y_r.clear()
            y_g.clear()
            del y_b_int
            del y_g_int
            del y_r_int

    else:
        for i in range(output_width):
            # pobranie wartości kolorów B, G, R poszczególnych komórek w kolumnie
            for j in range(input_height):
                y.append(blank[j, i])

            y_int = conv_interp(y, kernel_grid, kernel = kernel)

            # uzupełnienie płótna interpolowanymi wartościami
            for j in range(len(y_int)):
                increased_img[j, i] = (y_int[j])   

            y.clear()
            del y_int

    x.clear()
    x_intp.clear()

    return blank, increased_img
    
if __name__ == "__main__":

    # # tu jest zmniejszanie
    scale = [2, 3, 4]
    img_org = cv.imread("kot.jpg")
    img_org = cv.resize(img_org.copy(), (500, 500), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(img_org.copy(), cv.COLOR_BGR2GRAY)
    cv.imwrite("img_gray.png", img_gray)
    for i in scale:
        decreased = two_dim_decreasing_grayscale(img_org, i)
        cv.imshow("decreased", decreased)
        cv.waitKey(0)
        cv.imwrite(f"{i}"+".png", decreased)


    # # zwiększanie
    # scale = 2
    # # img = cv.imread("test2.png")
    # # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.imread("kot.jpg")
    # img = cv.resize(img.copy(), (100, 100), interpolation=cv.INTER_AREA)
    # img_org = img.copy()
    # img_resized_cv = img.copy()
    # # img_resized_cv = cv.cvtColor(img_resized_cv, cv.COLOR_BGR2GRAY)
    # img_resized_cv = cv.resize(img_resized_cv, (int(img_resized_cv.shape[1] * scale), int(img_resized_cv.shape[0] * scale)), interpolation=cv.INTER_AREA)
    # cv.imwrite("cvres.png", img_resized_cv)
    # for i in range(1, 6):
    #     _, output = two_dim_increasing(img, scale, kernel = i, BGR=True)
    #     cv.imshow("obraz", output)
    #     cv.imwrite("obraz" + f"{i}" + ".png", output)
    #     cv.waitKey(0)


    
    
    
