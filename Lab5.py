import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate as interp
import cv2 as cv

def two_dim_decreasing(img, scale, show = False):
    img_org = img.copy()
    img = img.copy()
    

    if not(isinstance(scale, int)):
         scale = int(scale)

    if(scale < 2):
         scale = 2

    mask = np.zeros((scale, scale))

    for i in range(scale):
         for j in range(scale):
              mask[i, j] = 1 / (scale * scale)

    blank = np.zeros((img_org.shape[1] // scale, img_org.shape[0] // scale, 3), dtype=np.uint8)
    
    for i in range(blank.shape[0]):
         for j in range(blank.shape[1]):
            #zebranie sumy wartości pixeli w masce
            for c in range(3):
                sum = 0
                for k in range(mask.shape[0]):
                    for l in range(mask.shape[1]):
                        sum = sum + img[i * scale + k, j * scale + l, c] * mask[k, l]

                blank[i, j, c] = sum

    if(show is True):
        print("Rozmiar oryginalnego obrazu wynosi", img_org.shape, ", a rozmiar zmniejszonego obrazu wynosi", blank.shape)
        cv.imshow("Oryginalny obraz", img_org)
        cv.imshow("Zmniejszony obraz", blank)
        cv.waitKey(0)

    return blank

def two_dim_increasing(img, scale, show = False):
    img_org = img.copy()

    if not(isinstance(scale, int)):
         scale = int(scale)

    if(scale < 2):
         scale = 2
    
    input_height = img.shape[0]
    input_width = img.shape[1]
    output_height = int(input_height * scale)
    output_width = int(input_width * scale)
    blank = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    increased_img = blank.copy()

    #listy pomocnicze do przechowywania danych odnośnie jednego wiersza / jednej kolumny
    y_b = []
    y_r = []
    y_g = []

    # listy pomocnicze przechowujące wartości x wykorzystane w interpolacji 1D
    x = []
    x_intp = []

    # dodanie odpowiednich wartości do vektora wykorzystanego później w interpolacji
    for i in range(scale * (input_width + 1)):
            x_intp.append(i)

    # stworzenie wektora wykorzystanego w ramach interpolacji do wskazania położenia x dla wartości y_r...
    for i in range(input_width + 1):
        x.append(scale * i)

    for i in range(input_height):
        # pobranie wartości kolorów B, G, R poszczególnych komórek w wierszu
        for j in range(input_width):
            y_b.append(img_org[i, j][0])
            y_r.append(img_org[i, j][2])
            y_g.append(img_org[i, j][1])

        t_b, c_b, k_b = interp.splrep(x, y_b, s = 0, k = 4)
        bspl_b = interp.BSpline(t_b, c_b, k_b, extrapolate= False)
        y_b_int = bspl_b(x_intp)

        t_g, c_g, k_g = interp.splrep(x, y_g, s = 0, k = 4)
        bspl_g = interp.BSpline(t_g, c_g, k_g, extrapolate= False)
        y_g_int = bspl_g(x_intp)

        t_r, c_r, k_r = interp.splrep(x, y_r, s = 0, k = 4)
        bspl_r = interp.BSpline(t_r, c_r, k_r, extrapolate= False)
        y_r_int = bspl_r(x_intp)

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

    # wyczyszcenie list pomocniczych dla interpolacji wierszów
    x.clear()
    x_intp.clear()

    # dodanie odpowiednich wartości do vektora wykorzystanego później w interpolacji
    for i in range(scale * input_height):
            x_intp.append(i)

    # stworzenie wektora wykorzystanego w ramach interpolacji do wskazania położenia x dla wartości y_r...
    for i in range(input_height):
        x.append(scale * i)
    
    for i in range(output_width):
        # pobranie wartości kolorów B, G, R poszczególnych komórek w kolumnie
        for j in range(input_height):
            y_b.append(blank[j, i][0])
            y_r .append(blank[j, i][2])
            y_g.append(blank[j, i][1])

        t_b, c_b, k_b = interp.splrep(x, y_b, s = 0, k = 4)
        bspl_b = interp.BSpline(t_b, c_b, k_b, extrapolate= False)
        y_b_int = bspl_b(x_intp)

        t_g, c_g, k_g = interp.splrep(x, y_g, s = 0, k = 4)
        bspl_g = interp.BSpline(t_g, c_g, k_g, extrapolate= False)
        y_g_int = bspl_g(x_intp)

        t_r, c_r, k_r = interp.splrep(x, y_r, s = 0, k = 4)
        bspl_r = interp.BSpline(t_r, c_r, k_r, extrapolate= False)
        y_r_int = bspl_r(x_intp)

        # uzupełnienie płótna interpolowanymi wartościami
        for j in range(len(y_b_int)):
            increased_img[j, i] = ([y_b_int[j], y_g_int[j], y_r_int[j]])   

        y_b.clear()
        y_r.clear()
        y_g.clear()
        del y_b_int
        del y_g_int
        del y_r_int

    x.clear()
    x_intp.clear()

    return blank, increased_img

if __name__ == "__main__":
    img_org = cv.imread("kwiaty.jpg")
    img = img_org.copy()
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank = np.zeros((input_height, input_width, 3), np.float64)
    print(blank.shape)
    pois_number = input_width * input_height
    poisson = np.random.poisson(1024.0, pois_number)

    # plt.figure()
    # plt.hist(poisson)
    # plt.show()

    for i in range(input_height):
        for j in range(input_width):
            blank[i, j, 0] = img[i, j, 0] + poisson[i * input_height + j]
            blank[i, j, 1] = img[i, j, 1] + poisson[i * input_height + j]
            blank[i, j, 2] = img[i, j, 2] + poisson[i * input_height + j]

    output = cv.normalize(blank, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    

    maly = two_dim_decreasing(output, 10)
    _, outpu2 = two_dim_increasing(maly, 10)
    cv.imshow("kwiaty szum", output)
    cv.imshow("kwiaty", img)
    cv.imshow("zmniejszony", maly)
    cv.imshow("powiekszony",outpu2)
    cv.waitKey(0)
