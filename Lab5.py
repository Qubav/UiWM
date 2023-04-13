import numpy as np
from scipy import interpolate as interp
import cv2 as cv
from Lab3 import two_dim_increasing
import time

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

def two_dim_increasing_BSpline(img, scale, show = False):
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
    for i in range(scale * input_width):
            x_intp.append(i)

    # stworzenie wektora wykorzystanego w ramach interpolacji do wskazania położenia x dla wartości y_r...
    for i in range(input_width):
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

    if(show is True):
        print("Rozmiar oryginalnego obrazu wynosi", img_org.shape, ", a rozmiar zmniejszonego obrazu wynosi", blank.shape)
        cv.imshow("Oryginalny obraz", img_org)
        cv.imshow("Zmniejszony obraz", blank)
        cv.waitKey(0)
    
    return blank, increased_img

def poissoning(image: np.array, lambda_value: float) -> np.ndarray:
    """
    :param image: numpy array of shape (H, W, C) or (H, W)
    :param lambda_value: number of simulated photons per pixel (or per channel, depending on the image)
    :return:
    """
    noised_image = np.random.poisson(image / image.max() * lambda_value) / lambda_value
    noised_image = np.clip(noised_image * 255, 0, 255).astype(np.uint8)
    return noised_image

def mse(img_org_, img_interp_):
    img_org = img_org_.copy()
    img_interp = img_interp_.copy()
    sum = 0
    for i in range(img_interp.shape[0]):
        for j in range(img_interp.shape[1]):
            for c in range(3):
                sum +=  (int(img_org[i, j, c]) - int(img_interp[i, j, c])) ** 2
    sum = sum / (img_interp.shape[0] * img_interp.shape[1])

    return sum

def mae(img_org_, img_interp_):
    img_org = img_org_.copy()
    img_interp = img_interp_.copy()
    sum = 0
    for i in range(img_interp.shape[0]):
        for j in range(img_interp.shape[1]):
            for c in range(3):
                sum += abs(int(img_org[i, j, c]) - int(img_interp[i, j, c]))

    sum = sum / (img_interp.shape[0] * img_interp.shape[1])

    return sum

if __name__ == "__main__":
    img_org = cv.imread("kwiaty.jpg")
    lamb = [1, 4, 16, 64, 256, 1024]
    scale = 10
    # ker = 3
    ker = "BSpline"
    mse_values = []
    mae_values = []

    for i in range(len(lamb)):
        time_start = time.time()
        img_poiss = poissoning(img_org, lamb[i])
        img_dec = two_dim_decreasing(img_poiss, scale)
        # _, img_inc = two_dim_increasing(img_dec, scale, True, kernel= ker)
        _, img_inc = two_dim_increasing_BSpline(img_dec, 10, show=True)
        mae_val = mae(img_poiss, img_inc)
        mae_values.append(mae_val)
        mse_val = mse(img_poiss, img_inc)
        mse_values.append(mse_val)
        time_end = time.time()
        print(" Czas trwania operacji dla obrazów przy lambda" + f"{lamb[i]}" + "wynosi", time_end - time_start, ".\n")
        # cv.imwrite("img_dec_" + f"lambda_{lamb[i]}_kernel_{ker}" + ".png", img_dec)
        # cv.imwrite("img_inc_" + f"lambda_{lamb[i]}_kernel_{ker}" + ".png", img_inc)
        # cv.imwrite("img_poiss_" + f"lambda_{lamb[i]}_kernel_{ker}" + ".png", img_poiss)
        print("Dla lambda =" + f"lambda_{lamb[i]}" + "wartość MAE wynosi: " + f"{mae_val}" + ", a wartość MSE wynosi: " + f"{mse_val}" + ".\n")
    
    with open("Wartosci_" + f"lambda_{lamb[i]}_kernel_{ker}" + ".txt", "w") as f:
            for i in range(len(lamb)):
                f.write("Dla parametru lambda: " + f"{lamb[i]}" + " wartość MSE wynosi: " + f"{mse_values[i]}" + ", a wartość MAE wynosi: " + f"{mae_values[i]}" + ".\n")

    print("Koniec!\n")

