import numpy as np
import cv2 as cv
from Lab3 import conv_interp
from Lab5 import mse

def cmos(img, Bayer = False, Fuji = True, save = False, filename = "obraz"):
    #pobranie wymiarów obrazu wejściowego
    input_height = img.shape[0]
    input_width = img.shape[1]

    #stworzenie podstawowej maski, wyznaczenie wartości skali oraz przypisanie typu macierzy na podstawie wybranego typu matrycy cmos
    if(Bayer is True):
        mask_g = np.array([[1, 0], [0, 1]], dtype = np.uint8)
        mask_b = np.array([[0, 0], [1, 0]], dtype = np.uint8)
        mask_r = np.array([[0, 1], [0, 0]], dtype = np.uint8)
        scale_h = input_height // 2
        scale_w = input_width // 2
        matrix_type = "Bayer"

    elif(Fuji is True):
        mask_g = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]], dtype = np.uint8)
        mask_b = np.array([[0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype = np.uint8)
        mask_r = np.array([[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], dtype = np.uint8)
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

def green_component_interpolation(img_org, show = False, save = False, kernel = 5):

    img = img_org.copy()
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank = np.zeros((input_height, input_width), dtype=np.uint8)

    y = []
    x = []
    x_intp = []

    #wyodrębnienie przestrzeni kolorystrycznych
    green_component = img[:, :, 1]

    for i in range(input_height):
        # wypełnienie wartości x i y do interpolacji
        for j in range(input_width):
            if(green_component[i, j] != 0):
                y.append(green_component[i, j])
                x.append(j)

        # uzupełnienie wartości interpolowanego x   
        for j in range(x[0], x[-1] + 1):
            x_intp.append(j)

        # przygotowanie do interpolacji
        period = x[1] - x[0]
        kernel_grid = np.tile(x_intp, (len(x), 1))
        offset = np.tile(np.expand_dims(x, axis= -1), (1, len(x_intp)))
        kernel_grid = (kernel_grid - offset) / period

        # dokonanie interpolacji
        y_intp = conv_interp(y, kernel_grid, kernel)
    
        # uzupełnienie pierwszej/ostatniej wartości dla x_intp i y_intp ze względu na rozpoczynanie się/kończenie tych wartości w odległości 1 komórki od krawędzi
        if(x_intp[0] != 0):
            x_intp.insert(0, 0)
            temp = y_intp[0]
            y_intp = np.insert(y_intp, 0, temp)
        
        if(x_intp[-1] != input_width - 1):
            x_intp.append(input_width - 1)
            temp = y_intp[-1]
            y_intp = np.append(y_intp, [temp])
        
        # uzupełnienie wiersza w płótnie
        for j in range(input_width):
            blank[i, j] = y_intp[j]

        # wyczyszcenie array'ów/ list
        x.clear()
        y.clear()
        x_intp.clear()
        del y_intp
    
    if(show is True):
        cv.imshow("green", blank)
        cv.waitKey(0)

    if(save is True):
        cv.imwrite("green_component" + f"_kernel_{kernel}" + ".png", blank)

    return blank

def blue_red_component_interpolation(img_org, red = False, blue = False, show = False, save = False, kernel = 5):

    img = img_org.copy()
    input_height = img.shape[0]
    input_width = img.shape[1]
    blank = np.zeros((input_height, input_width), dtype=np.uint8)
    blank2 = blank.copy()

    y = []
    x = []
    x_intp = []
    #wyodrębnienie przestrzeni kolorystrycznych i przypisanie odpowednich wartości zmiennym w zależności od interpolowanego koloru
    if(red is True):
        color_component = img[:, :, 2]
        start_h = 0
        stop_h = input_height

    elif(blue is True):
        color_component = img[:, :, 0]
        start_h = 1
        stop_h = input_height + 1

    # krok dla wysokości równy 2 bo w co drugim rzędzie nie ma wartości możliwych wo wykorzystania w ramach interpolacji
    for i in range(start_h, stop_h, 2):
        # wypełnienie wartości x i y do interpolacji
        for j in range(input_width):
            if(color_component[i, j] != 0):
                y.append(color_component[i, j])
                x.append(j)

        # uzupełnienie wartości interpolowanego x   
        for j in range(x[0], x[-1] + 1):
            x_intp.append(j)

       # przygotowanie do interpolacji
        period = x[1] - x[0]
        kernel_grid = np.tile(x_intp, (len(x), 1))
        offset = np.tile(np.expand_dims(x, axis= -1), (1, len(x_intp)))
        kernel_grid = (kernel_grid - offset) / period

        # dokonanie interpolacji
        y_intp = conv_interp(y, kernel_grid, kernel)
    
        # uzupełnienie pierwszej/ostatniej wartości dla x_intp i y_intp ze względu na rozpoczynanie się/kończenie tych wartości w odległości 1 komórki od krawędzi
        if(x_intp[0] != 0):
            x_intp.insert(0, 0)
            temp = y_intp[0]
            y_intp = np.insert(y_intp, 0, temp)
        
        if(x_intp[-1] != input_width - 1):
            x_intp.append(input_width - 1)
            temp = y_intp[-1]
            y_intp = np.append(y_intp, [temp])
        
        # uzupełnienie wiersza w płótnie
        for j in range(input_width):
            blank[i, j] = y_intp[j]

        # wyczyszcenie array'ów/ list
        x.clear()
        y.clear()
        x_intp.clear()
        del y_intp

    # drugi etap interpolacji - kolumny

    for i in range(input_width):
        # wypełnienie wartości x i y do interpolacji
        for j in range(input_height):
            if(blank[j, i] != 0):
                y.append(blank[j, i])
                x.append(j)

        # uzupełnienie wartości interpolowanego x   
        for j in range(x[0], x[-1] + 1):
            x_intp.append(j)

       # przygotowanie do interpolacji
        period = x[1] - x[0]
        kernel_grid = np.tile(x_intp, (len(x), 1))
        offset = np.tile(np.expand_dims(x, axis= -1), (1, len(x_intp)))
        kernel_grid = (kernel_grid - offset) / period

        # dokonanie interpolacji
        y_intp = conv_interp(y, kernel_grid, kernel)
    
        # uzupełnienie pierwszej/ostatniej wartości dla x_intp i y_intp ze względu na rozpoczynanie się/kończenie tych wartości w odległości 1 komórki od krawędzi
        if(x_intp[0] != 0):
            x_intp.insert(0, 0)
            temp = y_intp[0]
            y_intp = np.insert(y_intp, 0, temp)
        
        if(x_intp[-1] != input_width - 1):
            x_intp.append(input_width - 1)
            temp = y_intp[-1]
            y_intp = np.append(y_intp, [temp])
        
        # uzupełnienie wiersza w płótnie
        for j in range(input_width):
            blank2[j, i] = y_intp[j]

        # wyczyszcenie array'ów/ list
        x.clear()
        y.clear()
        x_intp.clear()
        del y_intp
    
    if(red is True):
        show_name = "red"
        file_name = "red_component" + f"_kernel_{kernel}" + ".png"
    else:
        show_name = "blue"
        file_name = "blue_component" + f"_kernel_{kernel}" + ".png"

    if(show is True):
        cv.imshow(show_name, blank2)
        cv.waitKey(0)

    if(save is True):
        cv.imwrite(file_name, blank2)

    return blank2

def demosaic(img_org, show = False, save = False, show_all = False, save_all = False, kernel = 5):
    img = img_org.copy()
    blue_component = blue_red_component_interpolation(img, blue=True, save=save_all, show=show_all, kernel=kernel)
    red_component = blue_red_component_interpolation(img, red=True, save=save_all, show=show_all, kernel=kernel)
    green_component = green_component_interpolation(img, save=save_all, show=show_all, kernel=kernel)
    demosaiced = cv.merge((blue_component, green_component, red_component))

    if(save is True or save_all is True):
        cv.imwrite("demosaiced" + f"_kernel_{kernel}" + ".png", demosaiced)

    if(show_all):
        cv.imshow("demosaiced", demosaiced)
        cv.imshow("red", red_component)
        cv.imshow("blue", blue_component)
        cv.imshow("green", green_component)
        cv.waitKey(0)
    
    elif(show is True):
        cv.imshow("demosaiced", demosaiced)
        cv.waitKey(0)

    return demosaiced

if __name__ == "__main__":

    img = cv.imread("kot.jpg")
    img = cv.resize(img.copy(), (228, 228), interpolation=cv.INTER_AREA)
    kernel_list = [2, 3, 4, 5]
    mse_values = []

    # wybór 
    Bayer_part = False
    Fuji_part = True

    if(Bayer_part is True):
        for i in range(len(kernel_list)):
            # maska cmos
            img_cmos = cmos(img, Bayer = True, Fuji = False, save = True, filename = "kot228x228")
            # demozajkowanie
            demosaiced = demosaic(img_cmos, save_all=True, kernel= kernel_list[i])
            mse_values.append(mse(img, demosaiced))
        
        with open("Wartosci_mse_demosaiced.txt", "w") as f:
            for i in range(len(mse_values)):
                f.write("Dla jądra interpolacji: " + f"{kernel_list[i]}" + " wartość MSE wynosi: " + f"{mse_values[i]}" + ".\n")

    if(Fuji_part is True):
        img_cmos = cmos(img, Bayer = False, Fuji = True, save = True, filename = "kot_Fuji_228x228")
        
    cv.imshow("cmos_img", img_cmos)
    cv.waitKey(0)




