import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as interp

def kernel_h1(t):
    if(t >= 0 and t <= 1):
        return 1
    else:
        return 0

def kernel_h3(t):
    if(t >= -1 and t <= 1):
        return 1 - abs(t)
    else:
        return 0

def kernel_h2(t):
    if(t >= - 1 / 2 and t <= 1 / 2):
        return 1
    else:
        return 0

def kernel_h4(t):
    if(t == 0):
        return 1
    else:
        return np.sin(t) / t / np.pi

def kernel_h5(t):
    if(t == 0):
        return 1
    elif(abs(t) > 0 and abs(t) < 1):
        return 3 / 2 * abs(t) ** 3 - 5 / 2 * abs(t) ** 2 + 1
    elif(abs(t) > 1 and abs(t) < 2):
        return - 1 / 2 * abs(t) ** 3 + 5 / 2 * abs(t) ** 2 - 4 * abs(t) + 2
    else:
        return 0

def bspline_interpolation(s_n = 10, i_s_n = 30, n = 3, sin_x = False, sin_x_power_minus_1 = False, signum_sin_8x = False):
    # wygenerowanie rzeczywistego przebiegu
    y_real = []
    x_real = np.linspace(0.001, 2 * np.pi, 10**4)
    for i in x_real:
        if(sin_x):
            y_real.append(np.sin(i))
        elif(sin_x_power_minus_1):
            y_real.append(np.sin(i ** (-1)))
        elif(signum_sin_8x):
            y_real.append(np.sign(np.sin(8 * i)))

    # wygenerowanie punktów pomiarowych
    y = []
    x = np.linspace(0.001, 2 * np.pi, s_n)
    for i in x:
        if(sin_x):
            y.append(np.sin(i))
        elif(sin_x_power_minus_1):
            y.append(np.sin(i**(-1)))
        elif(signum_sin_8x):
            y.append(np.sign(np.sin(8 * i)))
    
    # interpolacja wartosci funkcji w punktach na podstawie punktów pomiarowych
    xvals = np.linspace(0.001, 2 * np.pi, i_s_n)
    t, c, k = interp.splrep(x, y, s = 0, k = n)
    bspl = interp.BSpline(t, c, k, extrapolate= False)
    yvals = bspl(xvals)
    
    # wyliczenie błędu średniokwadratowego
    L = 0
    if(sin_x is True):
        for i in range(i_s_n):
            L = L + (np.sin(xvals[i])-yvals[i])**2
    elif(sin_x_power_minus_1 is True):
        for i in range(i_s_n):
            L = L + (np.sin(xvals[i]**(-1))-yvals[i])**2
    elif(signum_sin_8x is True):
        for i in range(i_s_n):
            L = L + (np.sign(np.sin(8 * xvals[i]))-yvals[i])**2
    L = L / i_s_n

    # wyświetlenie wykresu oraz wypisanie wartośći błędu średniokwadratowego
    plt.figure(2)
    plt.plot(x_real, y_real, lw = 4, label = "rezczywisty przebieg funkcji", color = "darkorange")
    plt.plot(x, y, 'o', label = "wartość funkcji w punkcie pomiaru", color = "black")
    plt.plot(xvals, yvals, label = "interpolowany przebieg funkcji", color = "royalblue")
    plt.title("Wykres interpolacji 1D przy użyciu funkcji sklejanych")
    plt.grid(True)
    if(signum_sin_8x is False): 
        plt.legend(loc = "upper right")
    else:
        plt.legend(loc = "center right")
    print("Wartość błędu średniokwadratowego dla interpolacji funkcjami sklejanymi wynosi:", L, "\n")
    plt.show()

def conv_interp(s_n = 10, i_s_n = 30, kernel = 5, sin_x = False, sin_x_power_minus_1 = False, signum_sin_8x = False) :
    # wygenerowanie rzeczywistego przebiegu
    y_real = []
    x_real = np.linspace(0.001, 2 * np.pi, 10**4)
    
    if(sin_x):
        for i in x_real:
            y_real.append(np.sin(i))
    elif(sin_x_power_minus_1):
        for i in x_real:
            y_real.append(np.sin(i ** (-1)))
    elif(signum_sin_8x):
        for i in x_real:
            y_real.append(np.sign(np.sin(8 * i)))
    
    # wygenerowanie punktów pomiarowych
    y = []
    x = np.linspace(0.001, 2 * np.pi, s_n)
    
    if(sin_x):
        for i in x:
            y.append(np.sin(i))
    elif(sin_x_power_minus_1):
        for i in x:
            y.append(np.sin(i**(-1)))
    elif(signum_sin_8x):
        for i in x:
            y.append(np.sign(np.sin(8 * i)))
    
    xvals = np.linspace(10**(-6), 2 * np.pi, i_s_n)
    period = x[1] - x[0]
    kernel_grid = np.tile(xvals, (len(x), 1))
    offset = np.tile(np.expand_dims(x, axis= -1), (1, len(xvals)))
    kernel_grid = (kernel_grid - offset) / period

    for i in range(kernel_grid.shape[0]):
        for j in range(kernel_grid.shape[1]):
            if (kernel == 1):
                kernel_grid[i, j] = kernel_h1(kernel_grid[i, j])
            elif (kernel == 2):
                kernel_grid[i, j] = kernel_h2(kernel_grid[i, j])
            elif (kernel == 3):
                kernel_grid[i, j] = kernel_h3(kernel_grid[i, j])
            elif (kernel == 4):
                kernel_grid[i, j] = kernel_h4(kernel_grid[i, j])
            elif (kernel == 5):
                kernel_grid[i, j] = kernel_h5(kernel_grid[i, j])
            else:
                print("Wprowadzono niepoprawną wartość zmiennej kernel!\n")
                
    yvals = np.dot(y, kernel_grid)
    
    # wyliczenie błędu średniokwadratowego
    L = 0
    if(sin_x is True):
        for i in range(i_s_n):
            L = L + (np.sin(xvals[i])-yvals[i])**2
    elif(sin_x_power_minus_1 is True):
        for i in range(i_s_n):
            L = L + (np.sin(xvals[i]**(-1))-yvals[i])**2
    elif(signum_sin_8x is True):
        for i in range(i_s_n):
            L = L + (np.sign(np.sin(8 * xvals[i]))-yvals[i])**2
    L = L / i_s_n
    print("Wartość błędu średniokwadratowego dla interpolacji konwolucją wynosi:", L, "\n")

    plt.figure(1)
    plt.plot(x_real, y_real, lw = 4, label = "rezczywisty przebieg funkcji", color = "darkorange")
    plt.plot(x, y, 'o', label = "wartość funkcji w punkcie pomiaru", color = "black")
    plt.plot(xvals, yvals, label = "interpolowany przebieg funkcji", color = "royalblue")
    plt.title("Wykres interpolacji 1D przy użyciu funkcji sklejanych")
    plt.grid(True)
    if(signum_sin_8x is False): 
        plt.legend(loc = "upper right")
    else:
        plt.legend(loc = "center right")
    plt.title("Wykres interpolacji 1D przy użyciu konwolucji")
    plt.show()


if __name__ == "__main__":
    # v_s_n = [16, 16*3, 16*5, 16*7]
    # for i in range(0, 4):
    #     bspline_interpolation(sin_x=False, sin_x_power_minus_1=False, signum_sin_8x=True,  s_n=v_s_n[i], i_s_n=10*v_s_n[i], n=1)
    # bspline_interpolation(sin_x=False, sin_x_power_minus_1=False, signum_sin_8x=True,  s_n=16*5, i_s_n=16*16, n=1)
    # for i in range(1, 6):
    #     conv_interp(s_n = 50, i_s_n = 500, sin_x = False, sin_x_power_minus_1=True, signum_sin_8x=False,  kernel=i)

    gg = [10, 25, 50]
    for i in range(0, 3):
        conv_interp(s_n = gg[i], i_s_n = gg[i] * 10, sin_x = True, sin_x_power_minus_1=False, signum_sin_8x=False,  kernel=5)
