from matplotlib.image import imread
import numpy as np
from matplotlib.pylab import plt
import os
import pywt


KEEP_TOP_VAL = 0.1
KEEP_BOTTOM_VAL = 0.5

# zmienne globalne dla keep factor size
MIN_VAL = 159600
MAX_VAL = 490000
MIN_TRANSFORMED = 0.8
MAX_TRANSFORMED = 1


class Photo:
    """FFT różne sposoby na wybór thresh i wartość zmiennej fft_thresh_type dla nich:
    1 - średnia wartość wartości bezwzględnych amplitud,
    2 - """

    def __init__(self, path: str, fft_thresh_type: int = 1) -> None:

        # atrybuty oryginalnego zdjęcia
        self.img = imread(os.path.join(path))
        self.fft_type = fft_thresh_type
        self.red = self.img[:, :, 0]
        self.green = self.img[:, :, 1]
        self.blue = self.img[:, :, 2]

        # atrybuty dla FFT
        self.fft_decompressed_img: np.ndarray = None
        self.fft_d_red: np.ndarray = None
        self.fft_d_green: np.ndarray = None
        self.fft_d_blue: np.ndarray = None
        self.fft_zero_val_coefs: float = None
        self.fft_mae_val: float = None

        # atrybuty dla transformaty falkowej
        self.wavelet_decompressed_img: np.ndarray = None
        self.wavelet_d_red: np.ndarray = None
        self.wavelet_d_green: np.ndarray = None
        self.wavelet_d_blue: np.ndarray = None
        self.wavelet_zero_val_coefs: float = None
        self.wavelet_mae_val: float = None

        # wywołanie metod
        self.fft()
        self.wavelet()
        self.mae()

    def show_img(self) -> None:
        """Metoda wyświetla oryginalne zdjęcie."""

        plt.figure()
        plt.title("photo")
        plt.imshow(self.img)
        plt.axis("off")
        plt.show()

    def show_fft_decompressed_img(self) -> None:
        """Metoda wyświetla zdjęcie po kompresji i późniejszej dekompresji."""

        plt.figure()
        plt.title("decompressed photo fft")
        plt.imshow(self.fft_decompressed_img)
        plt.axis("off")
        plt.show()

    def show_wavelet_decompressed_img(self) -> None:
        """Metoda wyświetla zdjęcie po kompresji i późniejszej dekompresji."""

        plt.figure()
        plt.title("decompressed photo wavelet")
        plt.imshow(self.wavelet_decompressed_img)
        plt.axis("off")
        plt.show()

    def show_img_components(self) -> None:
        """Metoda wyświetla nasycenie pikseli dla poszczególnych kanałów kolorystycznych RGB."""

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("red color chanel")
        plt.imshow(self.red, cmap="gray")
        plt.axis("off")
        plt.subplot(3, 1, 2)
        plt.title("green color chanel")
        plt.imshow(self.green, cmap="gray")
        plt.axis("off")
        plt.subplot(3, 1, 3)
        plt.title("blue color chanel")
        plt.imshow(self.blue, cmap="gray")
        plt.axis("off")
        plt.show()

    def keep_factor_size(self):
        """Metoda zwraca wartość od 0.8 do 1 w zależności od rozmiarów zdjęcia. 0.8 przy liczbie pikseli zdjęcia z najmniejszą ich liczbą i 1 dla liczby pikseli zdjęcia z najwięskzą ich liczbą."""

        normalized_val = (self.img.shape[0] * self.img.shape[1]) / (MAX_VAL - MIN_VAL)
        transformed_val = MIN_TRANSFORMED + normalized_val * (MAX_TRANSFORMED - MIN_TRANSFORMED)

        return transformed_val


    def fft_one_color_chanel(self, input_img: np.ndarray):

        keep = 0.1

        Bt = np.fft.fft2(input_img)
        Btsort = np.sort(np.abs(Bt.reshape(-1)))

        show = False
        if show is True:
            plt.figure()
            plt.plot(Btsort)
            plt.show()
        
        mean = np.mean(Btsort)

        # różne sposoby na określenie wartości thresh
        if self.fft_type == 1:
            thresh = mean

        if self.fft_type == 2:
            thresh = Btsort[int(np.floor((1-keep) * len(Btsort)))]

        ind = np.abs(Bt) > thresh
        Atlow = Bt * ind
        Alow = np.fft.ifft2(Atlow).real
        Alow = np.clip(Alow, 0, 255).astype(np.uint8)
        zeros = np.size(ind) - np.count_nonzero(ind)

        return Alow, zeros
    
    def wavelet_one_color_chanel(self, input_img: np.ndarray):

        n = 4
        w = "db1"
        coeffs = pywt.wavedec2(input_img, wavelet = w, level = n)
        coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

        Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

        keep = 0.1
        thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind

        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format = 'wavedec2')
        img_decompressed = pywt.waverec2(coeffs_filt, wavelet = w)
        img_decompressed = np.clip(img_decompressed, 0, 255).astype(np.uint8)
        zeros = np.size(ind) - np.count_nonzero(ind)

        return img_decompressed, zeros

    def fft(self) -> None:
        """"""

        img_components = [self.red, self.green, self.blue]
        c_img_components: list [np.ndarray] = []
        c_zero_coef: list [int] = []

        for component in img_components:
            c_img, c_coef = self.fft_one_color_chanel(component)
            c_img_components.append(c_img)
            c_zero_coef.append(c_coef)

        self.fft_zero_val_coefs = sum(c_zero_coef) / (3 * self.img.shape[0] * self.img.shape[1])
        self.fft_d_red = c_img_components[0]
        self.fft_d_green = c_img_components[1]
        self.fft_d_blue = c_img_components[2]
        self.fft_decompressed_img = np.dstack(c_img_components)
        
    def wavelet(self) -> None:

        img_components = [self.red, self.green, self.blue]
        c_img_components: list [np.ndarray] = []
        c_zero_coef: list [int] = []

        for component in img_components:
            c_img, c_coef = self.wavelet_one_color_chanel(component)
            c_img_components.append(c_img)
            c_zero_coef.append(c_coef)

        self.wavelet_zero_val_coefs = sum(c_zero_coef) / (3 * self.img.shape[0] * self.img.shape[1])
        self.wavelet_d_red = c_img_components[0]
        self.wavelet_d_green = c_img_components[1]
        self.wavelet_d_blue = c_img_components[2]
        self.wavelet_decompressed_img = np.dstack(c_img_components)

    def mae(self) -> None:
        """Metoda oblicza wartośc MAE dla zdjęć po dekompresji i uzupełnia wartość dla odpowiednich atrybutów obiektu."""

        sum_val = 0
        n = self.img.shape[0] * self.img.shape[1] * 3   # liczba pikseli * 3 ze względu na 3 obrazy dla różnych kolorów

        for c in range(3):
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    sum_val += np.abs(int(self.img[i, j, c]) - int(self.fft_decompressed_img[i, j, c]))
        
        self.fft_mae_val = sum_val / (n * np.max(self.img))

        sum_val = 0

        for c in range(3):
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    sum_val += np.abs(int(self.img[i, j, c]) - int(self.wavelet_decompressed_img[i, j, c]))
        
        self.wavelet_mae_val = sum_val / (n * np.max(self.img))
        

if __name__ == "__main__":

    # photo1 = Photo("panda.jpg", 1)
    # photo1 = Photo("circle.jpg", 2)
    photo1 = Photo("mond.jpg", 2)
    photo1.show_img()
    photo1.show_img_components()
    photo1.show_fft_decompressed_img()
    photo1.show_wavelet_decompressed_img()
    print(photo1.fft_mae_val)
    print(photo1.fft_zero_val_coefs)
    print(photo1.wavelet_mae_val)
    print(photo1.wavelet_zero_val_coefs)



    
    


