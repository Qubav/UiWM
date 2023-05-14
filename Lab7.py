from matplotlib.image import imread
import numpy as np
from matplotlib.pylab import plt
import os
import pywt


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

    def show_wavelet_decompressed_img(self):
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

    def fft_one_color_chanel(self, input_img: np.ndarray):

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
            x = 0

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

        keep = 0.05
        thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind

        coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format = 'wavedec2')
        img_decompressed = pywt.waverec2(coeffs_filt, wavelet = w).astype(np.uint8)
        zeros = np.size(ind) - np.count_nonzero(ind)

        return img_decompressed, zeros

    def fft(self):

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
        
    def wavelet(self):

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

    def mae(self):

        sum_val = 0
        n = self.img.shape[0] * self.img.shape[1] * 3

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
    # plt.rcParams["figure.figsize"] = [5, 5]
    # plt.rcParams.update({"font.size": 18})


    photo1 = Photo("panda.jpg", 1)
    photo1.show_img()
    photo1.show_img_components()
    # photo1.fft()
    photo1.show_fft_decompressed_img()
    photo1.show_wavelet_decompressed_img()
    # photo1.mae()
    print(photo1.fft_mae_val)
    print(photo1.fft_zero_val_coefs)


    A =  imread(os.path.join("panda.jpg"))
    # A =  imread(os.path.join("circle.jpg"))
    # A =  imread(os.path.join("mond.jpg"))
    A_r = A[:, :, 0]
    A_g = A[:, :, 1]
    A_b = A[:, :, 2]
    A_components = [A_r, A_g, A_b]

    # Fourier = True
    Wavelet = True

    # # # FFT

    # if Fourier is True:

    #     images_decompressed = []
    #     images_compressed = []

    #     plt.figure()
    #     plt.subplot(3, 1, 1)
    #     plt.imshow(A_r, cmap="gray")
    #     plt.subplot(3, 1, 2)
    #     plt.imshow(A_g, cmap="gray")
    #     plt.subplot(3, 1, 3)
    #     plt.imshow(A_b, cmap="gray")
    #     plt.axis("off")
        
        
        
    #     # for keep in (0.1, 0.05, 0.01, 0.002):
    #     for keep in (0.1, 0.2, 0.3, 0.4):
    #         A_new  = []

    #         for component in A_components:
    #             Alow, zeros = fft_with_keeping(component, keep)
    #             A_new.append(Alow)

    #         A_new_2 = np.dstack(A_new)
    #         # mae = mae(A, A_new_2)
    #         images_decompressed.append(A_new_2)


    #     for Alow in images_decompressed:
    #         plt.figure()
    #         plt.imshow(Alow, cmap="gray")
    #         plt.axis("off")
        
            
    #     plt.show()



    # Wavelet

    if Wavelet is True:

        B = np.mean(A, -1)

        n = 4
        w = "db1"
        coeffs = pywt.wavedec2(B, wavelet = w, level = n)
        coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

        Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

        for keep in (0.1, 0.05, 0.01, 0.005):
            thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
            ind = np.abs(coeff_arr) > thresh
            Cfilt = coeff_arr * ind

            coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format = 'wavedec2')
            Arecon = pywt.waverec2(coeffs_filt, wavelet = w)
            plt.figure()
            plt.imshow(Arecon.astype(np.uint8), cmap="gray")
            plt.axis("off")

        plt.figure()
        plt.imshow(B, cmap="gray")
        plt.show()



    
    


