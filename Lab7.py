from matplotlib.image import imread
import numpy as np
from matplotlib.pylab import plt
import os
import pywt


def fft_with_keeping(c_img: np.ndarray, keep) -> np.ndarray:
    Bt = np.fft.fft2(c_img)
    Btsort = np.sort(np.abs(Bt.reshape(-1)))
    # ind_top = Btsort[int(np.floor((1-keep) * len(Btsort)))]

    # opcja jakaś tam
    # ind_top = int(np.floor((1-keep) * len(Btsort)))
    # ind_bot = int(np.floor((keep * 3) * len(Btsort)))
    # mean = np.mean(Btsort[ind_bot:ind_top])

    mean = np.mean(Btsort)

    thresh = mean

    # plt.figure()
    # plt.plot(Btsort)
    # plt.show()
    # thresh = np.max(Btsort)
    ind = np.abs(Bt) > thresh
    Atlow = Bt * ind
    Alow = np.fft.ifft2(Atlow).real
    Alow = np.clip(Alow, 0, 255).astype(np.uint8)
    zeros = np.size(ind) - np.count_nonzero(ind)

    return Alow, zeros

def mae(img_Xo: np.ndarray, img_Xd: np.ndarray) -> float:

    n = img_Xd.shape[0] * img_Xd.shape[1] * 3
    suma = 0

    for c in range(3):
        for i in range(img_Xd.shape[0]):
            for j in range(img_Xd.shape[1]):
                suma += np.abs(img_Xo[i, j, c] - img_Xd[i, j, c])
    
    max_Xo = img_Xo.max()
    
    return suma / (n * max_Xo)

class Photo:

    def __init__(self, path: str) -> None:

        self.img = imread(os.path.join(path))
        self.red = self.img[:, :, 0]
        self.green = self.img[:, :, 1]
        self.blue = self.img[:, :, 2]
        self.compressed_img: np.ndarray = None
        self.c_red: np.ndarray = None
        self.c_green: np.ndarray = None
        self.c_blue: np.ndarray = None
        self.zero_val_coef: float = None
        self.mae_val: float = None
        self.coef_num: int = None
        self.fft()
        self.mae()
        # self.c_red = self.compressed_img[:, :, 0]
        # self.C_green = self.compressed_img[:, :, 1]
        # self.c_blue = self.compressed_img[:, :, 2]

    def show_img(self) -> None:
        plt.figure()
        plt.title("photo")
        plt.imshow(self.img)
        plt.axis("off")
        plt.show()

    def show_decompressed_img(self) -> None:
        plt.figure()
        plt.title("decompressed photo")
        plt.imshow(self.compressed_img)
        plt.axis("off")
        plt.show()

    def show_img_components(self) -> None:
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
        mean = np.mean(Btsort)
        thresh = mean
        ind = np.abs(Bt) > thresh
        Atlow = Bt * ind
        Alow = np.fft.ifft2(Atlow).real
        Alow = np.clip(Alow, 0, 255).astype(np.uint8)
        zeros = np.size(ind) - np.count_nonzero(ind)
        coef_num = np.size(ind)

        return Alow, zeros, coef_num
    
    def fft(self):

        img_components = [self.red, self.green, self.blue]
        c_img_components: list [np.ndarray] = []
        c_zero_coef: list [int] = []
        c_num = 0

        for component in img_components:
            c_img, c_coef, c_num_new = self.fft_one_color_chanel(component)
            c_num += c_num_new
            c_img_components.append(c_img)
            c_zero_coef.append(c_coef)

        self.zero_val_coef = sum(c_zero_coef) / 3 # średnia liczba niezerowych współczynników
        self.coef_num = c_num
        self.c_red = c_img_components[0]
        self.c_green = c_img_components[1]
        self.c_blue = c_img_components[2]
        self.compressed_img = np.dstack(c_img_components)
        

    def mae(self):

        sum_val = 0
        n = self.img.shape[0] * self.img.shape[1] * 3

        for c in range(3):
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    sum_val += np.abs(int(self.img[i, j, c]) - int(self.compressed_img[i, j, c]))
        
        self.mae_val = sum_val / (n * np.max(self.img))

if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = [5, 5]
    # plt.rcParams.update({"font.size": 18})


    photo1 = Photo("panda.jpg")
    photo1.show_img()
    photo1.show_img_components()
    # photo1.fft()
    photo1.show_decompressed_img()
    # photo1.mae()
    print(photo1.mae_val)
    print(photo1.coef_num)
    print(photo1.zero_val_coef)


    # A =  imread(os.path.join("panda.jpg"))
    # # A =  imread(os.path.join("circle.jpg"))
    # # A =  imread(os.path.join("mond.jpg"))
    # A_r = A[:, :, 0]
    # A_g = A[:, :, 1]
    # A_b = A[:, :, 2]
    # A_components = [A_r, A_g, A_b]

    # Fourier = True
    # Wavelet = False

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



    # # Wavelet

    # if Wavelet is True:

    #     B = np.mean(A, -1)

    #     n = 4
    #     w = "db1"
    #     coeffs = pywt.wavedec2(B, wavelet = w, level = n)
    #     coeff_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

    #     Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    #     for keep in (0.1, 0.05, 0.01, 0.005):
    #         thresh = Csort[int(np.floor((1 - keep) * len(Csort)))]
    #         ind = np.abs(coeff_arr) > thresh
    #         Cfilt = coeff_arr * ind

    #         coeffs_filt = pywt.array_to_coeffs(Cfilt, coeffs_slices, output_format = 'wavedec2')
    #         Arecon = pywt.waverec2(coeffs_filt, wavelet = w)
    #         plt.figure()
    #         plt.imshow(Arecon.astype(np.uint8), cmap="gray")
    #         plt.axis("off")

    #         # czy to jest to compressed zdj?
    #         plt.figure()
    #         plt.imshow(np.clip(Cfilt, 0, 255).astype(np.uint8), cmap="gray")
    #         plt.show()
        
    #     plt.show()



    
    


