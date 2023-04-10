import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

# funckja tworzy wykres funkcji opisanej na współrzędnych biegównych dającej obraz śmigła, następnie poddaje go aliasingowi i zapisuje na dysku film z kolejnymi klatkami,
# wykres funkcji oraz wykres funkcji po nałożeniu aliasingu
# res -> rozdzielczość, n = 3 lub n = 5 -> ilość łopatek śmigła, big_l -> ilość wierszy branych z konkretnej klatki
def aliasing(res = 256, n = 3, big_l = 2, filename_function = "Przebieg_funkcji", filename_aliasing = "Przebieg_funkcji_z_aliasingiem", video_name = "Video_aliasing"):
    blank = np.zeros(((res, res, 3)), dtype=np.uint8)
    big_m = 64
    m = 32
    frames = int(res/big_l)
    imgs = []

    for i in range(frames):
        #wyznaczenie współrzędnych biegunowych
        m = - big_m / 2 + i % big_m
        theta = np.arange(0, 3 * np.pi, 0.01)
        r = np.sin(n * theta + m * np.pi / 10)

        #stworzenie wykresu
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        fig.set_size_inches(4, 4)
        fig.set_dpi(res / 4)
        ax.plot(theta, r)
        ax.grid(True)

        # zmiana pyplot figure na array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        #wypełnienie kolejnych wierszy płótna odpowiednimi wierszami obecnie wygenerowanego wykresu
        #pierwsza pętla odopowiada za wiersz, druga za kolumnę
        for j in range( big_l):
            for k in range(res):
                if (i * big_l + j) >= res:
                    break
                blank[(i * big_l + j), k] = data[(i * big_l + j), k]
        
        #wypełnianie obrazu który zostanie wykorzystany w filmie
        temp = blank.copy()
        for j in range(big_l, res):
            for k in range(res):
                if (i * big_l + j) >= res:
                    break
                temp[(i * big_l + j), k] = data[(i * big_l + j), k]
        imgs.append(temp.copy())

    # zapisanie wykresów oryginalnego przebiegu funkcji i wykresu po aliasingu
    cv.imwrite(filename_function + f"_{n}_{big_l}_{res}" + ".png", data)
    cv.imwrite(filename_aliasing + f"_{n}_{big_l}_{res}" + ".png", blank)

    # stworzenie i zapisanie filmu na podstawie wygenerowanych obrazów
    cv_fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(video_name + f"_{n}_{big_l}_{res}" + ".mp4", cv_fourcc, len(imgs), (res, res))
    z = 0
    for i in imgs:
        video.write(i)
        # cv.imwrite(filename_aliasing + f"_{n}_{big_l}_{res}_{z}" + ".png", i)
        # z = z + 1

    video.release()
    print(len(imgs))

if __name__ == "__main__":
    aliasing(n = 5, big_l = 16, res = 256)