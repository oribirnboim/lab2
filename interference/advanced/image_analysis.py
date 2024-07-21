import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from PIL import Image
from typing import *
import os



d1 = array([66, 70, 78, 87, 95, 105, 120, 130, 142, 156, 180, 195, 125, 135, 165])
c1 = array([(710, 655), (706, 642), (690, 596), (675, 550), (662, 505), (745, 527), (727, 552), (728, 490), (685, 557), (719, 635), (592, 600), (860, 560), (802, 600), (746, 622), (721, 545)])
d2 = array([])


def load_image(path: str) -> array:
    return array(Image.open(path).convert('L'))


def analyse_image(image: array) -> array:
    res = array([])
    return res


def view_folder(path: str) -> None:
    for f in os.listdir(path):
        if f.endswith('.png'):
            image_path = os.path.join(path, f)
            image = load_image(image_path)
            plt.imshow(image, cmap='gray')
            plt.show()


def view_1() -> None:
    path = 'camera_roll'
    ls_dir = os.listdir(path)
    for i in range(len(ls_dir)):
        f = ls_dir[i]
        if f.endswith('.png'):
            image_path = os.path.join(path, f)
            image = load_image(image_path)
            plt.imshow(image, cmap='gray')
            center = c1[i]
            plt.scatter([center[0]], [center[1]], color='red', marker='x')
            plt.show()


def get_intensity(image: array, center: Tuple, radii: array) -> array:
    # Open the image and convert to grayscale
    cx, cy = center
    average_intensities = []

    for i in range(len(radii) - 1):
        mask = np.zeros_like(image, dtype=bool)
        bottom, top = radii[i], radii[i+1]
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r2 = (x - cx) ** 2 + (y - cy) ** 2
                if  r2 > bottom**2 and r2 < top**2:
                    mask[y, x] = True

        average_intensity = np.mean(image[mask])
        average_intensities.append(average_intensity)

    return average_intensities


def get_radii(resolution: float, max: float) -> array:
    step = max/resolution
    return array([i*step for i in range(resolution+1)])


def get_folder(path: str) -> array:
    res = []
    for f in os.listdir(path):
        if not f.endswith('.png'): continue
        p = os.path.join(path, f)
        res.append(load_image(p))
    return array(res)



def analyse_1() -> array:
    # path = 'camera_roll'
    # images = get_folder(path)
    resolution = 100
    max = 800
    # intensities = []
    radii = get_radii(resolution=resolution, max=max)
    # for i in range(len(images)):
    #     intensities.append(get_intensity(images[i], c1[i], radii=radii))
    # return array(intensities), radii[1:], d1
    return np.load('intensities1_res100_max_800.npy'), radii[1:], d1-57


def plot_1() -> None:
    intensities, radii, d1 = analyse_1()
    np.save('intensities1_res100_max_800.npy', intensities)
    for i in range(len(intensities)):
        intensity = intensities[i]
        plt.scatter([d1[i] for _ in radii], radii, c=intensity/np.max(intensity), marker='s', s=20, cmap='RdBu')
        plt.plot([0, 69, 140], [155, 60, 210], linestyle='--', color='gray', linewidth=5)
    plt.xlabel('d [cm]', fontsize=20)
    plt.ylabel('r [pixels]', fontsize=20)
    plt.ylim(0, 800)
    plt.xlim(0, 150)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    # path = 'camera_roll\image_2024-07-09T10-25-16.379.png'
    # i = load_image(path)
    # center = (710, 655)
    # resolution = 100
    # max = 800
    # radii = get_radii(resolution, max)
    # intensity = get_intensity(i, center, radii)
    # plt.imshow(i, cmap='gray')
    # plt.show()

    # plt.plot(radii[1:], intensity, 'o')
    # plt.show()


    plot_1()