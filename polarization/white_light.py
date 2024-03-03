import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import colorsys
import os
import re



def display_image(image_path):
    image = Image.open(image_path)
    dpi = (300, 300)
    image.save("temp_image.png", dpi=dpi)
    image = Image.open("temp_image.png")
    image.show()


def display_with_cross(image_path, pixel):
    image = Image.open(image_path)
    dpi = (300, 300)
    image.save("temp_image.png", dpi=dpi)
    image = Image.open("temp_image.png")
    mark_pixel(image, *pixel)
    image.show()


def mark_pixel(image, x, y, mark_size=80, mark_color=(255, 0, 0), mark_thickness=10):
    draw = ImageDraw.Draw(image)
    draw.line([(x - mark_size, y), (x + mark_size, y)], fill=mark_color, width=mark_thickness)
    draw.line([(x, y - mark_size), (x, y + mark_size)], fill=mark_color, width=mark_thickness)


def get_phone_rgb():
    rgb = []
    angle = []
    folder_path = 'white_light/color_vs_angle/0-120_20cm_phone'
    phone_pixel = (1900, 2000)
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_number = int(re.search(r'\d+', file_name).group(), 10)
            photo_angle = (file_number-218)*10
            angle.append(photo_angle)
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            photo_rgb = get_average_rgb(image, *phone_pixel)
            rgb.append(photo_rgb)
            print(photo_angle, photo_rgb)
    return np.array(angle), np.array(rgb)


def get_lamp_rgb():
    rgb = []
    angle = []
    folder_path = 'white_light/color_vs_angle/0-290_20cm_lamp'
    lamp_pixel = (3050, 2100)
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_number = int(re.search(r'\d+', file_name).group(), 10)
            photo_angle = (file_number-127)*10
            angle.append(photo_angle)
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            photo_rgb = get_average_rgb(image, *lamp_pixel)
            photo_rgb = normalize_rgb(photo_rgb)
            rgb.append(photo_rgb)
            print(photo_angle, photo_rgb)
    return np.array(angle), np.array(rgb)


def get_length_rgb():
    rgb = []
    length = []
    folder_path = r"C:\Users\TLP-312\lab2\polarization\white_light\color_vs_length\20-9cm_60deg_no14"
    lamp_pixel = (2400, 1950)
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_number = int(re.search(r'\d+', file_name).group(), 10)
            photo_length = 20 - file_number + 241
            if photo_length < 15: photo_length -= 1
            length.append(photo_length)
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            photo_rgb = get_average_rgb(image, *lamp_pixel)
            photo_rgb = normalize_rgb(photo_rgb)
            rgb.append(photo_rgb)
            print(photo_length, photo_rgb)
    return np.array(length), np.array(rgb)


def plot_phone():
    angle, rgb = get_phone_rgb()
    wavelength = np.array([compute_wavelength(color) for color in rgb])
    plt.errorbar(angle, wavelength, fmt='.')
    plt.xlim(25, 75)
    plt.grid()
    plt.xlabel('$\\theta[^\circ]$')
    plt.ylabel('$\lambda[nm]$')
    plt.show()


def normalize_rgb(rgb):
    r, g, b = rgb
    return (r/256, g/256, b/256)


def get_average_rgb(image, x, y, radius=5):
    """
    Get the average RGB values in a square region around the specified pixel.
    :param image: PIL Image object
    :param x: X-coordinate of the center pixel
    :param y: Y-coordinate of the center pixel
    :param radius: Radius of the square region
    :return: Average RGB values as a tuple
    """
    width, height = image.size

    # Define the region of interest
    x_start = max(0, x - radius)
    y_start = max(0, y - radius)
    x_end = min(width - 1, x + radius)
    y_end = min(height - 1, y + radius)

    # Collect RGB values in the region
    rgb_values = []
    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            rgb_values.append(image.getpixel((i, j)))

    # Calculate the average RGB values
    average_rgb = (
        sum([rgb[0] for rgb in rgb_values]) // len(rgb_values),
        sum([rgb[1] for rgb in rgb_values]) // len(rgb_values),
        sum([rgb[2] for rgb in rgb_values]) // len(rgb_values)
    )

    return average_rgb


def compute_wavelength(rgb):
    h = compute_h(rgb)
    L = 650 - 250 / 270 * h
    return L


def compute_h(rgb):
    hsl = colorsys.rgb_to_hls(*rgb)
    h = hsl[0]*360
    print(h)
    return h


def display_folder(folder_path):
    pixel = (1900, 2000)
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(folder_path, file_name)
            display_with_cross(image_path, pixel)


def plot_lamp_rgb():
    angle, rgb = get_lamp_rgb()
    angle = angle % 180
    theta_err = [2 for a in angle]
    rgb_err = [1/256 for a in angle]
    rgb = np.array(list(zip(*rgb)))
    colors = ['red', 'green', 'blue']
    names = ['R', 'G', 'B']
    for i in range(3):
        plt.errorbar(angle, rgb[i], xerr=theta_err, yerr=rgb_err, color=colors[i], fmt='--', label=names[i])
        plt.xlabel('$\\theta[^\circ]$')
        plt.ylabel('normalized RGB')
    plt.grid()
    plt.legend()
    plt.show()


def plot_length_rgb():
    angle, rgb = get_length_rgb()
    angle = angle % 180
    theta_err = [0.2 for a in angle]
    rgb_err = [1/256 for a in angle]
    rgb = np.array(list(zip(*rgb)))
    colors = ['red', 'green', 'blue']
    names = ['R', 'G', 'B']
    for i in range(3):
        plt.errorbar(angle, rgb[i], xerr=theta_err, yerr=rgb_err, color=colors[i], fmt='--', label=names[i])
        plt.xlabel('$l[cm]$')
        plt.ylabel('normalized RGB')
    plt.grid()
    plt.legend()
    plt.show()


def plot_chi():
    xlim = (420, 675)
    ylim = (0, 2.3)
    green_wavelength = 535
    blue_wavelength = 460
    orange_wavelength = 605
    lamda = [543, 594, 633]
    chi = [1.422, 1.35, 0.932]
    chi_err = [0.0003, 0.01, 0.001]
    plt.errorbar(lamda, chi, yerr=chi_err, fmt='.', label='measured $\chi$', markersize=10)
    plt.plot([green_wavelength, green_wavelength], ylim, color='green', linestyle='--', label='green area')
    plt.plot([blue_wavelength, blue_wavelength], ylim, color='blue', linestyle='--', label='blue area')
    plt.plot([orange_wavelength, orange_wavelength], ylim, color='orange', linestyle='--', label='orange area')
    def model(x):
        height = 0.3
        cutoff = 610
        smudge = 20
        return height*(1-np.tanh((x-cutoff)/smudge))
    def bump(x):
        height = 0.35
        cutoff = 460
        smudge = 20
        return height*(1-np.tanh((x-cutoff)/smudge))
    const = 0.85
    x_fit = np.linspace(*xlim, 1000)
    y_fit = np.array([model(x)+bump(x)+const for x in x_fit])
    plt.plot(x_fit, y_fit, label='hypothesis')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.grid()
    plt.xlabel('$\lambda[nm]$')
    plt.ylabel('$\chi[^\circ/cm]$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # display_folder('white_light/color_vs_angle/0-120_20cm_phone')
    # display_image('white_light/color_vs_angle/0-120_20cm_phone/DSC_0218.jpg')
    # display_image('DSC_0224.jpg')
    # display_image('pretty_photo.jpg')
    # plot_phone()
    # display_folder('white_light/color_vs_angle/0-120_20cm_phone')
    # get_lamp_rgb()
    # display_with_cross(r"C:\Users\TLP-312\lab2\polarization\white_light\color_vs_length\20-9cm_60deg_no14\DSC_0241.jpg", (2400, 1950))
    # plot_lamp_rgb()
    # plot_length_rgb()
    plot_chi()
