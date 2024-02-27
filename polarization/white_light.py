import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import colorsys


def display_image(image_path):
    image = Image.open(image_path)
    mark_pixel(image, 50, 50)
    image.show()


def mark_pixel(image, x, y, mark_size=5, mark_color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    draw.line([(x - mark_size, y), (x + mark_size, y)], fill=mark_color, width=2)
    draw.line([(x, y - mark_size), (x, y + mark_size)], fill=mark_color, width=2)



def compute_wavelength(rgb):
    h = compute_h
    L = 650 - 250 / 270 * h
    return L


def compute_h(rgb):
    hsl = colorsys.rgb_to_hls(*rgb)
    h = hsl[0]*360
    return h


if __name__ == "__main__":
    display_image('pretty_photo.jpg')