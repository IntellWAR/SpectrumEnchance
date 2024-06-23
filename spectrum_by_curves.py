# The original code by Danila Rudenko from the link https://colab.research.google.com/drive/1WhzrmbBund1sf5BRzkkmg4U4EKasRuKt?usp=sharing

import numpy as np
import typing
import math
from PIL import Image
import matplotlib.pyplot as plt


haracteristics: typing.Mapping[str, typing.Any] = {}
haracteristics ['len_abscissa'] = 1280
haracteristics ['len_ordinata'] = 1024
haracteristics ['gener_spectr_ordinata'] = range(0, 1024, 1)
param: typing.Tuple[float, float, float, float] = [0.15, 0.15, 55, 55]
# param: typing.Tuple[float, float, float, float] = [0.0, 0.0, 0, 0]


def get_color_not_integer_pixels(point: typing.Tuple[float, float], image: typing.List[typing.List[int]]) -> float: # выдаёт цвет не в целых пикселях
    def _get_color_integer_pixels(new_point: typing.Tuple[float, float], image: typing.List[typing.List[int]]) -> float: # выдаёт цвет в целых пикселях
        abscissa = new_point[0]
        ordinata = new_point[1]

        if abscissa < 0:
            return 0
        if ordinata < 0:
            return 0
        if abscissa >= len(image[0]):
            return 0
        if ordinata >= len(image):
            return 0
        return image[ordinata][abscissa]

    abscissa = math.floor(point[0])
    ordinata = math.floor(point[1])
    fractional_abscissa = point[0] - abscissa
    fractional_ordinata = point[1] - ordinata

    baza = _get_color_integer_pixels((abscissa, ordinata), image)
    baza_a = _get_color_integer_pixels((abscissa + 1, ordinata), image)
    baza_o = _get_color_integer_pixels((abscissa, ordinata  + 1), image)
    baza_o_a = _get_color_integer_pixels((abscissa + 1, ordinata + 1), image)
    result = baza_o_a * (fractional_abscissa) * (fractional_ordinata)
    result += baza_o * (1 - fractional_abscissa) * (fractional_ordinata)
    result += baza_a * (fractional_abscissa) * (1 - fractional_ordinata)
    result += baza * (1 - fractional_abscissa) * (1 - fractional_ordinata)
    return result


def get_param_from_relative_abscissa(index_param: int, relative_abscissa: float) -> float:
    return param[index_param] * (1 - relative_abscissa) + param[index_param + 1] * (relative_abscissa)


def get_new_abscissa(abscissa: float, ordinata: float) -> float:
    delata_relative_ordinata: float = ordinata / haracteristics ['len_ordinata'] * 2 - 1
    relative_abscissa: float = abscissa / haracteristics ['len_abscissa']
    shift = get_param_from_relative_abscissa(0, relative_abscissa) # сдвиг вершины параболы
    kappa = get_param_from_relative_abscissa(2, relative_abscissa) # кривизна параболы
    return abscissa + kappa * pow(delata_relative_ordinata - shift, 2) - kappa * pow(shift, 2)


def get_new_pointer(abscissa: float, ordinata: float) -> tuple[float, float]: # дописать что возврвщвется
    new_abscissa = get_new_abscissa(abscissa, ordinata)
    return tuple([new_abscissa, ordinata])


def get_one_line(abscissa: float, image: typing.List[typing.List[int]]) -> float:
      summ = 0
      for ordinate in haracteristics['gener_spectr_ordinata']:
        new_point = get_new_pointer(abscissa, ordinate)
        summ += get_color_not_integer_pixels(new_point, image)
      return summ


def get_spectr(gener_abscissa: typing.Iterator[float], images: typing.List[typing.List[typing.List[int]]]) -> tuple[float]:
    list_result = []
    for abscissa in gener_abscissa:
        summ = 0
        for image in images:
            summ += get_one_line(abscissa, image)
        list_result.append(summ)
    return tuple(list_result)

def return_grey_photo(list_photo_rgb) -> list[list[int]]:
    list_photo_grey = []
    for row in list_photo_rgb:
        row_grey = []
        for pixel in row:
            row_grey.append(int(pixel[1]))
        list_photo_grey.append(row_grey)
    return list_photo_grey


def return_spectr(image: np.array, left_limit_pix: int = 0, right_limit_pix: int = 1280) -> np.array:
    gener_spectr = np.arange(left_limit_pix, right_limit_pix, 1)  # от какого до какого пикселя делать спектр
    # image_rgb = np.array(Image.open('result.png').convert('RGB'))  # грузис изображение
    # image_grey = return_grey_photo(image_rgb)  # переводишь из RGB в rgey
    spectr = get_spectr(gener_spectr, images = [image]) # вычислить спектр
    spectr_np = np.array(spectr)
    return spectr_np

# gener_spectr = np.arange(0, 1280, 1)  # от какого до какого пикселя делать спектр
# image_rgb = np.array(Image.open('result.png').convert('RGB'))  # грузис изображение
# image_grey = return_grey_photo(image_rgb)  # переводишь из RGB в rgey
# spectr = get_spectr(gener_spectr, images = [image_grey]) # вычислить спектр
# print(np.array(spectr).shape)
# plt.figure(figsize=(16, 9))
# plt.plot(gener_spectr, spectr, 'g')
# plt.grid(True) #Включем координатную сетку
# plt.show()