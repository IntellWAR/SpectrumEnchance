# Copyright 2024 Korneev Pavel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import listdir, path, mkdir
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
import cv2
import sys
from scipy.signal import butter, filtfilt
import csv
import spectrum_by_curves
import re

dir_ref = path.normpath("static/512c38b9-090b-46fa-a32c-2fb3d5939ace")
dir = path.normpath("static/d5ad0963-d44e-4f02-891c-bb1fd482fc5f")

is_filtered = True  # Enable showing of the filtered signal in plot
is_nm_given = False  # If borders given in pixels then False
is_xscale_nm = True  # Enable scale in nanometers
right_limit_pix = 750
left_limit_pix = 5
# right_limit_nm = 720
# left_limit_nm = 445
STEP = 16  # Files in one iteration
is_parabolic_compensation = False  # Enable calculating the spectrum by parabolic curves


# right_limit_nm = 720
# left_limit_nm = 445

# Converting nanometers to pixel value
def convert_nm_to_pix(nm_value: float) -> float:
    return (158 / 59) * nm_value - 1178.146


def convert_pix_to_nm(pix_value: float) -> float:
    return (59 / 158) * (pix_value + 1178.146)


# Convert borders
if is_xscale_nm:
    if is_nm_given:
        right_limit_pix = round(convert_nm_to_pix(right_limit_nm))
        left_limit_pix = round(convert_nm_to_pix(left_limit_nm))
    else:
        right_limit_nm = round(convert_pix_to_nm(right_limit_pix))
        left_limit_nm = round(convert_pix_to_nm(left_limit_pix))
    x_values = convert_pix_to_nm(np.arange(left_limit_pix, right_limit_pix))
else:
    if is_nm_given:
        right_limit_pix = round(convert_nm_to_pix(right_limit_nm))
        left_limit_pix = round(convert_nm_to_pix(left_limit_nm))
    else:
        right_limit_nm = round(convert_pix_to_nm(right_limit_pix))
        left_limit_nm = round(convert_pix_to_nm(left_limit_pix))
    x_values = np.arange(left_limit_pix, right_limit_pix)

# create directories
LOGS_DIR = path.join(path.dirname(dir), 'logs_' + path.basename(dir))
if path.exists(LOGS_DIR) == False:
    mkdir(LOGS_DIR)
IMG_FILE_NAME = path.join(LOGS_DIR, 'img_reffered_' + path.basename(dir) + '.png')
PLOT_FILE_NAME_eps = path.join(LOGS_DIR, 'plot_reffered_' + path.basename(dir) + '.eps')
PLOT_FILE_NAME_png = path.join(LOGS_DIR, 'plot_reffered_' + path.basename(dir) + '.png')
WAVELEN_PLOT_NAME = path.join(LOGS_DIR, 'plot_in_wavelen_' + path.basename(dir) + '.eps')
plt.figure(figsize=(16, 9))
plt.rc('font', family='serif', size=16)


# Extracting number from filename
def extract_number(filename):
    # Регулярное выражение для извлечения числа после имени и перед '.png' в названии файла
    pattern = r'\.(\d+)\.'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Возвращаем бесконечность, если число не найдено, чтобы такие файлы оказались в конце

# Get file names in 'path' folder
def get_ordered_files(path_value: str) -> list[str]:
    files = [f'{path_value}{f}' for f in listdir(path_value) if
             path.isfile(path.join(path_value, f)) and f != 'sum_image.png']
    return sorted(files, key=lambda x: int(extract_number(x)))


# Get sum image
def get_sum_img(image_paths: list[str], start: int, size: int) -> np.array:
    a = cv2.imread(image_paths[6])
    return np.array(reduce(lambda x, y: x + y,
                           [cv2.imread(image_paths[i])[:, :, 0].astype('uint64') for i in range(start, start + size)]),
                    dtype='uint64')


# Get spectrum by pixel integrating image
def get_spectrum(image: np.array) -> np.array:
    return np.array([np.average(image[:, i]) for i in range(image.shape[1])], dtype='uint64')


# Butterworth filter
def lowpass_filter_but(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def normalize_array(array):
    max = np.max(array)
    if max == 0:
        return array
    return array / max


def get_ref_spectrum(dir: str = dir_ref) -> np.array:
    ordered_image_path = get_ordered_files(path.join(dir, ''))
    print(f'Loaded {len(ordered_image_path)} refference images')
    file_len = len(ordered_image_path)
    if is_parabolic_compensation:
        ref_spectrum = spectrum_by_curves.return_spectr(get_sum_img(ordered_image_path, 0, file_len))
    else:
        ref_spectrum = get_spectrum(get_sum_img(ordered_image_path, 0, file_len))
    ref_spectrum_norm = normalize_array(ref_spectrum[left_limit_pix:right_limit_pix])
    np.savetxt(path.join(LOGS_DIR, 'ref_array.csv'), ref_spectrum_norm, fmt='%d', delimiter=',')
    # plt.plot(ref_spectrum)
    # plt.show()
    return ref_spectrum_norm


if __name__ == '__main__':
    args = sys.argv[1:]
    # dir = args[0]

    ordered_image_path = get_ordered_files(path.join(dir, ''))
    print(f'Loaded {len(ordered_image_path)} images')
    metric_file = open(path.join(LOGS_DIR, 'metrics_reffered.csv'), mode='w', newline='')
    writer = csv.writer(metric_file)
    writer.writerow(['Iteration', 'RMS noise level', 'Mean of Signal', 'SNR_dB', 'DNR_dB'])

    file_number = len(ordered_image_path)
    spectrum_array = None
    ref_spectrum = get_ref_spectrum()
    step = STEP
    colors = plt.cm.tab10(np.linspace(0, 1, len(ordered_image_path) // STEP))  # Массив цветов для графика
    iteration = 0
    for i in range(0, file_number, step):
        iteration += 1
        if is_parabolic_compensation:
            spectrum = spectrum_by_curves.return_spectr(
                get_sum_img(ordered_image_path, i, STEP if i + STEP <= file_number else file_number - i))
        else:
            spectrum = get_spectrum(
                get_sum_img(ordered_image_path, i, STEP if i + STEP <= file_number else file_number - i))
        spectrum_norm = normalize_array(spectrum[left_limit_pix:right_limit_pix])
        spectrum_with_ref = spectrum_norm / ref_spectrum
        spectrum_with_ref_norm = spectrum_with_ref
        color = colors[iteration % len(colors)]  # Выбор цвета для графика
        plt.plot(x_values, spectrum_with_ref_norm, label=f'Iteration {iteration}', color=color)
        filtered_spectrum = lowpass_filter_but(spectrum_with_ref_norm, cutoff_freq=12, fs=500)
        if is_filtered:
            plt.plot(x_values, filtered_spectrum, color=color)
        RMS_noise = np.sqrt(np.mean((filtered_spectrum - spectrum_with_ref_norm) ** 2))
        signal_mean = np.mean(spectrum_with_ref_norm)
        SNR_dB = 10 * np.log10(np.mean(spectrum_with_ref_norm ** 2) / (RMS_noise ** 2))
        DNR_dB = 10 * np.log10((np.max(spectrum_with_ref_norm) ** 2) / (RMS_noise ** 2))
        print('Iteration {}, RMS noise level = {:.4f}, Mean of signal = {:.2f} SNR_dB = {:.2f}, DNR_dB = {:.2f}'.format(
            iteration, RMS_noise, signal_mean, SNR_dB, DNR_dB), end='\n')

        writer.writerow([iteration, RMS_noise, signal_mean, SNR_dB, DNR_dB])
        if spectrum_array is None:
            spectrum_array = spectrum_norm.reshape(-1, 1)
        else:
            spectrum_array = np.hstack((spectrum_array, spectrum_norm.reshape(-1, 1)))
    metric_file.close()

    with open(path.join(LOGS_DIR, 'spectrum_array_reffered.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Iteration1", "Iteration2", "Iteration3", "Iteration4", "Iteration5", "Iteration6"])
        writer.writerows(spectrum_array)

    # Plot drawing
    # plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.yscale('log')
    if is_xscale_nm:
        plt.xlabel('длина волны, нм')
    else:
        plt.xlabel('горизонтальная координата матрицы, пиксели')
    plt.ylabel('отношение интенсивностей (исследуемый/опорный)')
    plt.minorticks_on()
    plt.grid(which='major', linewidth='1.5')
    plt.grid(which='minor', linewidth='0.5')
    plt.savefig(PLOT_FILE_NAME_png, format='png')
    plt.savefig(PLOT_FILE_NAME_eps, format='eps')
