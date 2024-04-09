from os import listdir, path, mkdir
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
import cv2
import sys
from scipy.signal import butter, filtfilt
import csv

dir_ref = path.normpath("static/5dead71c-9c63-4944-b54c-88489922066a")
dir = path.normpath("static/ec02a37a-2134-430f-944e-37bdc1a79d8f")

STEP = 16

LOGS_DIR = path.join(path.dirname(dir), 'logs_' + path.basename(dir))
if path.exists(LOGS_DIR) == False:
    mkdir(LOGS_DIR)
IMG_FILE_NAME = path.join(LOGS_DIR,
                          'img_reffered_' + path.basename(dir) + '.png')
PLOT_FILE_NAME_eps = path.join(LOGS_DIR,
                               'plot_reffered_' + path.basename(dir) + '.eps')
PLOT_FILE_NAME_png = path.join(LOGS_DIR,
                               'plot_reffered_' + path.basename(dir) + '.png')
WAVELEN_PLOT_NAME = path.join(LOGS_DIR,
                              'plot_in_wavelen_' + path.basename(dir) + '.eps')
plt.figure(figsize=(16, 9))
plt.rc('font', family='serif', size=16)


# Get file names in 'path' folder
def get_ordered_files(path_value: str) -> list[str]:
    files = [
        f'{path_value}{f}' for f in listdir(path_value)
        if path.isfile(path.join(path_value, f)) and f != 'sum_image.png'
    ]
    return sorted(files, key=lambda x: int(x.split('.')[1]))


# Get sum image
def get_sum_img(image_paths: list[str], start: int, size: int) -> np.array:
    a = cv2.imread(image_paths[6])
    return np.array(reduce(lambda x, y: x + y, [
        cv2.imread(image_paths[i])[:, :, 0].astype('uint64')
        for i in range(start, start + size)
    ]),
                    dtype='uint64')


# Get spectrum by pixel integrating image
def get_spectrum(image: np.array) -> np.array:
    return np.array([np.average(image[:, i]) for i in range(image.shape[1])],
                    dtype='uint64')


# Converting nanometers to pixel value
def convert_nm_to_pix(nm_value: float) -> float:
    return (158 / 59) * nm_value - 1178.146


def convert_pix_to_nm(pix_value: float) -> float:
    return (59 / 158) * (pix_value + 1178.146)


# Butterworth filter
def lowpass_filter_but(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_ref_spectrum(dir: str = dir_ref) -> np.array:
    ordered_image_path = get_ordered_files(path.join(dir, ''))
    print(f'Loaded {len(ordered_image_path)} refference images')
    file_len = len(ordered_image_path)
    ref_spectrum = get_spectrum(get_sum_img(ordered_image_path, 0,
                                            file_len)) / (file_len / STEP)
    np.savetxt(path.join(LOGS_DIR, 'ref_array.csv'),
               ref_spectrum,
               fmt='%d',
               delimiter=',')
    # plt.plot(ref_spectrum)
    # plt.show()
    return ref_spectrum


if __name__ == '__main__':
    args = sys.argv[1:]
    # dir = args[0]

    ordered_image_path = get_ordered_files(path.join(dir, ''))
    print(f'Loaded {len(ordered_image_path)} images')
    metric_file = open(path.join(LOGS_DIR, 'metrics_reffered.csv'),
                       mode='w',
                       newline='')
    writer = csv.writer(metric_file)
    writer.writerow(
        ['Iteration', 'RMS noise level', 'Mean of Signal', 'SNR_dB', 'DNR_dB'])

    file_number = len(ordered_image_path)
    spectrum_array = None
    ref_spectrum = get_ref_spectrum()
    x_values = convert_pix_to_nm(np.arange(0, 1280))
    step = STEP
    iteration = 0
    for i in range(0, len(ordered_image_path) - step, step):
        iteration += 1
        spectrum = get_spectrum(
            get_sum_img(
                ordered_image_path, i,
                STEP if i + STEP < file_number else file_number - STEP + 1))
        koef = np.mean(ref_spectrum) / np.mean(spectrum)
        # normalised_spectrum = spectrum * koef - ref_spectrum
        normalised_spectrum = spectrum - ref_spectrum
        plt.plot(x_values, normalised_spectrum, label=f'Iteration {iteration}')
        filtered_spectrum = lowpass_filter_but(normalised_spectrum,
                                               cutoff_freq=12,
                                               fs=500)
        plt.plot(x_values, filtered_spectrum)
        RMS_noise = np.sqrt(
            np.mean((filtered_spectrum - normalised_spectrum)**2))
        signal_mean = np.mean(normalised_spectrum)
        SNR_dB = 10 * np.log10(
            np.mean(normalised_spectrum**2) / (RMS_noise**2))
        DNR_dB = 10 * np.log10(
            (np.max(normalised_spectrum)**2) / (RMS_noise**2))
        print(
            'Iteration {}, RMS noise level = {:.2f}, Mean of signal = {:.2f} SNR_dB = {:.2f}, DNR_dB = {:.2f}'
            .format(iteration, RMS_noise, signal_mean, SNR_dB, DNR_dB),
            end='\n')

        writer.writerow([iteration, RMS_noise, signal_mean, SNR_dB, DNR_dB])
        if spectrum_array is None:
            spectrum_array = spectrum.reshape(-1, 1)
        else:
            spectrum_array = np.hstack(
                (spectrum_array, spectrum.reshape(-1, 1)))
    metric_file.close()

    with open(path.join(LOGS_DIR, 'spectrum_array_reffered.csv'),
              'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Iteration1", "Iteration2", "Iteration3", "Iteration4",
            "Iteration5", "Iteration6"
        ])
        writer.writerows(spectrum_array)

    # plt.ylim(-50, 50)
    # plt.gca().spines['bottom'].set_position('zero')
    plt.legend(loc='best')
    plt.xticks(ticks=np.arange(400, 1000, 50))
    plt.savefig(PLOT_FILE_NAME_png, format='png')
    plt.savefig(PLOT_FILE_NAME_eps, format='eps')
