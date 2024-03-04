from os import listdir, path, mkdir
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from scipy.signal import butter, filtfilt
import csv


# Get file names in 'path' folder
def get_ordered_files(path_value: str) -> list[str]:
    files = [
        f'{path_value}{f}' for f in listdir(path_value)
        if path.isfile(path.join(path_value, f)) and f != 'sum_image.png'
    ]
    return sorted(files, key=lambda x: int(x.split('.')[1]))


# Get sum image
def get_sum_img(image_paths: list[str], start: int, size: int) -> np.array:
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


# Butterworth filter
def lowpass_filter_but(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


STEP = 16

if __name__ == '__main__':
    args = sys.argv[1:]
    # dir = args[0]
    dir = path.normpath("static/ec02a37a-2134-430f-944e-37bdc1a79d8f")
    LOGS_DIR = path.join(path.dirname(dir), 'logs_' + path.basename(dir))
    if path.exists(LOGS_DIR) == False:
        mkdir(LOGS_DIR)
    IMG_FILE_NAME = path.join(LOGS_DIR, 'img_' + path.basename(dir) + '.png')
    PLOT_FILE_NAME = path.join(LOGS_DIR, 'plot_' + path.basename(dir) + '.png')
    plt.figure(figsize=(16, 9))

    ordered_image_path = get_ordered_files(path.join(dir, ''))
    print(f'Loaded {len(ordered_image_path)} images')

    file_len = len(ordered_image_path)
    length = file_len
    sum_img = get_sum_img(ordered_image_path, 0, length)
    np.savetxt(path.join(LOGS_DIR, 'sum_image_array.csv'),
               sum_img,
               fmt='%d',
               delimiter=',')
    norm_img = cv2.normalize(sum_img,
                             None,
                             alpha=0,
                             beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
    print(cv2.imwrite(IMG_FILE_NAME, norm_img))
    np.savetxt(path.join(LOGS_DIR, 'norm_image_array.csv'),
               norm_img,
               fmt='%d',
               delimiter=',')
    # plt.imshow(norm_img)

    metric_file = open(path.join(LOGS_DIR, 'metrics.csv'),
                       mode='w',
                       newline='')
    writer = csv.writer(metric_file)
    writer.writerow(
        ['Iteration', 'RMS noise level', 'Mean of Signal', 'SNR_dB', 'DNR_dB'])
    file_number = len(ordered_image_path)
    spectrum_array = None
    step = STEP
    iteration = 0
    for i in range(0, len(ordered_image_path) - step, step):
        iteration += 1
        spectrum = get_spectrum(
            get_sum_img(
                ordered_image_path, i,
                STEP if i + STEP < file_number else file_number - STEP + 1))
        plt.plot(spectrum, label=f'Iteration {iteration}')
        filtered_spectrum = lowpass_filter_but(spectrum,
                                               cutoff_freq=12,
                                               fs=500)
        plt.plot(filtered_spectrum)
        RMS_noise = np.sqrt(np.mean((filtered_spectrum - spectrum)**2))
        signal_mean = np.mean(spectrum)
        SNR_dB = 10 * np.log10(np.mean(spectrum**2) / (RMS_noise**2))
        DNR_dB = 10 * np.log10((np.max(spectrum)**2) / (RMS_noise**2))
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

    with open(path.join(LOGS_DIR, 'spectrum_array.csv'), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Iteration1", "Iteration2", "Iteration3", "Iteration4",
            "Iteration5", "Iteration6"
        ])
        writer.writerows(spectrum_array)

    vertical_lines_x = [convert_nm_to_pix(i) for i in range(400, 750, 50)]
    y_lim = plt.gca().get_ylim()
    plt.vlines(vertical_lines_x,
               ymin=y_lim[0],
               ymax=y_lim[1],
               color='gray',
               linestyle='--')

    plt.legend(loc='best')
    plt.savefig(PLOT_FILE_NAME)
    # plt.show()
