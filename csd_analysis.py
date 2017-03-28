import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

# set matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')


def csd_analysis():
    plt.ioff()
    root = tk.Tk()
    root.withdraw()

    init_dir = 'F:/NHP'
    file_path = tk.filedialog.askopenfilename(parent=root, initialdir=init_dir, title='Please select the mat file.')

    CSD_info = sio.loadmat(file_path)
    t = CSD_info['t']
    probe_even = CSD_info['probe_even']
    probe_odd = CSD_info['probe_odd']
    CSD_matrix1 = CSD_info['CSD_matrix1']
    CSD_matrix2 = CSD_info['CSD_matrix2']

    odd_chans = np.array([24, 23, 25, 22, 26, 21, 27, 20, 28, 19, 29, 18, 30, 17, 31, 16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11,
                          5, 10, 6, 9, 7, 8, 56, 55, 57, 54, 58, 53, 59, 52, 60, 51, 61, 50, 62, 49, 63, 48, 32, 47, 33, 46,
                          34, 45, 35, 44, 36, 43, 37, 42, 38, 41, 39, 40])
    even_chans = np.array([88, 87, 89, 86, 90, 85, 91, 84, 92, 83, 93, 82, 94, 81, 95, 80, 64, 79, 65, 78, 66, 77, 67, 76,
                           68, 75, 69, 74, 73, 70, 72, 71, 120, 119, 121, 118, 122, 117, 123, 116, 115, 124, 114, 125, 113,
                           126, 112, 127, 111, 96, 110, 97, 109, 98, 108, 99, 107, 100, 106, 101, 105, 102, 104, 103])

    # average first 25ms to remove bad channels
    prestim_csd1 = np.mean(CSD_matrix1[:, 0:25], axis=1)
    prestim_csd2 = np.mean(CSD_matrix2[:, 0:25], axis=1)

    norm_csd1 = (CSD_matrix1.transpose() - prestim_csd1).transpose()
    norm_csd2 = (CSD_matrix2.transpose() - prestim_csd2).transpose()

    # plot normalized CSDs
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.set_cmap('jet_r')
    ax1.imshow(norm_csd1[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto')
    ax2.imshow(norm_csd2[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto')

    ax1.set_xlabel('Time (ms)')
    ax2.set_xlabel('Time (ms)')

    ax1.set_ylabel('Channel (even)')
    ax2.set_ylabel('Channel (odd)')

    plt.savefig(os.path.dirname(file_path) + '/csd_image.pdf', format='pdf')
    plt.close()

    csd_avg = np.zeros([64, 401])
    for i in np.arange(0, 64):
        csd_row = np.zeros([2, 401])

        try:
            idx1 = np.where((probe_even[0] - 1) == i)
            csd_row[0, :] = norm_csd1[idx1, :]
        except ValueError:
            csd_row[0, :] = np.nan

        try:
            idx2 = np.where((probe_odd[0] - 1) == i)
            csd_row[1, :] = norm_csd2[idx2, :]
        except ValueError:
            csd_row[1, :] = np.nan

        csd_row_avg = np.nanmean(csd_row, axis=0)
        csd_avg[i, :] = csd_row_avg

    fig2, ax = plt.subplots(1)
    plt.set_cmap('jet_r')
    ax.imshow(csd_avg[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Channel (odd)')

    plt.savefig(os.path.dirname(file_path) + '/csd_image_avg.pdf', format='pdf')
    plt.close()

    return csd_avg
