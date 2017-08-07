import glob
import numpy as np
import scipy as sp
from scipy import signal
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'F:/NHP/AD8/Ephys/20161205/hartley_gray'
trials_path = glob.glob(data_folder + '/trials.csv')[0]

trials = pd.read_csv(trials_path, index_col=0)

stim_count = np.zeros([np.size(np.unique(trials.sf_x)), np.size(np.unique(trials.sf_y))])
for i, sfx in enumerate(np.unique(trials.sf_x)):
    for j, sfy in enumerate(np.unique(trials.sf_y)):
        stim_count[i, j] = stim_count[i, j] + trials[(trials.sf_x == sfx) &
                                                     (trials.sf_y == sfy)].shape[0]

samp_ID = np.arange(trials.stim_sample[0], trials['stim_sample'][trials.index[-1]] + int(1000 * 0.02))
n = int(np.floor(np.size(samp_ID) * 1000 / 25000))
stim_samp = np.floor(np.append(trials['stim_sample'].as_matrix(),
                               trials['stim_sample'][trials.index[-1]] + 469) / 25).astype(int) - 1

stim_seq = np.zeros([stim_samp[-1], 2])
sf = trials[['sf_x', 'sf_y']].as_matrix()
sfx_unique = np.unique(trials.sf_x)
sfy_unique = np.unique(trials.sf_y)

for i in range(0, trials.shape[0]):
    x = np.where(sfx_unique == sf[i, 0])[0]
    y = np.where(sfy_unique == sf[i, 1])[0]

    stim_seq[stim_samp[i]:stim_samp[i+1], :] = np.vstack((x*np.ones(np.size(np.arange(stim_samp[i], stim_samp[i+1]))),
                                                         y*np.ones(np.size(np.arange(stim_samp[i], stim_samp[i+1]))))).T

stim_seq[np.where((stim_seq[:, 0] == 0) & (stim_seq[:, 1] == 0)), :] = [30, 1]
stim_start = np.where((stim_seq[:, 0] != 30) & (stim_seq[:, 1] != 2))[0][0]

x = {}
r = {}

for cell in range(1, 74):
    st = sio.loadmat("C:\\Users\\anupam\\Dropbox\\NHP cells\\spike_times_%d.mat" % cell)['spike_times']
    revcorr_images_f = sio.loadmat("C:\\Users\\anupam\\Dropbox\\NHP cells\\revcorr_images_f_%d.mat" % cell)['revcorr_images_f']

    kern = np.zeros(np.shape(revcorr_images_f))
    for i in range(revcorr_images_f.shape[2]):
        kern[:, :, i] = revcorr_images_f[:, :, i] / stim_count

    kern = np.nan_to_num(kern)

    x_full = np.zeros([np.shape(stim_seq)[0] - stim_start])
    for i in range(stim_start, np.shape(stim_seq)[0]):
        sfx = stim_seq[i-199:i+1, 0]
        sfy = stim_seq[i-199:i+1, 1]

        x_full[i-stim_start] = np.sum(kern[sfx.astype(int), sfy.astype(int), range(0, 200)])
        # for j in range(0, 200):
        #     x_full[i-stim_start] = x_full[i-stim_start] + kern[int(sfx[j]), int(sfy[j]), j]

    r_t = np.zeros(np.size(x_full))
    st_shift = st - trials.stim_sample[0] / 25000
    width = 25

    for i in range(int(np.ceil(width/2)), np.size(x_full)):
        bin_l = (i-np.floor(width/2)) / 1000
        bin_r = (i+np.floor(width/2)) / 1000

        r_t[i] = np.size(np.where((st_shift > bin_l) & (st_shift < bin_r))[0])

    corr = sp.signal.fftconvolve(r_t, x_full[::-1], 'full')
    lags = np.arange(-int(np.size(corr) / 2), int(np.size(corr) / 2 + 1))

    delay = lags[(corr == np.max(corr))] - 1
    x[cell] = x_full[0:-delay]
    r[cell] = r_t[delay:]

    bin_width = np.ptp(x[cell]) / 20
    x_bins = np.arange(np.min(x[cell]) + (bin_width / 2), np.max(x[cell]) - (bin_width / 2) + 0.01, bin_width)

    r_bins = np.zeros(np.size(x_bins))
    for bin in np.arange(0, np.size(x_bins)):
        r_bins[bin] = np.mean(r_t[((x[cell] < x_bins[bin] + bin_width / 2) & (x[cell] > x_bins[bin] - bin_width / 2))])

    plt.ioff()
    fig, ax = plt.subplots()
    ax.scatter(x_bins, r_bins)
    ax.set_xlabel('Filtered Stimulus')
    ax.set_ylabel('Response')
    plt.savefig('C:\\Users\\anupam\\Dropbox\\NHP cells\\response_cell%d' % cell)
    plt.close()

plt.ion()
