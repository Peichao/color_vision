import os
import numpy as np
import pandas as pd
from detect_peaks import detect_peaks


def extract_sync_samples(file_path):
    num_chans = 385

    lfp_mm = (np.memmap(file_path, np.int16, mode='r').reshape(-1, num_chans)).T
    sync_data = lfp_mm[num_chans - 1, :]

    # sync_data_df = pd.DataFrame(sync_data, columns=['sync_data'])
    # sync_data_df['binary'] = sync_data_df.sync_data.apply(lambda x: format(x, '04b'))
    # sync_data_df['binary'] = sync_data_df.sync_data.
    # sync_data_bits = np.fromstring(sync_data_df.binary.values.astype(str), dtype=int).reshape(-1, 16) - 48

    return sync_data


def extract_nidq_samples(file_path):
    num_chans = 2
    chans = ['photodiode', 'epoc']
    nidq_mm = (np.memmap(file_path, np.int16, mode='r').reshape(-1, num_chans))
    nidq_df = pd.DataFrame(nidq_mm, columns=chans)

    channel = 'epoc'
    nidq_df['tag'] = nidq_df[channel] > 1000
    # first row is a True preceded by a False
    fst = nidq_df.index[nidq_df['tag'] & ~ nidq_df['tag'].shift(1).fillna(False)]
    # last row is a True followed by a False
    lst = nidq_df.index[nidq_df['tag'] & ~ nidq_df['tag'].shift(-1).fillna(False)]
    timing = np.asarray([(i, j) for i, j in zip(fst, lst) if j > i + 4])
    return nidq_df, timing


def get_imec_flips(sync_data):
    diff = np.diff(sync_data)
    fst = np.where(diff == -1)[0] + 1
    lst = np.where(diff == 1)[0]

    flips = np.vstack((lst, fst)).T
    return flips


def get_stim_samples(nidq_df, start_sample, end_sample, hi_cut=1450, lo_cut=500):
    """
    Get starting sample index of each 20ms pulse from photodiode.
    """
    data = nidq_df[start_sample:end_sample]
    peaks = detect_peaks(data.photodiode.as_matrix(), mph=lo_cut, mpd=200)
    peaks += start_sample

    # import peakutils as peakutils
    # peaks = peakutils.indexes(data.photodiode, thres=(1100/data.photodiode.max()), min_dist=200)

    peaks_high = peaks[np.where(data.photodiode[peaks] > hi_cut)]
    peaks_high = peaks_high[1:]
    peaks_low = peaks[np.where(np.logical_and(data.photodiode[peaks] > lo_cut, data.photodiode[peaks] < hi_cut))]

    peaks = np.sort(np.concatenate([peaks_high, peaks_low]))

    # Find rising edges of photodiode and mark as trial beginning and end
    data_peaks = pd.DataFrame()
    data_peaks['photodiode'] = data.photodiode[peaks]

    data_peaks['high'] = data_peaks.photodiode > 1500
    high = data_peaks.index[data_peaks['high'] & ~ data_peaks['high'].shift(1).fillna(False)]

    data_peaks['low'] = data_peaks.photodiode < 1500
    low = data_peaks.index[data_peaks['low'] & ~ data_peaks['low'].shift(1).fillna(False)]

    stim_times = np.sort(np.concatenate([high, low]))
    return stim_times

'''
lf_file_path = "G:\\data-test\\000_015\\AE3_000_015_g0_t0.imec.ap.bin"
ni_file_path = "G:\\data-test\\000_015\\AE3_000_015_g0_t0.nidq.bin"
imec_data = extract_sync_samples(lf_file_path)
imec_flips = get_imec_flips(imec_data, channel=16)
nidq_data, timing = extract_nidq_samples(ni_file_path)

timing = timing / 25000
imec_flips = imec_flips / 30000

nidq_norm = timing[:, 1] - timing[0, 1]
imec_norm = imec_flips[:, 1] - imec_flips[0, 1]
multiplier = np.nanmean(nidq_norm / imec_norm)
'''
