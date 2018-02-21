import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import phase3_sync

data_folder = '/Volumes/Seagate Backup Plus Drive/AE4/007_000_hartley_achromatic/'
nidq_path = glob.glob(data_folder + '*.nidq.bin')[0]
imlf_path = glob.glob(data_folder + '*.lf.bin')[0]
# analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

nidq_df, timing = phase3_sync.extract_nidq_samples(nidq_path)
sync_data = phase3_sync.extract_sync_samples(imlf_path)

im_flips = phase3_sync.get_imec_flips(sync_data)
ni_flips = np.round(timing / 10).astype('int64')
ni_im_offset = ni_flips[0, 0] - im_flips[0, 0]
ni_flips_offset = ni_flips - ni_im_offset
ni_drift = np.mean(im_flips.ravel()[1:] / ni_flips_offset.ravel()[1:])

timing_trials, _timing = np.shape(timing)
trial_samples = []

for trial in np.arange(timing_trials):
    print('Processing trial %d' % trial + 1)
    start_sample = timing[trial, 0]
    end_sample = timing[trial, 1]
    channel = 'photodiode'
    timing_trial = phase3_sync.get_stim_samples(nidq_df, start_sample, end_sample)
    trial_samples.append(timing_trial)
