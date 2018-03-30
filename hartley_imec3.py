import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import phase3_sync

data_folder = 'H:/AE4/U004/004_005_hartley_achromatic/'
print('Loading sync channels and calculating timing info.')
nidq_path = glob.glob(data_folder + '*.nidq.bin')[0]
imlf_path = glob.glob(data_folder + '*.ap.bin')[0]
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]
analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]

nidq_df, timing = phase3_sync.extract_nidq_samples(nidq_path)
sync_data = phase3_sync.extract_sync_samples(imlf_path)

im_flips = phase3_sync.get_imec_flips(sync_data)
ni_flips = timing

im_flips_z = im_flips - im_flips[0, 0]
ni_flips_z = ni_flips - ni_flips[0, 0]
ni_drift = np.mean(im_flips_z.ravel()[1:] / ni_flips_z.ravel()[1:])
ni_flips_corrected = (ni_flips_z * ni_drift) + im_flips[0, 0]

timing_trials, _timing = np.shape(timing)
trial_samples = np.array([])

for trial in np.arange(timing_trials):
    print('Processing trial %d' % (trial + 1))
    start_sample = timing[trial, 0]
    end_sample = timing[trial, 1]
    channel = 'photodiode'
    timing_trial = phase3_sync.get_stim_samples(nidq_df, start_sample, end_sample)
    trial_samples = np.append(trial_samples, timing_trial)

np.save(data_folder + analyzer_name + '_ni_flips.npy', timing)
np.save(data_folder + analyzer_name + '_im_flips.npy', im_flips)
np.save(data_folder + analyzer_name + '_trial_samples.npy', trial_samples)

# if not os.path.exists(data_folder + '/stim_samples.npy'):
#     self.trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
# else:
#     self.trials['stim_sample'] = np.load(data_folder + 'stim_samples.npy')
trial_samples_z = trial_samples - ni_flips[0, 0]
trial_samples_corrected = (trial_samples_z * ni_drift) + im_flips[0, 0]
