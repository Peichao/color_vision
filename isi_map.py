import os
import glob
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import functions

animal = 'AE2'
unit = '000'
exp = '004'

main_dir = 'H:/AE2/u000_e004'
mov_folder = main_dir
# mov_folder = os.path.join(main_dir, animal, 'ISI data', 'u%s_e%s' % (unit, exp))
mov_file = glob.glob(mov_folder + '/trial_response.mat')[0]

# analyzer_path = os.path.join(main_dir, animal, 'AnalyzerFiles', animal, '%s_u%s_%s' % (animal, unit, exp) + '.analyzer')
analyzer_folder = 'H:/AE2/AnalyzerFiles/AE2'
analyzer_path = os.path.join(analyzer_folder, '%s_u%s_%s' % (animal, unit, exp) + '.analyzer')
trial_info, stim_time = functions.analyzer_pg_conds(analyzer_path)
response_filt = sio.loadmat(mov_file)['response']

# response_filt = np.zeros([trials['response'][0, 0].shape[0],
#                           trials['response'][0, 0].shape[1],
#                           trials['response'].size])
#
# for trial_idx, trial in enumerate(trials['response'][0, :]):
#     print('Processing trial %d.' % (trial_idx + 1))
#     response_filt[:, :, trial_idx] = functions.butterworthbpf(trial, 1, 40, 4)

cond_1 = np.where((trial_info.ori == 45) | (trial_info.ori == 215))[0]
cond_2 = np.where((trial_info.ori == 135) | (trial_info.ori == 315))[0]
# cond_1 = np.where((trial_info.colormod == 9))[0]
# cond_2 = np.where((trial_info.colormod == 1))[0]

mean_response_1 = np.mean(response_filt[:, :, cond_1], axis=2)
mean_response_2 = np.mean(response_filt[:, :, cond_2], axis=2)

diff_means = mean_response_2 - mean_response_1
t_map = diff_means * np.sqrt(np.size(cond_1) / np.std(diff_means.ravel()))

response_1 = response_filt[:, :, cond_1]
response_2 = response_filt[:, :, cond_2]

stat, p = sp.stats.ttest_ind(response_1, response_2, axis=2)
ttest_clip = functions.isi_clip(p, 2)

t_map_clip = functions.isi_clip(t_map, 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(t_map, cmap='binary')
ax2.imshow(t_map_clip, cmap='binary')

plt.show()
