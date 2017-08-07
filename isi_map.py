import os
import glob
import numpy as np
import scipy.io as sio
import functions

animal = 'AE1'
unit = '000'
exp = '004'

main_dir = 'F:/NHP'
mov_folder = os.path.join(main_dir, animal, 'ISI data', 'u%s_e%s' % (unit, exp))
mov_file = glob.glob(mov_folder + '/trial_response.mat')[0]

analyzer_path = os.path.join(main_dir, animal, 'AnalyzerFiles', animal, '%s_u%s_%s' % (animal, unit, exp) + '.analyzer')
trial_info, stim_time = functions.analyzer_pg_conds(analyzer_path)
trials = sio.loadmat(mov_file)['trials']

# response_filt = np.zeros([trials['response'][0, 0].shape[0],
#                           trials['response'][0, 0].shape[1],
#                           trials['response'].size])
#
# for trial_idx, trial in enumerate(trials['response'][0, :]):
#     print('Processing trial %d.' % (trial_idx + 1))
#     response_filt[:, :, trial_idx] = functions.butterworthbpf(trial, 1, 40, 4)

