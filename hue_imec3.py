import os
import glob
import numpy as np
import pandas as pd
import functions

data_folder = 'F:/NHP/AE4/AE4_U003_E009_achromatic grating/'
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

data_path = glob.glob(data_folder + '*ap.bin')[0]
csv_path = glob.glob(data_folder + '*.csv')[0]
spikesort = 'jrclust'

spikes = pd.DataFrame()
if spikesort == 'jrclust':
    spikes = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    spikes = functions.kilosort_info(data_folder)

if not os.path.exists(data_folder + '/_images'):
    os.makedirs(data_folder + '/_images')

# get waveform and index of max channel (for each cluster)
# waveform, min_index = waveform.waveform_analysis(data_folder)

analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
print("Getting timing information: " + analyzer_name)
trial_num, stim_time = functions.analyzer_pg_conds(analyzer_path)

trial_num['direction'] = trial_num.ori
trial_num.ori[(trial_num.ori >= 180) & (trial_num.ori != 256)] = trial_num.ori[(trial_num.ori >= 180) &
                                                                               (trial_num.ori != 256)] - 180

def extract_sync_samples(file_path):
    num_chans = 385

    lfp_mm = (np.memmap(file_path, dtype='uint16', mode='r').reshape(-1, num_chans)).T
    sync_data = lfp_mm[num_chans - 1, :]

    sync_data_df = pd.DataFrame(sync_data, columns=['sync_data'])
    sync_data_df['binary'] = sync_data_df.sync_data.apply(lambda x: format(x, '04b'))
    sync_data_bits = np.fromstring(sync_data_df.binary.values.astype(str), dtype=int).reshape(-1, 16) - 48

    return sync_data_bits


def get_imec_flips(imec_data, channel):
    diff = np.diff(imec_data[:, channel-1])
    fst = np.where(diff == -1)[0] + 1
    lst = np.where(diff == 1)[0]

    flips = np.vstack((lst, fst)).T
    return flips


if not os.path.exists(data_folder + 'stim_start_samples.npy'):
    # trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0)[1::3] / 25000

    imec_data = extract_sync_samples(data_path)
    imec_flips = get_imec_flips(imec_data, channel=16)

    trial_num['stim_start'] = imec_flips[:, 0] / 30000
    trial_num['stim_end'] = imec_flips[:, 1] / 30000
    np.save(os.path.dirname(data_path) + '/stim_start_samples.npy', trial_num.stim_start * 30000)
    np.save(os.path.dirname(data_path) + '/stim_end_samples.npy', trial_num.stim_end * 30000)
else:
    trial_num['stim_start'] = np.load(data_folder + 'stim_start_samples.npy') / 30000
    trial_num['stim_end'] = np.load(data_folder + 'stim_end_samples.npy') / 30000

trial_num['stim_start'] = trial_num['stim_start'] + stim_time[0]
trial_num['stim_end'] = trial_num['stim_end'] - stim_time[1]

