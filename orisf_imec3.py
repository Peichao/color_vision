import os
import glob
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import functions

data_folder = 'F:/NHP/AE4/U006/006_003_achromatic_grating/'
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

data_path = glob.glob(data_folder + '*ap.bin')[0]
spikesort = 'jrclust'

spikes = pd.DataFrame()
fs = 30000

if spikesort == 'jrclust':
    csv_path = glob.glob(data_folder + '*.csv')[0]
    spikes = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    spikes = functions.kilosort_info(data_folder, fs)

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

respond_1 = []


def extract_sync_samples(file_path, downsample_factor):
    num_chans = 385

    lfp_mm = (np.memmap(file_path, dtype='uint16', mode='r').reshape(-1, num_chans)).T
    sync_data = lfp_mm[num_chans - 1, :]

    sync_data = sp.signal.decimate(sync_data, downsample_factor)

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

    downsample_factor = 10
    imec_data = extract_sync_samples(data_path, downsample_factor)
    imec_flips = get_imec_flips(imec_data, channel=16)
    imec_flips *= downsample_factor

    trial_num['stim_start'] = imec_flips[:, 0] / fs
    trial_num['stim_end'] = imec_flips[:, 1] / fs
    np.save(os.path.dirname(data_path) + '/stim_start_samples.npy', trial_num.stim_start * fs)
    np.save(os.path.dirname(data_path) + '/stim_end_samples.npy', trial_num.stim_end * fs)
else:
    trial_num['stim_start'] = np.load(data_folder + 'stim_start_samples.npy') / 30000
    trial_num['stim_end'] = np.load(data_folder + 'stim_end_samples.npy') / 30000

trial_num['stim_start'] = trial_num['stim_start'] + stim_time[0]
trial_num['stim_end'] = trial_num['stim_end'] - stim_time[1]

var1 = 'direction'
var2 = 's_freq'
make_plot = True
import matplotlib.pyplot as plt

for cluster in np.sort(spikes.cluster.unique())[np.sort(spikes.cluster.unique()) > 0]:
    cluster_str = 'cluster' + str(cluster)
    print('Processing %s' % cluster_str)

    sp_cluster = spikes[spikes.cluster == cluster]
    start_times = trial_num['stim_start'].as_matrix()
    end_times = trial_num['stim_end'].as_matrix()
    all_times = np.sort(np.concatenate([start_times, end_times]))

    spike_cut = pd.cut(sp_cluster.time, bins=all_times)
    spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
    trial_num[cluster_str] = (spike_counts.values[::2].flatten() / stim_time[2])

    counts_ori = trial_num.groupby([var1, var2])[cluster_str].mean()
    counts_sf = trial_num.groupby([var2, var1])[cluster_str].mean()
    sem_ori = trial_num.groupby([var1, var2])[cluster_str].sem()
    sem_sf = trial_num.groupby([var2, var1])[cluster_str].sem()

    s_freq_max = counts_ori.unstack(level=1).mean(axis=0).idxmax()
    orientation_max = counts_ori.unstack(level=1).mean(axis=1).idxmax()
    if counts_sf[s_freq_max].max() > 1:
        respond_1.append(cluster)

    if make_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        if spikesort == 'jrclust':
            fig.suptitle("Cluster %d, Site %d" % (cluster, spikes[spikes.cluster == cluster].max_site.mode()[0]),
                         fontsize=16)
        elif spikesort == 'kilosort':
            fig.suptitle("Cluster %d" % cluster, fontsize=16)

        colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

        counts_ori[orientation_max].plot(ax=ax1, linewidth=2)
        ax1.fill_between(counts_ori[orientation_max].index,
                         counts_ori[orientation_max] - sem_ori[orientation_max],
                         counts_ori[orientation_max] + sem_ori[orientation_max],
                         color='#E24A33', alpha=0.5)
        ax1.set_title('Spatial Frequency Tuning')
        ax1.set_ylabel('Spike Rate (Hz)')
        ax1.set_xlabel('Spatial Frequency (cycles/degree)')

        counts_sf[s_freq_max].plot(ax=ax2, linewidth=2)
        ax2.fill_between(counts_sf[s_freq_max].index,
                         counts_sf[s_freq_max] - sem_sf[s_freq_max],
                         counts_sf[s_freq_max] + sem_sf[s_freq_max],
                         color='#E24A33', alpha=0.5)
        ax2.set_title('Orientation Tuning')
        ax2.set_ylabel('Spike Rate (Hz)')
        ax2.set_xlabel('Orientation (degrees)')

        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        fig.subplots_adjust(top=0.88)
        plt.savefig(data_folder + '_images/' + cluster_str + '_all.pdf', bbox_inches='tight', format='pdf')
        plt.close()

        ax = plt.subplot(111, projection='polar')
        ax.plot(np.radians(counts_sf[s_freq_max].index), counts_sf[s_freq_max].values)
        ax.set_xticks(np.radians(counts_sf[s_freq_max].index))
        ax.set_rlabel_position(-20)
        plt.savefig(data_folder + '_images/' + cluster_str + '_polar.pdf', bbox_inches='tight', format='pdf')
        plt.close()
