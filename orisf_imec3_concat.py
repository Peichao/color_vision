import os
import glob
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import signal
import pandas as pd
import functions
import matplotlib.pyplot as plt

plot_polar = 'n'

data_folder = 'H:/AE4/004/AE4_u004_009/'
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]
spikesort = 'kilosort'
fs = 30000
start_sample = 291397164

spikes = pd.DataFrame()
if spikesort == 'jrclust':
    csv_path = glob.glob(data_folder + '*.csv')[0]
    spikes = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    kilosort_folder = os.path.dirname(os.path.dirname(data_folder)) + '/'
    spikes = functions.kilosort_info(kilosort_folder, fs)
spikes['time'] = spikes['time'] - (start_sample/fs)

if not os.path.exists(data_folder + '/_images'):
    os.makedirs(data_folder + '/_images')

# get waveform and index of max channel (for each cluster)
# waveform, min_index = waveform.waveform_analysis(data_folder)

analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
trial_num, stim_time = functions.analyzer_pg_conds(analyzer_path)

trial_num['direction'] = trial_num.ori
trial_num.ori[(trial_num.ori >= 180) & (trial_num.ori != 256)] = trial_num.ori[(trial_num.ori >= 180) &
                                                                               (trial_num.ori != 256)] - 180

stim_samples_fs = 2500
stim_samples_mat = sio.loadmat(data_folder + 'stim_samples.mat', squeeze_me=True, struct_as_record=False)['eventTimes']
stim_start = stim_samples_mat[1] / stim_samples_fs
stim_end = stim_samples_mat[2] / stim_samples_fs

trial_num['stim_start'] = stim_start
trial_num['stim_end'] = stim_end

var1 = 'direction'
var2 = 's_freq'
make_plot = 'y'

respond_1 = []

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

        if plot_polar == 'y':
            ax = plt.subplot(111, projection='polar')
            ax.plot(np.radians(counts_sf[s_freq_max].index), counts_sf[s_freq_max].values)
            ax.set_xticks(np.radians(counts_sf[s_freq_max].index))
            ax.set_rlabel_position(-20)
            plt.savefig(data_folder + '_images/' + cluster_str + '_polar.pdf', bbox_inches='tight', format='pdf')
            plt.close()
