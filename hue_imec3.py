import os
import glob
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import signal
import pandas as pd
import functions
import matplotlib.pyplot as plt

data_folder = 'F:/NHP/AE4/U006/006_002_hue/'
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

data_path = glob.glob(data_folder + '*ap.bin')[0]
csv_path = glob.glob(data_folder + '*.csv')[0]
spikesort = 'kilosort'
fs = 30000

spikes = pd.DataFrame()
if spikesort == 'jrclust':
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

stim_samples_fs = 2500
stim_samples_mat = sio.loadmat(data_folder + 'stim_samples.mat', squeeze_me=True, struct_as_record=False)['eventTimes']
stim_start = stim_samples_mat[1] / stim_samples_fs
stim_end = stim_samples_mat[2] / stim_samples_fs

trial_num['stim_start'] = stim_start
trial_num['stim_end'] = stim_end

var1 = 'direction'
var2 = 's_freq'
make_plot = 'y'

color_hex = ['#af1600', '#8a4600', '#5a5d01', '#2a6600', '#006a00', '#006931', '#006464', '#0058b6', '#002DFF', '#6a2ade', '#97209b', '#aa1c50']

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

    if make_plot == 'y':
        fig, ax = plt.subplots()
        if spikesort == 'jrclust':
            fig.suptitle("Cluster %d, Site %d" % (cluster, spikes[spikes.cluster == cluster].max_site.mode()[0]),
                         fontsize=16)
        elif spikesort == 'kilosort':
            fig.suptitle("Cluster %d" % cluster, fontsize=16)

        ax.scatter(trial_num.groupby('colormod').mean()[cluster_str].index.values,
                   trial_num.groupby('colormod').mean()[cluster_str].values,
                   color=color_hex, s=100, zorder=2)
        ax.fill_between(trial_num.groupby('colormod').mean()[cluster_str].index.values,
                        trial_num.groupby('colormod').mean()[cluster_str].values -
                        trial_num.groupby('colormod').sem()[cluster_str].values,
                        trial_num.groupby('colormod').mean()[cluster_str].values +
                        trial_num.groupby('colormod').sem()[cluster_str].values,
                        alpha=0.5, color='#808080', zorder=1)
        ax.xaxis.set_ticks([])
        ax.set_ylabel('Response (sp/s)')
        plt.savefig(data_folder + '_images/' + cluster_str + '_color.pdf', bbox_inches='tight', format='pdf')
        plt.close()
