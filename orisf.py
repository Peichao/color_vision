import os
import glob
import functions
import waveform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# TODO quantify how many visually selective neurons there are per recording
# TODO quantify number of orientation selective neurons
# TODO characterize spatial frequency tuning of neurons

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')
plt.ioff()

data_folder = 'F:/NHP/AD8/Ephys/20161205/orisf_gray/'
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

data_path = glob.glob(data_folder + '*.bin')[0]
csv_path = glob.glob(data_folder + '*.csv')[0]
spikesort = 'jrclust'

sp = pd.DataFrame()
if spikesort == 'jrclust':
    sp = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    sp = functions.kilosort_info(data_folder)

if not os.path.exists(data_folder + '_images'):
    os.makedirs(data_folder + '_images')

# get waveform and index of max channel (for each cluster)
waveform, min_index = waveform.waveform_analysis(data_folder)

analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
print("Getting timing information: " + analyzer_name)
trial_num, stim_time = functions.analyzer_pg(analyzer_path)
trial_num = pd.DataFrame(trial_num, columns=['orientation', 's_frequency'])

if not os.path.exists(data_folder + 'stim_samples.npy'):
    trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0)[1::3] / 25000
else:
    trial_num['stim_start'] = np.load(data_folder + 'stim_samples.npy') / 25000

trial_num['stim_end'] = trial_num.stim_start + stim_time[2]

for cluster in np.sort(sp.cluster.unique())[np.sort(sp.cluster.unique()) > 0]:
    cluster_str = 'cluster' + str(cluster)

    sp_cluster = sp[sp.cluster == cluster]
    start_times = trial_num['stim_start'].as_matrix()
    end_times = trial_num['stim_end'].as_matrix()
    all_times = np.sort(np.concatenate([start_times, end_times]))

    spike_cut = pd.cut(sp_cluster.time, bins=all_times)
    spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
    trial_num[cluster_str] = (spike_counts.values[::2].flatten() / stim_time[2])

    counts_ori = trial_num.groupby(['orientation', 's_frequency'])[cluster_str].mean()
    counts_sf = trial_num.groupby(['s_frequency', 'orientation'])[cluster_str].mean()
    sem_ori = trial_num.groupby(['orientation', 's_frequency'])[cluster_str].sem()
    sem_sf = trial_num.groupby(['s_frequency', 'orientation'])[cluster_str].sem()

    s_freq_max = counts_ori.drop(256).unstack(level=1).mean(axis=0).idxmax()
    orientation_max = counts_ori.drop(256).unstack(level=1).mean(axis=1).idxmax()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Cluster %d, Site %d" % (cluster, sp[sp.cluster == cluster].max_site.mode()[0]), fontsize=16)

    colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

    counts_ori.drop(256)[orientation_max].plot(ax=ax1, linewidth=2)
    ax1.fill_between(counts_ori.drop(256)[orientation_max].index,
                     counts_ori.drop(256)[orientation_max] - sem_ori.drop(256)[orientation_max],
                     counts_ori.drop(256)[orientation_max] + sem_ori.drop(256)[orientation_max],
                     color='#E24A33', alpha=0.5)
    ax1.set_title('Spatial Frequency Tuning')
    ax1.set_ylabel('Spike Rate (Hz)')
    ax1.set_xlabel('Spatial Frequency (cycles/degree)')

    counts_sf.drop(256)[s_freq_max].plot(ax=ax2, linewidth=2)
    ax2.fill_between(counts_sf.drop(256)[s_freq_max].index,
                     counts_sf.drop(256)[s_freq_max] - sem_sf.drop(256)[s_freq_max],
                     counts_sf.drop(256)[s_freq_max] + sem_sf.drop(256)[s_freq_max],
                     color='#E24A33', alpha=0.5)
    ax2.set_title('Orientation Tuning')
    ax2.set_ylabel('Spike Rate (Hz)')
    ax2.set_xlabel('Orientation (degrees)')

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(top=0.88)
    plt.savefig(data_folder + '_images/' + cluster_str + '_all.pdf', bbox_inches='tight', format='pdf')
    plt.close()
