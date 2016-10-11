import os
import glob
import functions
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

data_folder = 'F:/NHP/20160921/001_040_ori_sf/'
analyzer_paths = sorted(list(glob.iglob(data_folder + '*.analyzer')))

if len(analyzer_paths) > 1:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time'])
    exp_info['analyzer_path'] = analyzer_paths

data_path = list(glob.iglob(data_folder + '*.bin'))[0]
csv_path = list(glob.iglob(data_folder + '*.csv'))[0]
spikesort = 'jrclust'

sp = pd.DataFrame()
if spikesort == 'jrclust':
    sp = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    sp = functions.kilosort_info(data_folder)

if not os.path.exists(data_folder + '_images'):
    os.makedirs(data_folder + '_images')

plt.ioff()
for cluster in np.sort(sp.cluster.unique()):
    cluster_str = 'cluster' + str(cluster)
    fig, axarr = plt.subplots(2, len(analyzer_paths))

    for i, analyzer_path in enumerate(analyzer_paths):
        analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
        print("Analyzing " + analyzer_name)

        trial_num, stim_time = functions.analyzer_pg(analyzer_path)
        trial_num = pd.DataFrame(trial_num, columns=['orientation', 's_frequency'])

        if len(analyzer_paths) > 1:
            start_time = exp_info.loc[exp_info.analyzer == analyzer_name].start_time
            end_time = exp_info.loc[exp_info.analyzer == analyzer_name].end_time
            trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, start_time, end_time) / 25000
        else:
            trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0, 0) / 25000

        trial_num['stim_end'] = trial_num.stim_start + stim_time[2]

        sp_cluster = sp[sp.cluster == cluster]
        start_times = trial_num.stim_start.as_matrix()
        end_times = trial_num.stim_end.as_matrix()
        all_times = np.sort(np.concatenate([start_times, end_times]))

        spike_cut = pd.cut(sp_cluster.time, bins=all_times)
        spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
        trial_num['cluster' + str(cluster)] = spike_counts.values[::2] / stim_time[2]

        counts = trial_num.groupby(['orientation', 's_frequency']).mean().drop(256)

        axarr[0, i].plot(counts[cluster_str].unstack(level=0).mean(axis=1))
        axarr[0, i].set_title('Spatial Frequency Tuning')
        axarr[0, i].set_ylabel('Spike Rate (Hz)')
        axarr[0, i].set_xlabel('Spatial Frequency (cycles/degree)')

        axarr[1, i].plot(counts[cluster_str].unstack(level=0).mean(axis=0))
        axarr[1, i].set_title('Orientation Tuning')
        axarr[1, i].set_ylabel('Spike Rate (Hz)')
        axarr[1, i].set_xlabel('Orientation (degrees)')

    plt.tight_layout()
    plt.savefig(data_folder + '_images/' + cluster_str + '_all.pdf', format='pdf')
    plt.close()
plt.ion()
