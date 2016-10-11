import os
import glob
import functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'E:/color_vision/002_007_ori_sf/'
analyzer_paths = sorted(list(glob.iglob(data_folder + '*.analyzer')))

if len(analyzer_paths) > 1:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time'])
    exp_info['analyzer_path'] = analyzer_paths

data_path = list(glob.iglob(data_folder + '*.bin'))[0]
csv_path = list(glob.iglob(data_folder + '*.csv'))[0]
spikesort = 'kilosort'

for analyzer_path in analyzer_paths:
    analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
    if not os.path.exists(data_folder + analyzer_name + '_images'):
        os.makedirs(data_folder + analyzer_name + '_images')

    trial_num, stim_time = functions.analyzer_pg(analyzer_path)
    trial_num = pd.DataFrame(trial_num, columns=['orientation', 's_frequency'])

    if len(analyzer_paths) > 1:
        start_time = exp_info.loc[exp_info.analyzer == analyzer_name].start_time
        end_time = exp_info.loc[exp_info.analyzer == analyzer_name].end_time
        trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, start_time, end_time) / 25000
    else:
        trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0, 0) / 25000

    trial_num['stim_end'] = trial_num.stim_start + stim_time[2]

    sp = pd.DataFrame()
    if spikesort == 'jrclust':
        sp = functions.jrclust_csv(csv_path)
    elif spikesort == 'kilosort':
        sp = functions.kilosort_info(data_folder)

    for cluster in np.sort(sp.cluster.unique()):
        sp_cluster = sp[sp.cluster == cluster]
        start_times = trial_num.stim_start.as_matrix()
        end_times = trial_num.stim_end.as_matrix()
        all_times = np.sort(np.concatenate([start_times, end_times]))

        spike_cut = pd.cut(sp_cluster.time, bins=all_times)
        spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
        trial_num['cluster' + str(cluster)] = spike_counts.values[::2] / stim_time[2]

    counts = trial_num.groupby(['orientation', 's_frequency']).mean().drop(256)

    plt.ioff()
    for cluster in np.sort(sp.cluster.unique()):
        cluster_str = 'cluster' + str(cluster)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title(cluster_str)
        ax1.plot(counts[cluster_str].unstack(level=0).mean(axis=1))
        ax2.plot(counts[cluster_str].unstack(level=0).mean(axis=0))
        plt.savefig(data_folder + analyzer_name + '_images/' + cluster_str + '_all.pdf', format='pdf')
        plt.close()
    plt.ion()
