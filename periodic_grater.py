import os
import glob
import functions
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

data_folder = 'F:/NHP/AD8/Ephys/20161205/orisf_gray'
analyzer_paths = sorted(list(glob.iglob(data_folder + '*.analyzer')))

if len(analyzer_paths) > 1:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time', 'stimulus'])

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

trial_num = {}
stim_time = 0
for i, analyzer_path in enumerate(analyzer_paths):
    analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
    stim = exp_info[exp_info.analyzer == analyzer_name]['stimulus'][i]
    print("Getting timing information: " + analyzer_name)

    trial_num[stim], stim_time = functions.analyzer_pg(analyzer_path)
    trial_num[stim] = pd.DataFrame(trial_num[stim], columns=['orientation', 's_frequency'])

    if len(analyzer_paths) > 1:
        start_time = exp_info.loc[exp_info.analyzer == analyzer_name].start_time
        end_time = exp_info.loc[exp_info.analyzer == analyzer_name].end_time
        trial_num[stim]['stim_start'] = functions.get_stim_samples_pg(data_path, start_time, end_time) / 25000
    else:
        trial_num[stim]['stim_start'] = functions.get_stim_samples_pg(data_path, 0, 0) / 25000

    trial_num[stim]['stim_end'] = trial_num[stim].stim_start + stim_time[2]
all_trials = pd.concat(trial_num).sort(['stim_start'])
all_trials.index.names = ['analyzer', 'trial']
all_trials_no_blanks = all_trials[all_trials.orientation != 256]

plt.ioff()
for cluster in np.sort(sp.cluster.unique())[np.sort(sp.cluster.unique()) > 0]:
    cluster_str = 'cluster' + str(cluster)

    sp_cluster = sp[sp.cluster == cluster]
    start_times = all_trials_no_blanks['stim_start'].as_matrix()
    end_times = all_trials_no_blanks['stim_end'].as_matrix()
    all_times = np.sort(np.concatenate([start_times, end_times]))

    spike_cut = pd.cut(sp_cluster.time, bins=all_times)
    spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
    all_trials_no_blanks[cluster_str] = (spike_counts.values[::2].flatten() / stim_time[2])

    counts = all_trials_no_blanks.groupby([all_trials_no_blanks.index.get_level_values(0),
                                            'orientation', 's_frequency']).mean()
    sem = all_trials_no_blanks.groupby([all_trials_no_blanks.index.get_level_values(0),
                                            'orientation', 's_frequency']).sem()

    s_freq_max = counts[cluster_str].unstack(level=1).mean(axis=1).unstack(0).idxmax()
    s_freq_max_idx = s_freq_max.index.tolist()
    s_freq_max_list = s_freq_max.tolist()

    orientation_max = counts[cluster_str].unstack(level=2).mean(axis=1).unstack(0).idxmax()
    orientation_max_idx = orientation_max.index.tolist()
    orientation_max_list = orientation_max.tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Cluster %d, Site %d" % (cluster, sp[sp.cluster == cluster].max_site.mode()[0]), fontsize=16)

    colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

    for i, idx in enumerate(s_freq_max_idx):
        counts.loc[orientation_max_idx[i], orientation_max_list[i], :][cluster_str].\
            unstack(level=1).mean(axis=1).unstack(level=0).plot(ax=ax1, linewidth=2, color=colors[i])
        s_freq_x = np.array(counts.loc[orientation_max_idx[i], orientation_max_list[i], :][cluster_str].\
                            unstack(level=1).mean(axis=1).unstack(level=0).index.tolist())
        s_freq_y = np.array(counts.loc[orientation_max_idx[i], orientation_max_list[i], :][cluster_str].\
                            unstack(level=1).mean(axis=1).unstack(level=0).values.tolist()).T[0]
        s_freq_sem = np.array(sem.loc[orientation_max_idx[i], orientation_max_list[i], :][cluster_str].\
                              unstack(level=1).mean(axis=1).unstack(level=0).values.tolist()).T[0]
        ax1.fill_between(s_freq_x, s_freq_y + s_freq_sem, s_freq_y - s_freq_sem, alpha=0.5, color=colors[i])

        counts.loc[s_freq_max_idx[i], :, s_freq_max_list[i]][cluster_str].\
            unstack(level=2).mean(axis=1).unstack(level=0).plot(ax=ax2, linewidth=2, color=colors[i])
        orientation_x = np.array(counts.loc[s_freq_max_idx[i], :, s_freq_max_list[i]][cluster_str].\
                                 unstack(level=2).mean(axis=1).unstack(level=0).index.tolist())
        orientation_y = np.array(counts.loc[s_freq_max_idx[i], :, s_freq_max_list[i]][cluster_str].\
                                 unstack(level=2).mean(axis=1).unstack(level=0).values.tolist()).T[0]
        orientation_sem = np.array(sem.loc[s_freq_max_idx[i], :, s_freq_max_list[i]][cluster_str].\
                                   unstack(level=2).mean(axis=1).unstack(level=0).values.tolist()).T[0]
        ax2.fill_between(orientation_x, orientation_y + orientation_sem,
                         orientation_y - orientation_sem, alpha=0.5, color=colors[i])

    ax1.set_title('Spatial Frequency Tuning')
    ax1.set_ylabel('Spike Rate (Hz)')
    ax1.set_xlabel('Spatial Frequency (cycles/degree)')

    ax2.set_title('Orientation Tuning')
    ax2.set_ylabel('Spike Rate (Hz)')
    ax2.set_xlabel('Orientation (degrees)')

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(top=0.88)
    plt.savefig(data_folder + '_images/' + cluster_str + '_all.pdf', bbox_inches='tight', format='pdf')
    plt.close()
plt.ion()
