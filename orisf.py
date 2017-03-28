import os
import glob
import functions
import csd_analysis
import waveform
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib
from matplotlib import colors, cm
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

spikes = pd.DataFrame()
if spikesort == 'jrclust':
    spikes = functions.jrclust_csv(csv_path)
elif spikesort == 'kilosort':
    spikes = functions.kilosort_info(data_folder)

if not os.path.exists(data_folder + '/_images'):
    os.makedirs(data_folder + '/_images')

# get waveform and index of max channel (for each cluster)
waveform, min_index = waveform.waveform_analysis(data_folder)

analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]
print("Getting timing information: " + analyzer_name)
trial_num, stim_time = functions.analyzer_pg(analyzer_path)
trial_num = pd.DataFrame(trial_num, columns=['orientation', 's_frequency'])

if not os.path.exists(data_folder + 'stim_samples.npy'):
    trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0)[1::3] / 25000
    # trial_num['stim_start'] = functions.get_stim_samples_pg(data_path, 0) / 25000
else:
    trial_num['stim_start'] = np.load(data_folder + 'stim_samples.npy')[1::3] / 25000

trial_num['stim_end'] = trial_num.stim_start + stim_time[2]

for cluster in np.sort(spikes.cluster.unique())[np.sort(spikes.cluster.unique()) > 0]:
    functions.orisf_plot(data_folder, spikes, cluster, trial_num, stim_time, make_plot=True)

connect_array = functions.xcorr_psth_all(data_folder, spikes, trial_num, stim_time,
                                         bin_size=0.001, maxlags=200, plot_lags=50)

plt.close()
fig, ax = plt.subplots()
normal = matplotlib.colors.Normalize(connect_array.min(), connect_array.max())
connect_table = ax.table(cellText=connect_array.astype(int), rowLabels=['infragranular', 'granular', 'supragranular'],
                         colLabels=['infragranular', 'granular', 'supragranular'],
                         cellLoc='center', loc='center',
                         cellColours=matplotlib.cm.summer(normal(connect_array)), alpha=0.5)
connect_table.scale(1, 2)
plt.subplots_adjust(left=0.2, bottom=0.2)
fig.patch.set_visible(False)
ax.axis('off')
plt.show()
