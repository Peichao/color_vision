import os
import glob
import tkinter as tk
from tkinter.filedialog import askopenfilename
import functions
import pandas as pd
import numpy as np

tk.Tk().withdraw()
params_paths = askopenfilename(initialdir='F:/NHP',
                               filetypes=(("Mat files", "*.mat"), ("All files", "*.*")))
data_folder = os.path.dirname(params_paths) + '/'

start_time = 0  # seconds
end_time = None  # seconds

if type(params_paths) != str:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time', 'stimulus'])

data_path = glob.glob(data_folder + '*nidq.bin')[0]
jrclust_path = glob.glob(data_folder + '*.csv')[0]
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

sp = functions.jrclust_csv(jrclust_path)
trials = functions.get_params(params_paths, analyzer_path)

if not os.path.exists(data_folder + '/stim_samples.npy'):
    trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
else:
    trials['stim_sample'] = np.load(data_folder + 'stim_samples.npy')
trials['stim_time'] = trials.stim_sample / 25000

# xcorr = functions.xcorr_spiketime_all(data_folder, sp, maxlag=50, spacing=1)

for clust in np.arange(1, 42):
    print('analyzing cluster %d' % clust)
    p = functions.PlotRF(trials, sp.time[sp.cluster == clust].as_matrix(), analyzer_path, clust)

cluster = 3
p = functions.PlotRF(trials, sp.time[sp.cluster == cluster].as_matrix(), analyzer_path, cluster)
p.show()
