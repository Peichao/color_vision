import os
import glob
import tkinter as tk
from tkinter.filedialog import askopenfilename
import functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

for i in range(1, 74):
    cluster = i
    p = functions.PlotRF(trials, sp.time[sp.cluster == cluster].as_matrix(), analyzer_path, cluster)
    loadz = np.load(data_folder + 'revcorr_images_f_%d.npz' % cluster)
    # plt.imshow(loadz['images_f'][:, :, 155].T,
    #            interpolation='bicubic', cmap='jet', origin='lower',
    #            extent=[loadz['sfx_vals'].min(), loadz['sfx_vals'].max(),
    #            loadz['sfy_vals'].min(), loadz['sfy_vals'].max()])
    # plt.xlabel(r'$\omega_x$')
    # plt.ylabel(r'$\omega_y$')
    # p.show()
    #
    # trials.to_csv(data_folder + 'trials.csv')
    import scipy.io as sio
    revcorr_images_f = loadz['images_f']
    sio.savemat(data_folder + 'revcorr_images_f_%d.mat' % cluster, {'revcorr_images_f': revcorr_images_f})
    sio.savemat(data_folder + 'spike_times_%d.mat' % cluster, mdict={'spike_times': sp[sp.cluster == cluster].time.as_matrix()})

data_path = "F:/NHP/hartley_units/scone/20160919_scone.csv"
date = 20160919
csd_info = pd.read_csv("csd_borders.csv")
hartley_units = pd.read_csv(data_path, index_col=0)
hartley_units['csd_top'] = 31
hartley_units['csd_bottom'] = 11
probe_geo = functions.probe_geo
hartley_units['relative_depth'] = probe_geo[:, 1][hartley_units.max_site - 1] - \
                                  hartley_units['csd_bottom'][1]
hartley_units.to_csv(data_path)
