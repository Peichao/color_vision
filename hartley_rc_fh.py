import functions
import numpy as np

image_path = 'E:/color_vision/images/'
image_array = np.load(image_path + 'image_array.npy')

params_path = 'E:/color_vision/20160916/2016.9.16xx1_002_002.mat'
data_path = 'E:/color_vision/20160916/20160916_photodiode_g0_t0.nidq.bin'

trials = functions.get_params(params_path)
trials['stim_sample'] = functions.get_stim_samples(data_path)
trials['stim_time'] = trials.stim_sample / 25000

spike_times = np.random.uniform(39, 1800, size=5000)
p = functions.PlotRF(trials, spike_times, image_array)
p.show()
