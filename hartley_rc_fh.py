import functions
import numpy as np

image_path = 'E:/color_vision/images/'
image_array = np.load(image_path + 'image_array.npy')

params_path = 'E:/color_vision/20160914/2016.9.14xx0_010_000.mat'
data_path = 'E:/color_vision/20160915_photodiode2_g0_t0.nidq.bin'

trials = functions.get_params(params_path)
# trials['stim_sample'] = functions.get_stim_samples(data_path)
# trials['stim_time'] = trials.stim_sample / 25000
trials['stim_time'] = np.arange(0, 600, 0.02)

spike_times = np.random.uniform(20, 500, size=5000)
p = functions.PlotRF(trials, spike_times, image_array)
p.show()
