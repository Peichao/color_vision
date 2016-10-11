import functions
import numpy as np

image_path = 'E:/color_vision/images/'
image_array = np.load(image_path + 'image_array.npy')

params_path = 'E:/color_vision/20160919/hartley_l1/2016.9.19AD7_000_001.mat'
data_path = 'E:/color_vision/20160919/hartley_l1/20160919_unit000_001_Lcone_hartley_g0_t0.nidq.bin'
jrclust_path = 'E:/color_vision/20160919/hartley_l1/20160919_unit000_001_Lcone_hartley_g0_t0.nidq.csv'

sp = functions.jrclust_csv(jrclust_path)

trials = functions.get_params(params_path)
trials['stim_sample'] = functions.get_stim_samples_fh(data_path)
trials['stim_time'] = trials.stim_sample / 25000

p = functions.PlotRF(trials, sp.time[sp.cluster == 5].as_matrix(), image_array)
p.show()
