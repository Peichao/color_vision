import glob
import functions
import pandas as pd
import numpy as np

image_path = 'E:/color_vision/images/'
image_array = np.load(image_path + 'image_array.npy')

data_folder = 'F:/NHP/20160921/001_044_hartley/'
params_paths = "F:/NHP/20160921/001_044_hartley/2016.9.21AD7_001_044.mat"

start_time = 0
end_time = 31

if type(params_paths) != str:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time', 'stimulus'])

data_path = 'F:/NHP/20160921/001_044_hartley/20160921_AD7_001_044-47_hartley_g0_t0.exported.nidq.bin'
jrclust_path = 'F:/NHP/20160921/001_044_hartley/20160921_AD7_001_044-47_hartley_g0_t0.exported.nidq.csv'

sp = functions.jrclust_csv(jrclust_path)

trials = functions.get_params(params_paths)

# trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
# trials['stim_time'] = trials.stim_sample / 25000
test_trials = functions.get_stim_samples_fh(data_path, start_time, end_time)

p = functions.PlotRF(trials, sp.time[sp.cluster == 89].as_matrix(), image_array)
p.show()
