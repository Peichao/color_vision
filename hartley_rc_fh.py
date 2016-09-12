import numpy as np
import scipy.io as sio
import h5py
import pandas as pd
import matplotlib.pyplot as plt

array_file = '/Users/anupam/Library/Mobile Documents/' \
             'com~apple~CloudDocs/Documents/Callaway Lab/color_vision/imageArray.mat'
exp_params_file = '/Users/anupam/Library/Mobile Documents/' \
                  'com~apple~CloudDocs/Documents/Callaway Lab/color_vision/2016.9.9xx0_008_003.mat'

# imageArray = sio.loadmat(array_file, squeeze_me=True, struct_as_record=False)['imageArray']
exp_params = sio.loadmat(exp_params_file, squeeze_me=True, struct_as_record=False)['randlog_T1']
imageArray_hdf5 = h5py.File(array_file, 'r')
imageArray = imageArray_hdf5['imageArray']

for i, hdf5object in enumerate(imageArray_hdf5['imageArray'][()][0]):
    imageArray[i] = hdf5object[()]

columns = ['quadrant', 'kx', 'ky', 'bit', 'color']
conds = pd.DataFrame(exp_params.domains.Cond, columns=columns)

trials = pd.DataFrame(exp_params.seqs.frameseq, columns=['cond'])
trials -= 1

trials['quadrant'] = trials.cond.map(conds['quadrant'])
trials['kx'] = trials.cond.map(conds['kx'])
trials['ky'] = trials.cond.map(conds['ky'])
trials['bit'] = trials.cond.map(conds['bit'])
trials['color'] = trials.cond.map(conds['color'])
