import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

array_file = '/Users/anupam/Downloads/imageArray.mat'
exp_params_file = '/Users/anupam/Downloads/xx0_001_002.mat'

imageArray = sio.loadmat(array_file, squeeze_me=True, struct_as_record=False)['imageArray']
exp_params = sio.loadmat(exp_params_file, squeeze_me=True, struct_as_record=False)['randlog_T1']

color_dom = pd.Series(exp_params.domains.colordom)
ori_dom = pd.Series(exp_params.domains.oridom)
sf_dom = pd.Series(exp_params.domains.sfdom)
phase_dom = pd.Series(exp_params.domains.phasedom)

trials_dict = {'color': exp_params.seqs.colorseq,
               'ori': exp_params.seqs.oriseq,
               'sf': exp_params.seqs.sfseq,
               'phase': exp_params.seqs.phaseseq}
trials = pd.DataFrame(trials_dict)
trials -= 1

trials['color'] = trials.color.map(color_dom)
trials['ori'] = trials.ori.map(ori_dom)
trials['sf'] = trials.sf.map(sf_dom)
trials['phase'] = trials.phase.map(phase_dom)
trials_unique = trials.drop_duplicates()

stim_times = np.arange(0, 10, 0.02)
