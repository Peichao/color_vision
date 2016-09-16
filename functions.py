import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import AxesImage

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')


def get_params(file_path):
    exp_params_all = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    r_seed = []
    for key in exp_params_all:
        if key.startswith('randlog'):
            r_seed.append(key)

    all_trials = pd.DataFrame()
    for seed in sorted(r_seed):
        exp_params = exp_params_all[seed]
        columns = ['quadrant', 'kx', 'ky', 'bwdom', 'color']
        conds = pd.DataFrame(exp_params.domains.Cond, columns=columns)
        trials = pd.DataFrame(exp_params.seqs.frameseq, columns=['cond'])
        x_size = exp_params.domains.x_size
        y_size = exp_params.domains.y_size
        trials -= 1

        for column in columns:
            trials[column] = trials.cond.map(conds[column])

        trials['ori'] = np.degrees(np.arctan(trials.ky / trials.kx)) % 360
        trials['sf'] = np.sqrt((trials.kx/x_size)**2 + (trials.ky/y_size)**2)

        all_trials = pd.concat([all_trials, trials], axis=0, ignore_index=True)

    return all_trials


def build_image_array(image_path):
    print('Building array of image files.')
    image_files = list(glob.iglob(image_path + '*.mat'))
    image_array = np.zeros([130, 120, 3, len(image_files)])
    for i in range(len(image_files)):
        image_array[:, :, :, i] = sio.loadmat(image_path + 'imageArray' + str(i + 1) + '.mat')['image'][319:449,
                                                                                                        447:567, :]
    image_array = image_array.astype('uint8')
    np.save(image_path + 'image_array.npy', image_array)
    return image_array


def get_unique_trials(params_path):
    trials = get_params(params_path)
    trials_unique = trials.drop_duplicates()

    return trials_unique


def average_image(image_list, image_array):
    mean_image = np.mean(image_array[:, :, :, image_list], axis=3)

    return mean_image


def get_mean_image(trials, spike_times, image_array):
    stim_idx = trials.stim_time.searchsorted(spike_times, side='right') - 1
    image_idx = trials.cond[stim_idx]
    mean_image = average_image(image_idx.values, image_array)
    return mean_image


def get_mean_point(x, y, trials, spike_times, image_array):
    stim_idx = trials.stim_time.searchsorted(spike_times, side='right') - 1
    image_idx = trials.cond[stim_idx]
    mean_point = np.mean(image_array[y, x, :, image_idx], axis=0)
    return mean_point


def get_stim_samples(file_path):
    data = (np.memmap(file_path, np.int16, mode='r').reshape(-1, 129)).T
    data = pd.DataFrame(data[128, :], columns=['photodiode'])

    from detect_peaks import detect_peaks
    peaks = detect_peaks(data.photodiode, mph=1000, mpd=200)
    trial_break_idx = np.where(np.diff(peaks) > 5000)[0]
    trial_break_idx = np.sort(np.concatenate([trial_break_idx, trial_break_idx + 1,
                                              trial_break_idx - 1, trial_break_idx - 2]))
    peaks = np.delete(peaks, trial_break_idx)[1:-3]

    data_peaks = pd.DataFrame()
    data_peaks['photodiode'] = data.photodiode[peaks]

    data_peaks['high'] = data_peaks.photodiode > 1500
    high = data_peaks.index[data_peaks['high'] & ~ data_peaks['high'].shift(1).fillna(False)]

    data_peaks['low'] = data_peaks.photodiode < 1500
    low = data_peaks.index[data_peaks['low'] & ~ data_peaks['low'].shift(1).fillna(False)]

    stim_times = np.sort(np.concatenate([high, low]))
    return stim_times


class PlotRF(object):
    def __init__(self, trials=None, spike_times=None, image_array=None):
        self.x = None
        self.y = None

        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Receptive Field')
        self.fig2, self.ax2 = plt.subplots()

        self.slider_ax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03], axisbg='yellow')
        self.slider = Slider(self.slider_ax, 'Value', 0, 0.20, valinit=0)
        self.slider.on_changed(self.update)
        self.slider.drawon = False

        self.trials = trials
        self.spike_times = spike_times
        self.image_array = image_array
        self.starting_image = get_mean_image(self.trials, self.spike_times, self.image_array)

        self.l = self.ax.imshow(self.starting_image, picker=True)

        self.t = np.arange(-0.2, 0, 0.001)
        self.ydata = np.zeros(np.size(self.t))
        self.l2, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlim([-0.20, 0])

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

    def update(self, value):
        spike_times_new = (np.array(self.spike_times) - value).tolist()
        self.l.set_data(get_mean_image(self.trials, spike_times_new, self.image_array))
        self.slider.valtext.set_text('{:03.3f}'.format(value))
        self.fig.canvas.draw()

    def onpick(self, event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            mouse_event = event.mouseevent
            self.x = int(np.round(mouse_event.xdata))
            self.y = int(np.round(mouse_event.ydata))
            for i, time in enumerate(self.t):
                spike_times_new_pick = (np.array(self.spike_times) + time).tolist()
                y_val = get_mean_point(self.x, self.y, self.trials, spike_times_new_pick, self.image_array)
                self.ydata[i] = np.mean(y_val)
            self.l2.set_data(self.t, self.ydata)
            self.ax2.set_title('Spike-Triggered Average for Point [%d, %d]' % (self.x, self.y))
            self.ax2.set_xlabel('Time Before Spike (seconds)')
            self.ax2.set_ylabel('Spike-Triggered Average')
            self.ax2.relim()
            self.ax2.autoscale_view(True, True, True)
            self.fig2.canvas.draw()

    def show(self):
        plt.show()
