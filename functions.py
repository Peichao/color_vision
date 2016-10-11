import glob
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import AxesImage

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')


def get_params(file_path):
    """
    Input path to mat file from stimulus computer that contains trial sequence info.
    :param file_path: String. File path for mat file from stimulus computer.
    :return: pandas.DataFrame. DataFrame containing all stimulus sequence information.
    """
    exp_params_all = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

    # Create list of all trials from mat file.
    r_seed = []
    for key in exp_params_all:
        if key.startswith('randlog'):
            r_seed.append(key)

    # Pull all trial information into DataFrame. Includes all Hartley information and stimulus size
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
    """
    Build NumPy array of image files using raw image files created with get_images.m.
    :param image_path: String. Path to image files created with get_images.m.
    :return: numpy.ndarray. Array of raw image files in order.
    """
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
    """
    Return DataFrame of unique trials from all trials. Can be used to check if certain trial numbers are missing.
    :param params_path: String. Path to mat file of parameters from stimulus computer.
    :return: pandas.DataFrame. DataFrame containing all stimulus sequence information with duplicate trials removed.
    """
    trials = get_params(params_path)
    trials_unique = trials.drop_duplicates()

    return trials_unique


def get_image_idx(trials, spike_times):
    """
    Return index of image presented at all specified spike times.
    AG: modified to use pandas.searchsorted for speed.
    :param trials: pandas.DataFrame. All stimulus information.
    :param spike_times: np.ndarray. All spike times.
    :return: pandas.Series. Image index (out of all stimuli possible) at each spike time from spike_times.
    """
    stim_idx = trials.stim_time.searchsorted(spike_times, side='right') - 1
    image_idx = trials.cond[stim_idx]
    return image_idx


def get_mean_image(trials, spike_times, image_array):
    """
    Get mean image at specified spike times.
    :param trials: pandas.DataFrame. All stimulus information including stimulus presented.
    :param spike_times: np.ndarray. All spike times.
    :param image_array: np.ndarray. Raw images for all possible stimuli.
    :return: np.ndarray. Mean image.
    """
    image_idx = get_image_idx(trials, spike_times)
    mean_image = np.mean(image_array[:, :, :, image_idx], axis=3)
    return mean_image


def get_mean_point(x, y, trials, spike_times, image_array):
    """
    Get mean point (x, y) of all images at specified spike times.
    :param x: Integer. X coordinate.
    :param y: Integer. Y coordinate.
    :param trials: pandas.DataFrame. All stimulus information including stimulus presented.
    :param spike_times: np.ndarray. All spike times.
    :param image_array: np.ndarray. Raw images for all possible stimuli.
    :return: Float. Mean point.
    """
    image_idx = get_image_idx(trials, spike_times)
    mean_point = np.mean(image_array[y, x, :, image_idx], axis=0)
    return mean_point


def get_stim_samples_fh(data_path):
    """
    Get starting sample index of each 20ms pulse from photodiode.
    :param file_path: String. Path to binary file from recording.
    :return: np.ndarray. Starting times of each stimulus (1 x nStimuli).
    """
    data = (np.memmap(data_path, np.int16, mode='r').reshape(-1, 129)).T
    data = pd.DataFrame(data[128, :], columns=['photodiode'])

    hi_cut = 1500
    lo_cut = 800

    from detect_peaks import detect_peaks
    peaks = detect_peaks(data.photodiode, mph=lo_cut, mpd=200)

    # import peakutils as peakutils
    # peaks = peakutils.indexes(data.photodiode, thres=(1100/data.photodiode.max()), min_dist=200)

    peaks_high = peaks[np.where(data.photodiode[peaks] > hi_cut)]
    peaks_low = peaks[np.where(np.logical_and(data.photodiode[peaks] > lo_cut, data.photodiode[peaks] < hi_cut))]

    # In case of signal creep, remove all low peaks between trials
    # Edited to keep peaks due to stuck frames and only remove if final peak to peak distance > 500 (due to end pulse)
    trial_break_idx = np.where(np.diff(peaks_high) > 5000)[0]
    ind_delete = []
    for i, idx in enumerate(trial_break_idx):
        if peaks_high[idx] - peaks_high[idx - 1] < 500:
            ind_delete.append(i)
        else:
            peaks_low_delete = np.where(np.logical_and(peaks_low > peaks_high[idx], peaks_low < peaks_high[idx+1]))
            peaks_low = np.delete(peaks_low, peaks_low_delete)
    trial_break_idx = np.delete(trial_break_idx, ind_delete)

    # Remove pulses from beginning and end of trial (onset and end pulses)
    trial_break_idx = np.sort(np.concatenate([trial_break_idx, trial_break_idx + 1]))

    peaks_high = np.delete(peaks_high, trial_break_idx)[1:-1]
    peaks = np.sort(np.concatenate([peaks_high, peaks_low]))

    # Find rising edges of photodiode and mark as trial beginning and end
    data_peaks = pd.DataFrame()
    data_peaks['photodiode'] = data.photodiode[peaks]

    data_peaks['high'] = data_peaks.photodiode > 1500
    high = data_peaks.index[data_peaks['high'] & ~ data_peaks['high'].shift(1).fillna(False)]

    data_peaks['low'] = data_peaks.photodiode < 1500
    low = data_peaks.index[data_peaks['low'] & ~ data_peaks['low'].shift(1).fillna(False)]

    stim_times = np.sort(np.concatenate([high, low]))
    return stim_times


def get_stim_samples_pg(data_path, start_time, end_time):
    data = (np.memmap(data_path, np.int16, mode='r').reshape(-1, 129)).T

    if end_time.values[0] > 0:
        data = pd.DataFrame(data[128, start_time * 60 * 25000:end_time * 60 * 25000], columns=['photodiode'])
    else:
        data = pd.DataFrame(data[128, :], columns=['photodiode'])

    from detect_peaks import detect_peaks
    peaks = detect_peaks(data.photodiode, mph=1500, mpd=200)
    fst = np.array([peaks[1]])
    stim_times_remaining = peaks[np.where(np.diff(peaks) > 10000)[0]] + 1
    stim_times = np.append(fst, stim_times_remaining)
    # dropped = np.where(np.diff(peaks) < 500)[0]
    # peaks_correct = np.delete(peaks, dropped + 1)
    # stim_times = peaks_correct[1::19]
    return stim_times


def analyzer_pg(analyzer_path):
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    b_flag = 0
    if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol[0] == 'blank':
        b_flag = 1

    if b_flag == 0:
        trial_num = np.zeros((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats),
                             (len(analyzer.loops.conds[0].symbol))))
    else:
        trial_num = np.zeros(((len(analyzer.loops.conds) - b_flag) * len(analyzer.loops.conds[0].repeats) +
                              len(analyzer.loops.conds[-1].repeats),
                              (len(analyzer.loops.conds[0].symbol))))

    for count in range(0, len(analyzer.loops.conds) - b_flag):
        trial_vals = np.zeros(len(analyzer.loops.conds[count].symbol))

        for count2 in range(0, len(analyzer.loops.conds[count].symbol)):
            trial_vals[count2] = analyzer.loops.conds[count].val[count2]

        for count3 in range(0, len(analyzer.loops.conds[count].repeats)):
            aux_trial = analyzer.loops.conds[count].repeats[count3].trialno
            trial_num[aux_trial - 1, :] = trial_vals

    for blank_trial in range(0, len(analyzer.loops.conds[-1].repeats)):
        aux_trial = analyzer.loops.conds[-1].repeats[blank_trial].trialno
        trial_num[aux_trial - 1, :] = np.array([256, 256])

    stim_time = np.zeros(3)
    for count4 in range(0, 3):
        stim_time[count4] = analyzer.P.param[count4][2]

    return trial_num, stim_time


def jrclust_csv(csv_path):
    sp = pd.read_csv(csv_path, names=['time', 'cluster', 'max_site'])
    return sp


def kilosort_info(data_folder):
    sp = pd.DataFrame()
    sp['time'] = np.load(data_folder + 'spike_times.npy').flatten() / 25000
    sp['cluster'] = np.load(data_folder + 'spike_clusters.npy')
    return sp


class PlotRF(object):
    """
    Use reverse correlation to plot average stimulus preceding spikes from each unit.
    """
    def __init__(self, trials=None, spike_times=None, image_array=None):
        """

        :param trials: pandas.DataFrame. Contains stimulus times and Hartley stimulus information.
        :param spike_times: np.ndarray. Contains all spike times (1 x nSpikes).
        :param image_array: np.ndarray. Contains all image data (height x width x 3 x nStimuli).
        """
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
        self.spike_times = spike_times[(spike_times > trials.stim_time.min() + 0.2) &
                                       (spike_times < trials.stim_time.max())]
        self.image_array = image_array
        self.starting_image = get_mean_image(self.trials, self.spike_times, self.image_array)

        self.l = self.ax.imshow(self.starting_image, picker=True)

        self.t = np.arange(-0.2, 0, 0.001)
        self.ydata = np.zeros(np.size(self.t))
        self.l2, = self.ax2.plot([], [], 'r-')
        self.ax2.set_xlim([-0.20, 0])

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

    def update(self, value):
        """

        :param value: Float. Value from slider on image.
        :return: New image presented containing shifted time window.
        """
        spike_times_new = self.spike_times - value
        self.l.set_data(get_mean_image(self.trials, spike_times_new, self.image_array))
        self.slider.valtext.set_text('{:03.3f}'.format(value))
        self.fig.canvas.draw()

    def onpick(self, event):
        """

        :param event: Mouse click on image.
        :return: New data on figure l2 displaying average contrast over time at point clicked during event.
        """
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
        """

        :return: Image displayed.
        """
        plt.show()
