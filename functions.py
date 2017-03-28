import os
import glob
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sp
from scipy import stats
import scipy.io as sio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import AxesImage
from matplotlib.backends.backend_pdf import PdfPages
import csd_analysis
import waveform

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')

probe_geo = np.zeros([128, 2])
probe_geo[1::2, 0] = 1
probe_geo[0::2, 1] = np.arange(64)
probe_geo[1::2, 1] = probe_geo[0::2, 1]
ref_sites_idx = np.array([1, 18, 33, 50, 65, 82, 97, 114]) - 1
ref_sites = probe_geo[ref_sites_idx, :]
probe_geo = np.delete(probe_geo, ref_sites_idx, axis=0)


def get_params(file_path, analyzer_path):
    """
    Input path to mat file from stimulus computer that contains trial sequence info.
    :param file_path: String. File path for mat file from stimulus computer.
    :return: pandas.DataFrame. DataFrame containing all stimulus sequence information.
    """
    exp_params_all = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    x_size = analyzer.P.param[5][2]
    y_size = analyzer.P.param[6][2]

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
        trials -= 1

        for column in columns:
            trials[column] = trials.cond.map(conds[column])

        trials['ori'] = np.degrees(np.arctan(trials.ky / trials.kx)) % 360
        trials['sf'] = np.sqrt((trials.kx/x_size)**2 + (trials.ky/y_size)**2)

        all_trials = pd.concat([all_trials, trials], axis=0, ignore_index=True)

    return all_trials


def stim_size_pixels(analyzer_path):
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    x_size = analyzer.P.param[5][2]
    y_size = analyzer.P.param[6][2]

    screen_xcm = analyzer.M.screenXcm
    screen_ycm = analyzer.M.screenYcm

    x_pixels = analyzer.M.xpixels
    y_pixels = analyzer.M.ypixels

    screen_dist = analyzer.M.screenDist

    pix_per_xcm = x_pixels / screen_xcm
    pix_per_ycm = y_pixels / screen_ycm

    x_cm = 2 * screen_dist * np.tan(x_size / 2 * np.pi / 180) + 0.1
    xN = np.round(x_cm * pix_per_xcm)

    y_cm = 2 * screen_dist * np.tan(y_size / 2 * np.pi / 180) + 0.1
    yN = np.round(y_cm * pix_per_ycm)

    return xN, yN


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


def get_image_info(trials, spike_times):
    """
    Return index of image presented at all specified spike times.
    AG: modified to use pandas.searchsorted for speed.
    :param trials: pandas.DataFrame. All stimulus information.
    :param spike_times: np.ndarray. All spike times.
    :return: pandas.Series. Image index (out of all stimuli possible) at each spike time from spike_times.
    """
    stim_idx = trials.stim_time.searchsorted(spike_times, side='right') - 1
    image_kx = trials.kx[stim_idx].as_matrix()
    image_ky = trials.ky[stim_idx].as_matrix()

    return image_kx, image_ky


def hart_transform(kx, ky, xN, yN):
    nx = np.arange(xN)
    ny = np.arange(yN)
    nxmat, nymat = np.meshgrid(nx, ny)

    alpha = (2 * np.pi * kx * nxmat / xN) + (2 * np.pi * ky * nymat / yN)
    hart_image = np.sin(alpha) + np.cos(alpha)
    return hart_image


def get_mean_image(trials, spike_times, image_array):
    """
    Get mean image at specified spike times.
    :param trials: pandas.DataFrame. All stimulus information including stimulus presented.
    :param spike_times: np.ndarray. All spike times.
    :param image_array: np.ndarray. Raw images for all possible stimuli.
    :return: np.ndarray. Mean image.
    """
    # image_kx, image_ky = get_image_info(trials, spike_times)
    # mean_image = np.mean(image_array[:, :, :, image_idx], axis=3)
    # return mean_image


def get_mean_point(x, y, trials, spike_times):
    """
    Get mean point (x, y) of all images at specified spike times.
    :param x: Integer. X coordinate.
    :param y: Integer. Y coordinate.
    :param trials: pandas.DataFrame. All stimulus information including stimulus presented.
    :param spike_times: np.ndarray. All spike times.
    :param image_array: np.ndarray. Raw images for all possible stimuli.
    :return: Float. Mean point.
    """
    # kx, ky = get_image_info(trials, spike_times)
    # mean_point = np.mean(image_array[y, x, :, image_idx], axis=0)
    # return mean_point


def get_stim_samples_fh(data_path, start_time, end_time=None):
    """
    Get starting sample index of each 20ms pulse from photodiode.
    :param file_path: String. Path to binary file from recording.
    :return: np.ndarray. Starting times of each stimulus (1 x nStimuli).
    """
    data = (np.memmap(data_path, np.int16, mode='r').reshape(-1, 129)).T
    if end_time is not None:
        data = pd.DataFrame(data[128, start_time * 25000:end_time * 25000], columns=['photodiode'])
    else:
        data = pd.DataFrame(data[128, start_time * 25000:], columns=['photodiode'])

    hi_cut = 1500
    lo_cut = 600

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

    stim_times = np.sort(np.concatenate([high, low])) + (start_time * 25000)

    data_folder = os.path.dirname(data_path)
    np.save(data_folder + '/stim_samples.npy', stim_times)
    return stim_times


def get_stim_samples_pg(data_path, start_time, end_time=None):
    data = (np.memmap(data_path, np.int16, mode='r').reshape(-1, 129)).T

    if end_time is not None:
        data = pd.DataFrame(data[128, start_time * 60 * 25000:end_time * 60 * 25000], columns=['photodiode'])
    else:
        data = pd.DataFrame(data[128, :], columns=['photodiode'])

    from detect_peaks import detect_peaks
    peaks = detect_peaks(data.photodiode, mph=1500, mpd=200)
    fst = np.array([peaks[0]])
    stim_times_remaining = peaks[np.where(np.diff(peaks) > 10000)[0] + 1]
    stim_times = np.append(fst, stim_times_remaining)
    stim_times += (start_time * 60 * 25000)
    # dropped = np.where(np.diff(peaks) < 500)[0]
    # peaks_correct = np.delete(peaks, dropped + 1)
    # stim_times = peaks_correct[1::19]

    np.save(os.path.dirname(data_path) + '/stim_samples.npy', stim_times)

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
    def __init__(self, trials=None, spike_times=None, analyzer_path=None):
        """

        :param trials: pandas.DataFrame. Contains stimulus times and Hartley stimulus information.
        :param spike_times: np.ndarray. Contains all spike times (1 x nSpikes).
        :param image_array: np.ndarray. Contains all image data (height x width x 3 x nStimuli).
        """
        self.analyzer_path = analyzer_path
        self.xN, self.yN = stim_size_pixels(self.analyzer_path)

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

        for i in np.where(np.diff(trials.stim_time) > 0.3)[0]:
            beginning = trials.loc[i, 'stim_time']
            end = trials.loc[i+1, 'stim_time'] + 0.2
            spike_times = spike_times[(spike_times < beginning) |
                                      (spike_times > end)]

        self.image_kx, self.image_ky = get_image_info(self.trials, self.spike_times)

        self.starting_image = np.zeros([self.xN.astype(int), self.yN.astype(int)])
        for i in np.arange(self.image_kx.size):
            hart_image = hart_transform(self.image_kx[i], self.image_ky[i], self.xN, self.yN)
            self.starting_image += hart_image

        self.starting_image /= self.image_kx.size

        self.l = self.ax.imshow(self.starting_image[self.xN/4:self.xN*3/4, self.yN/4:self.yN*3/4], cmap='jet', vmin=-1.414, vmax=1.414, picker=True)

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
        image_kx_new, image_ky_new = get_image_info(self.trials, spike_times_new)

        new_image = np.zeros([self.xN.astype(int), self.yN.astype(int)])

        for i in np.arange(image_kx_new.size):
            hart_image = hart_transform(image_kx_new[i], image_ky_new[i], self.xN, self.yN)
            new_image += hart_image
        new_image /= image_kx_new.size

        self.l.set_data(new_image[self.xN/4:self.xN*3/4, self.yN/4:self.yN*3/4])
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
                image_kx_new_pick, image_ky_new_pick = get_image_info(self.trials, spike_times_new_pick)
                mean_point = 0

                for j in np.arange(image_kx_new_pick.size):
                    hart_image = hart_transform(image_kx_new_pick[j], image_ky_new_pick[j], self.xN, self.yN)
                    mean_point += hart_image[self.x, self.y]
                mean_point /= image_kx_new_pick.size

                # y_val = get_mean_point(self.x, self.y, self.trials, spike_times_new_pick)
                self.ydata[i] = mean_point
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


class GetLayers(object):
    """
    Plot CSD in interactive manner and allow clicks to get bottom and top of layer 4C.
    """
    # TODO write code to get bottom and top of granular layers

    def __init__(self, csd_data=None):
        self.csd_data = csd_data
        self.fig, self.ax = plt.subplots()

        self.ax.imshow(csd_data[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto', picker=True)
        self.ax.set_xlabel('Time (ms)')
        self.ax.set_ylabel('Channel (odd)')

        self.x = None
        self.y = None
        self.xcoords = []
        self.ycoords = []

        self.l = self.fig.canvas.mpl_connect('pick_event', self.onpick)

    def onpick(self, event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            if len(self.xcoords) == 0:
                print('Picked lower boundary of Layer 4.')
            else:
                print('Picked upper boundary of Layer 4.')
            mouse_event = event.mouseevent
            self.x = int(np.round(mouse_event.xdata))
            self.y = int(np.round(mouse_event.ydata))

            self.xcoords.append(self.x)
            self.ycoords.append(self.y)

            if len(self.xcoords) == 2:
                self.fig.canvas.mpl_disconnect(self.l)
                plt.close()

    def show(self):
        plt.show()


def xcorr(a, b, maxlag=50, spacing=1):
    """
    Computes cross-correlation of spike times held in vector a and b with shift ms precision
    :param a: first array to be compared
    :param b: second array to be compared
    :param maxlag: maximum time shift to compute
    :param spacing: spacing between time shifts
    :return: index of time shifts and array of number of occurrences
    """
    xcorr_series = pd.Series(0, index=np.arange(-maxlag, maxlag + spacing))

    for i in np.arange(0, maxlag+(spacing*2), spacing):
        if i == 0:
            common = np.intersect1d(a, b).size
            xcorr_series.loc[i] = common
        else:
            bins_sub = np.concatenate((a - (i-spacing), a-i))
            bins_sub.sort()
            bins_add = np.concatenate((a + (i-spacing), a+i+0.0001))
            bins_add.sort()

            hist_sub = np.histogram(b, bins_sub)[0][0::2].sum()
            hist_add = np.histogram(b, bins_add)[0][0::2].sum()

            xcorr_series.loc[-i] = hist_sub
            xcorr_series.loc[i-spacing] = hist_add

    return xcorr_series


def xcorr_spiketime_all(data_folder, sp, maxlag=50, spacing=1, orisf=False, trial_num=None, stim_time=None):
    """
    Computes cross-correlation using spike times, computes all possible combinations between many pairs of neurons.
    :param sp: pandas.DataFrame of all spike times
    :param maxlag: maximum time shift to compute
    :param spacing: spacing between time shifts
    :return:
        dictionary of correlation, dictionary of indexes
    """
    from itertools import combinations

    clusters = sp[sp.cluster > 0].cluster.unique()
    clusters.sort()
    comb = list(combinations(clusters, 2))

    xcorr_df = pd.DataFrame()
    csd_avg = csd_analysis.csd_analysis()
    boundaries = GetLayers(csd_avg)
    boundaries.show()
    probe_geo_lims = np.searchsorted(probe_geo[:, 1], boundaries.ycoords)

    waveforms = waveform.waveforms(data_folder)

    if not os.path.exists(data_folder + '/_images'):
        os.makedirs(data_folder + '/_images')

    for idx, i in enumerate(comb):
        print('Now calculating combination ' + str(i))
        xcorr_series = xcorr(sp[sp.cluster == i[0]].time * 1000, sp[sp.cluster == i[1]].time * 1000, maxlag, spacing)
        xcorr_median = xcorr_series.median()
        xcorr_df[i] = xcorr_series
        mid_points = xcorr_df[i].index[len(xcorr_df[i].index) * 19 / 40:len(xcorr_df[i].index) * 21 / 40]
        corr_zscore = (xcorr_df[i].loc[mid_points].max() - xcorr_df[i].mean()) / xcorr_df[i].std()

        # if (xcorr_median != 0) & ((xcorr_df[i].loc[mid_points] > (xcorr_median * 3)).sum() > 0):
        if corr_zscore > 3:
            xcorr_fig(data_folder, sp, xcorr_df[i], csd_avg, waveforms, i, idx, probe_geo_lims,
                      orisf, trial_num, stim_time)

    return xcorr_df


def xcorr_fig(data_folder, sp, xcorr_series, csd_avg, waveforms, comb, idx, probe_geo_lims,
              orisf=False, trial_num=None, stim_time=None):
    plt.ioff()
    with PdfPages(data_folder + '_images/xcorr_fig%d.pdf' % idx) as pdf:
        fig = plt.figure()
        grid_geo = (6, 6)
        ax1 = plt.subplot2grid(grid_geo, (0, 0), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid(grid_geo, (0, 5), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid(grid_geo, (3, 0), rowspan=3, colspan=6)
        ax4 = plt.subplot2grid(grid_geo, (0, 3), rowspan=3, colspan=1)
        ax5 = plt.subplot2grid(grid_geo, (0, 4), rowspan=3, colspan=1)
        fig.set_size_inches(8.5, 11)

        colors = ['#E24A33', '#348ABD']

        ax1.imshow(csd_avg[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Channel (odd)')

        ax2.set_xlim([0, 2])
        ax2.set_xticks(np.arange(2))
        ax2.set_yticks(np.arange(64))
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax2.grid(True)

        import matplotlib.patches as patches
        max_site_1 = int(sp[sp.cluster == comb[0]].max_site.mode())
        max_site_2 = int(sp[sp.cluster == comb[1]].max_site.mode())
        max_sites = (max_site_1, max_site_2)

        max_locations = np.searchsorted(probe_geo_lims, max_sites)
        location_dict = {0: 'infragranular',
                         1: 'granular',
                         2: 'supragranular'}

        for i in np.arange(ref_sites.shape[0]):
            ax2.add_patch(patches.Rectangle((ref_sites[i, 0], ref_sites[i, 1]), 1, 1, color='k'))

        for i, chan in enumerate(max_sites):
            ax2.add_patch(patches.Rectangle((probe_geo[chan - 1, 0], probe_geo[chan - 1, 1]), 1, 1, color=colors[i]))

        ax1.set_title('Cross Correlogram for %s' % str(comb), fontsize=12)

        ax3.bar(xcorr_series.index, xcorr_series, width=1)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Number of Spikes')
        ax3.set_xlim([-1*xcorr_series.index.max(), xcorr_series.index.max()])
        ax3.xaxis.set_ticks(xcorr_series.index[::5])

        norm_plots = {}
        for j in np.arange(2):
            waveform = waveforms[comb[j] - 1, :, :]
            norm_plots[j] = np.transpose((waveform - waveform.min()) / waveform.ptp()) + probe_geo[:, 1]
            ax4.plot(np.arange(0, 39) / 25, norm_plots[j][:, np.where(probe_geo[:, 0] == 0)[0]], color=colors[j])
            ax5.plot(np.arange(0, 39) / 25, norm_plots[j][:, np.where(probe_geo[:, 0] == 1)[0]], color=colors[j])

        ax4.set_title(location_dict[max_locations[0]], fontsize=12)
        ax5.set_title(location_dict[max_locations[1]], fontsize=12)
        ax4.set_xlim([0, 1.5])
        ax4.set_ylim([0, 64])
        ax5.set_xlim([0, 1.5])
        ax5.set_ylim([0, 64])
        ax4.xaxis.set_ticks([0, 1.5])
        ax5.xaxis.set_ticks([0, 1.5])

        ax4.set_xlabel('Time (ms)')
        ax5.set_xlabel('Time (ms)')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        if orisf:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.set_size_inches(8.5, 11)
            color_list = ['#E24A33', '#348ABD']

            for i in np.arange(2):
                cluster = comb[i]
                trial_num_cluster = orisf_plot(data_folder, sp, cluster, trial_num, stim_time)
                cluster_str = 'cluster' + str(cluster)

                counts_ori = trial_num_cluster.groupby(['orientation', 's_frequency'])[cluster_str].mean()
                counts_sf = trial_num_cluster.groupby(['s_frequency', 'orientation'])[cluster_str].mean()
                sem_ori = trial_num_cluster.groupby(['orientation', 's_frequency'])[cluster_str].sem()
                sem_sf = trial_num_cluster.groupby(['s_frequency', 'orientation'])[cluster_str].sem()

                s_freq_max = counts_ori.drop(256).unstack(level=1).mean(axis=0).idxmax()
                orientation_max = counts_ori.drop(256).unstack(level=1).mean(axis=1).idxmax()

                fig.suptitle("Cluster %d, Site %d" % (cluster, sp[sp.cluster == cluster].max_site.mode()[0]),
                             fontsize=16)

                counts_ori.drop(256)[orientation_max].plot(ax=ax1, linewidth=2, color=color_list[i])
                ax1.fill_between(counts_ori.drop(256)[orientation_max].index,
                                 counts_ori.drop(256)[orientation_max] - sem_ori.drop(256)[orientation_max],
                                 counts_ori.drop(256)[orientation_max] + sem_ori.drop(256)[orientation_max],
                                 color=color_list[i], alpha=0.5)

                counts_sf.drop(256)[s_freq_max].plot(ax=ax2, linewidth=2, color=color_list[i])
                ax2.fill_between(counts_sf.drop(256)[s_freq_max].index,
                                 counts_sf.drop(256)[s_freq_max] - sem_sf.drop(256)[s_freq_max],
                                 counts_sf.drop(256)[s_freq_max] + sem_sf.drop(256)[s_freq_max],
                                 color=color_list[i], alpha=0.5)

            ax1.set_title('Spatial Frequency Tuning')
            ax1.set_ylabel('Spike Rate (Hz)')
            ax1.set_xlabel('Spatial Frequency (cycles/degree)')

            ax2.set_title('Orientation Tuning')
            ax2.set_ylabel('Spike Rate (Hz)')
            ax2.set_xlabel('Orientation (degrees)')

            fig.tight_layout()
            fig.subplots_adjust(right=0.8)
            fig.subplots_adjust(top=0.88)

            pdf.savefig(fig)
            plt.close()


def xcorr_maxlag(x, y, maxlag):
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    t = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return t.dot(px)


def xcorr_psth_all(data_folder, spikes, trial_num, stim_time, bin_size=0.001, maxlags=200, plot_lags=50):
    connect_array = np.zeros([3, 3])

    def centered(arr, num):
        centered_arr = arr[int(np.round(arr.size / 2)) - num: int(np.round(arr.size / 2)) + num + 1]
        return centered_arr

    def z_score(value, mean, std):
        z = (value - mean) / std
        return z

    csd_avg = csd_analysis.csd_analysis()
    boundaries = GetLayers(csd_avg)
    boundaries.show()
    probe_geo_lims = np.searchsorted(probe_geo[:, 1], boundaries.ycoords)
    waveforms = waveform.waveforms(data_folder)

    num_bins = int(np.sum(stim_time) / bin_size)
    num_trials = trial_num.stim_start.size
    num_clusters = (spikes.cluster.unique() > 0).sum()
    psth = np.zeros((num_trials, num_bins, num_clusters))

    for cluster in range(num_clusters):
        sp_cluster = spikes[spikes.cluster == cluster + 1]
        print('Analyzing cluster %d.' % (cluster + 1))
        for i in range(trial_num.stim_start.size):
            pre_time = trial_num.stim_start[i] - stim_time[0]
            end_time = trial_num.stim_start[i] + stim_time[1] + stim_time[2]
            sp_cluster_trial = sp_cluster[(sp_cluster.time > trial_num.stim_start[i]) & (sp_cluster.time < trial_num.stim_start[i] + stim_time[2])]
            hist, bin_edges = np.histogram(sp_cluster_trial.time, bins=num_bins, range=(pre_time, end_time))
            psth[i, :, cluster] = hist

    from itertools import combinations

    # psth_roll = np.roll(psth, 1, axis=0)
    psth_shuffle = np.take(psth, np.random.rand(psth.shape[0]).argsort(), axis=0)

    clusters = spikes[spikes.cluster > 0].cluster.unique()
    clusters.sort()
    comb = list(combinations(clusters, 2))

    for idx, i in enumerate(comb):
        print('Now calculating combination ' + str(i))

        # xc = np.zeros(maxlags * 2 + 1)
        # predictor = np.zeros(maxlags * 2 + 1)
        # for trial in range(psth.shape[0]):
        #     xc_trial = centered(np.correlate(psth[trial, :, i[1]-1], psth[trial, :, i[0]-1], 'full'), maxlags)
        #     xc += xc_trial
        #
        #     predictor_trial = centered(np.correlate(psth[trial, :, i[1]-1], psth_shuffle[trial, :, i[0]-1], 'full'),
        #                                maxlags)
        #     predictor += predictor_trial

        xc = xcorr_maxlag(psth[:, :, i[1]-1].ravel(), psth[:, :, i[0]-1].ravel(), maxlags)
        predictor = xcorr_maxlag(psth[:, :, i[1] - 1].ravel(), psth_shuffle[:, :, i[0] - 1].ravel(), maxlags)

        x = np.arange(-plot_lags, plot_lags + 1, 1)

        # pst_1 = psth[:, :, i[1]-1].sum(axis=0)
        # pst_2 = psth[:, :, i[0]-1].sum(axis=0)
        # predictor = np.correlate(pst_1, pst_2, 'full') / num_trials
        # predictor = centered(predictor, maxlags)

        corrected = xc - predictor

        predictor_std = np.std(centered(predictor, 50))
        xc_mean_pre = np.mean(xc[int(np.round(xc.size / 2)) - 50: int(np.round(xc.size / 2)) - 10])
        xc_mean_post = np.mean(xc[int(np.round(xc.size / 2)) + 10: int(np.round(xc.size / 2)) + 50])
        xc_mean = np.mean([xc_mean_pre, xc_mean_post])

        z_scores = np.zeros(6)
        p_vals = np.zeros(6)

        for shifter in range(1, 4):
            p_vals[(-shifter) + 3] = sp.stats.poisson.pmf(xc[int(np.round(xc.size / 2)) - shifter],
                                                          predictor[int(np.round(predictor.size / 2)) - shifter])
            p_vals[shifter + 2] = sp.stats.poisson.pmf(xc[int(np.round(xc.size / 2)) + shifter],
                                                       predictor[int(np.round(predictor.size / 2)) + shifter])

            z_scores[(-shifter) + 3] = z_score(xc[int(np.round(xc.size / 2)) - shifter], xc_mean, predictor_std)
            z_scores[shifter + 2] = z_score(xc[int(np.round(xc.size / 2)) + shifter], xc_mean, predictor_std)

        # if any((i < 0.001) & (j < 0.001) for i, j in zip(p_vals, p_vals[1:])):
        #     xcorr_fig_psth(data_folder, spikes, x, centered(corrected, plot_lags), csd_avg, waveforms, i, idx,
        #                    probe_geo_lims, trial_num, stim_time)

        if z_scores.max() > 4.44:
            max_locations = xcorr_fig_psth(data_folder, spikes, x, centered(xc, plot_lags), csd_avg, waveforms, i, idx,
                                           probe_geo_lims, trial_num, stim_time)
            if np.argmax(z_scores) < 4:
                connect_array[max_locations[1], max_locations[0]] += 1
            else:
                connect_array[max_locations[0], max_locations[1]] += 1

    return connect_array


def xcorr_fig_psth(data_folder, sp, x, xcorr, csd_avg, waveforms, comb, idx, probe_geo_lims,
                   trial_num, stim_time):
    plt.ioff()
    with PdfPages(data_folder + '_images/xcorr_fig%d.pdf' % idx) as pdf:
        fig = plt.figure()
        grid_geo = (6, 6)
        ax1 = plt.subplot2grid(grid_geo, (0, 0), rowspan=3, colspan=3)
        ax2 = plt.subplot2grid(grid_geo, (0, 5), rowspan=3, colspan=1)
        ax3 = plt.subplot2grid(grid_geo, (3, 0), rowspan=3, colspan=6)
        ax4 = plt.subplot2grid(grid_geo, (0, 3), rowspan=3, colspan=1)
        ax5 = plt.subplot2grid(grid_geo, (0, 4), rowspan=3, colspan=1)
        fig.set_size_inches(8.5, 11)

        colors = ['#E24A33', '#348ABD']

        ax1.imshow(csd_avg[:, 0:100], interpolation='bicubic', origin='lower', aspect='auto')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Channel (odd)')

        ax2.set_xlim([0, 2])
        ax2.set_xticks(np.arange(2))
        ax2.set_yticks(np.arange(64))
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax2.grid(True)

        import matplotlib.patches as patches
        max_site_1 = int(sp[sp.cluster == comb[0]].max_site.mode())
        max_site_2 = int(sp[sp.cluster == comb[1]].max_site.mode())
        max_sites = (max_site_1, max_site_2)

        max_locations = np.searchsorted(probe_geo_lims, max_sites)
        location_dict = {0: 'infragranular',
                         1: 'granular',
                         2: 'supragranular'}

        for i in np.arange(ref_sites.shape[0]):
            ax2.add_patch(patches.Rectangle((ref_sites[i, 0], ref_sites[i, 1]), 1, 1, color='k'))

        for i, chan in enumerate(max_sites):
            ax2.add_patch(
                patches.Rectangle((probe_geo[chan - 1, 0], probe_geo[chan - 1, 1]), 1, 1, color=colors[i]))

        ax1.set_title('Cross Correlogram for %s' % str(comb), fontsize=12)

        ax3.bar(x, xcorr, width=1)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Number of Spikes')
        ax3.set_xlim([x[0], x[1]])
        ax3.xaxis.set_ticks(x[::5])

        norm_plots = {}
        for j in np.arange(2):
            waveform = waveforms[comb[j] - 1, :, :]
            norm_plots[j] = np.transpose((waveform - waveform.min()) / waveform.ptp()) + probe_geo[:, 1]
            ax4.plot(np.arange(waveform.shape[1]) / 25, norm_plots[j][:, np.where(probe_geo[:, 0] == 0)[0]],
                     color=colors[j])
            ax5.plot(np.arange(waveform.shape[1]) / 25, norm_plots[j][:, np.where(probe_geo[:, 0] == 1)[0]],
                     color=colors[j])

        ax4.set_title(location_dict[max_locations[0]], fontsize=12)
        ax5.set_title(location_dict[max_locations[1]], fontsize=12)
        ax4.set_xlim([0, 1.5])
        ax4.set_ylim([0, 64])
        ax5.set_xlim([0, 1.5])
        ax5.set_ylim([0, 64])
        ax4.xaxis.set_ticks([0, 1.5])
        ax5.xaxis.set_ticks([0, 1.5])

        ax4.set_xlabel('Time (ms)')
        ax5.set_xlabel('Time (ms)')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(8.5, 11)
        color_list = ['#E24A33', '#348ABD']

        for i in np.arange(2):
            cluster = comb[i]
            trial_num_cluster = orisf_plot(data_folder, sp, cluster, trial_num, stim_time)
            cluster_str = 'cluster' + str(cluster)

            counts_ori = trial_num_cluster.groupby(['orientation', 's_frequency'])[cluster_str].mean()
            counts_sf = trial_num_cluster.groupby(['s_frequency', 'orientation'])[cluster_str].mean()
            sem_ori = trial_num_cluster.groupby(['orientation', 's_frequency'])[cluster_str].sem()
            sem_sf = trial_num_cluster.groupby(['s_frequency', 'orientation'])[cluster_str].sem()

            s_freq_max = counts_ori.drop(256).unstack(level=1).mean(axis=0).idxmax()
            orientation_max = counts_ori.drop(256).unstack(level=1).mean(axis=1).idxmax()

            # fig.suptitle(
            #     "Cluster %d, Site %d" % (cluster, sp[sp.cluster == cluster].max_site.mode()[0]),
            #     fontsize=16)

            counts_ori.drop(256)[orientation_max].plot(ax=ax1, linewidth=2, color=color_list[i])
            ax1.fill_between(counts_ori.drop(256)[orientation_max].index,
                             counts_ori.drop(256)[orientation_max] - sem_ori.drop(256)[orientation_max],
                             counts_ori.drop(256)[orientation_max] + sem_ori.drop(256)[orientation_max],
                             color=color_list[i], alpha=0.5)

            counts_sf.drop(256)[s_freq_max].plot(ax=ax2, linewidth=2, color=color_list[i])
            ax2.fill_between(counts_sf.drop(256)[s_freq_max].index,
                             counts_sf.drop(256)[s_freq_max] - sem_sf.drop(256)[s_freq_max],
                             counts_sf.drop(256)[s_freq_max] + sem_sf.drop(256)[s_freq_max],
                             color=color_list[i], alpha=0.5)

        ax1.set_title('Spatial Frequency Tuning')
        ax1.set_ylabel('Spike Rate (Hz)')
        ax1.set_xlabel('Spatial Frequency (cycles/degree)')

        ax2.set_title('Orientation Tuning')
        ax2.set_ylabel('Spike Rate (Hz)')
        ax2.set_xlabel('Orientation (degrees)')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        pdf.savefig(fig)
        plt.close()
    return max_locations


def xcorr_sp_shuff(data_folder, sp, maxlags=50, shift=1, cp_sig=0.1):
    """
    Computes cross-correlation using spike bins, computes all possible combinations between many pairs of neurons.
    :param sp: pandas.DataFrame of all spike times
    :param maxlags: maximum time shift to compute
    :param spacing: spacing between time shifts
    :return:
        dictionary of correlation, dictionary of indexes
    """
    sp['time_ms'] = sp.time * 1000
    binned = pd.DataFrame()
    bins = np.arange(0, int(np.ceil(sp.time_ms.max())), shift)
    for i in np.sort(sp[sp.cluster > 0].cluster.unique()):
        binned[i] = np.histogram(sp[sp.cluster == i].time_ms, bins)[0]

    binned_shuff = binned.reindex(np.random.permutation(binned.index))
    binned_shuff = binned_shuff.reindex()
    binned_shuff.index = binned.index

    index = np.arange(-maxlags, maxlags+shift, shift)
    from itertools import combinations
    xcorr = pd.DataFrame(0, index=index, columns=list(combinations(binned.columns, 2)))
    xcorr_shuff = pd.DataFrame(0, index=index, columns=list(combinations(binned.columns, 2)))

    count = 0
    for i in binned.columns:
        nonzero_bins = binned[i].nonzero()[0]
        for j in np.arange(0, maxlags+shift, shift):
            corr_add = binned.loc[nonzero_bins + j/shift].sum().as_matrix()
            corr_sub = binned.loc[nonzero_bins - j/shift].sum().as_matrix()

            corr_add_shuff = binned_shuff.loc[nonzero_bins + j/shift].sum().as_matrix()
            corr_sub_shuff = binned_shuff.loc[nonzero_bins - j/shift].sum().as_matrix()

            xcorr.loc[j, xcorr.columns[count:count + binned.columns.max() - i]] = corr_add[i:]
            xcorr.loc[-j, xcorr.columns[count:count + binned.columns.max() - i]] = corr_sub[i:]

            xcorr_shuff.loc[j, xcorr_shuff.columns[count:count + binned.columns.max()-i]] = corr_add_shuff[i:]
            xcorr_shuff.loc[-j, xcorr_shuff.columns[count:count + binned.columns.max()-i]] = corr_sub_shuff[i:]

        count += binned.columns.max()-i

    corr_prob = (xcorr.loc[1] - xcorr_shuff.loc[1]) / xcorr.sum()
    if not os.path.exists(data_folder + '/_images'):
        os.makedirs(data_folder + '/_images')

    plt.ioff()
    for i, corr_prob_sig_idx in enumerate(corr_prob[corr_prob > cp_sig].index):
        fig, ax = plt.subplots()
        ax.bar(index-(shift/2), xcorr[corr_prob_sig_idx], label='matched',
               width=shift, color='#348ABD', alpha=0.5)
        ax.bar(index-(shift/2), xcorr_shuff[corr_prob_sig_idx], label='shuffled',
               width=shift, color='#E24A33', alpha=0.5)

        max_site_1 = int(sp[sp.cluster == corr_prob_sig_idx[0]].max_site.mode())
        max_site_2 = int(sp[sp.cluster == corr_prob_sig_idx[1]].max_site.mode())

        ax.set_xlim([-maxlags, maxlags])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Number of Events')
        ax.legend(loc=4)
        plt.suptitle('Cross-Correlogram for %s, CP = %.2f' % (str(corr_prob_sig_idx), corr_prob[corr_prob_sig_idx]))
        ax.set_title('Probe Sites: %d, %d' % (max_site_1, max_site_2), fontsize=10)

        plt.savefig(data_folder + '/_images/corr%d.pdf' % i, format='pdf')
        plt.close()
    plt.ion()

    return xcorr, xcorr_shuff


def xcorr_sp_mean(data_folder, sp, maxlags=40, shift=1, cz_sig=2.5):
    """
    Computes cross-correlation using spike bins, computes all possible combinations between many pairs of neurons.
    :param sp: pandas.DataFrame of all spike times
    :param maxlags: maximum time shift to compute
    :param spacing: spacing between time shifts
    :return:
        dictionary of correlation, dictionary of indexes
    """
    sp['time_ms'] = sp.time * 1000
    binned = pd.DataFrame()
    bins = np.arange(0, int(np.ceil(sp.time_ms.max())), shift)
    for i in np.sort(sp[sp.cluster > 0].cluster.unique()):
        binned[i] = np.histogram(sp[sp.cluster == i].time_ms, bins)[0]

    index = np.arange(-maxlags, maxlags+shift, shift)
    from itertools import combinations
    xcorr = pd.DataFrame(0, index=index, columns=list(combinations(binned.columns, 2)))

    count = 0
    for i in binned.columns:
        print('Processing column %d of %d' % (i, binned.columns.max()))
        nonzero_bins = binned[i].nonzero()[0]
        for j in np.arange(0, maxlags+shift, shift):
            print('Processing time shift of %d' % j)
            corr_add = binned.loc[nonzero_bins + j/shift].sum().as_matrix()
            corr_sub = binned.loc[nonzero_bins - j/shift].sum().as_matrix()

            xcorr.loc[j, xcorr.columns[count:count + binned.columns.max() - i]] = corr_add[i:]
            xcorr.loc[-j, xcorr.columns[count:count + binned.columns.max() - i]] = corr_sub[i:]

        count += binned.columns.max()-i

    mid_points = xcorr.index[len(xcorr.index)*19/40:len(xcorr.index)*21/40]
    corr_zscore = (xcorr.loc[mid_points].max() - xcorr.mean()) / xcorr.std()

    if not os.path.exists(data_folder + '/_images'):
        os.makedirs(data_folder + '/_images')

    plt.ioff()
    for i, corr_prob_sig_idx in enumerate(corr_zscore[corr_zscore > cz_sig].index):
        fig, ax = plt.subplots()
        ax.bar(index-(shift/2), xcorr[corr_prob_sig_idx], label='matched',
               width=shift, color='#348ABD', alpha=0.5)

        max_site_1 = int(sp[sp.cluster == corr_prob_sig_idx[0]].max_site.mode())
        max_site_2 = int(sp[sp.cluster == corr_prob_sig_idx[1]].max_site.mode())

        ax.set_xlim([-maxlags, maxlags])
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Number of Events')
        plt.suptitle('Cross-Correlogram for %s, CZ = %.2f' % (str(corr_prob_sig_idx), corr_zscore[corr_prob_sig_idx]))
        ax.set_title('Probe Sites: %d, %d' % (max_site_1, max_site_2), fontsize=10)

        plt.savefig(data_folder + '/_images/xcorr%d.pdf' % i, format='pdf')
        plt.close()
    plt.ion()

    return xcorr


def orisf_plot(data_folder, sp, cluster, trial_num, stim_time, make_plot=False):
    cluster_str = 'cluster' + str(cluster)

    sp_cluster = sp[sp.cluster == cluster]
    start_times = trial_num['stim_start'].as_matrix()
    end_times = trial_num['stim_end'].as_matrix()
    all_times = np.sort(np.concatenate([start_times, end_times]))

    spike_cut = pd.cut(sp_cluster.time, bins=all_times)
    spike_counts = pd.value_counts(spike_cut).reindex(spike_cut.cat.categories)
    trial_num[cluster_str] = (spike_counts.values[::2].flatten() / stim_time[2])

    counts_ori = trial_num.groupby(['orientation', 's_frequency'])[cluster_str].mean()
    counts_sf = trial_num.groupby(['s_frequency', 'orientation'])[cluster_str].mean()
    sem_ori = trial_num.groupby(['orientation', 's_frequency'])[cluster_str].sem()
    sem_sf = trial_num.groupby(['s_frequency', 'orientation'])[cluster_str].sem()

    s_freq_max = counts_ori.drop(256).unstack(level=1).mean(axis=0).idxmax()
    orientation_max = counts_ori.drop(256).unstack(level=1).mean(axis=1).idxmax()

    if make_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("Cluster %d, Site %d" % (cluster, sp[sp.cluster == cluster].max_site.mode()[0]), fontsize=16)

        colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

        counts_ori.drop(256)[orientation_max].plot(ax=ax1, linewidth=2)
        ax1.fill_between(counts_ori.drop(256)[orientation_max].index,
                         counts_ori.drop(256)[orientation_max] - sem_ori.drop(256)[orientation_max],
                         counts_ori.drop(256)[orientation_max] + sem_ori.drop(256)[orientation_max],
                         color='#E24A33', alpha=0.5)
        ax1.set_title('Spatial Frequency Tuning')
        ax1.set_ylabel('Spike Rate (Hz)')
        ax1.set_xlabel('Spatial Frequency (cycles/degree)')

        counts_sf.drop(256)[s_freq_max].plot(ax=ax2, linewidth=2)
        ax2.fill_between(counts_sf.drop(256)[s_freq_max].index,
                         counts_sf.drop(256)[s_freq_max] - sem_sf.drop(256)[s_freq_max],
                         counts_sf.drop(256)[s_freq_max] + sem_sf.drop(256)[s_freq_max],
                         color='#E24A33', alpha=0.5)
        ax2.set_title('Orientation Tuning')
        ax2.set_ylabel('Spike Rate (Hz)')
        ax2.set_xlabel('Orientation (degrees)')

        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        fig.subplots_adjust(top=0.88)
        plt.savefig(data_folder + '_images/' + cluster_str + '_all.pdf', bbox_inches='tight', format='pdf')
        plt.close()

    return trial_num
