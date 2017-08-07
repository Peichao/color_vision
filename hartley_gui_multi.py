import glob
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.ticker as mtick
import functions

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class RecordingInfo(object):
    def __init__(self, params_path=None, start_time=0.0, end_time=None):
        data_folder = os.path.dirname(params_path) + '/'

        data_path = glob.glob(data_folder + '*nidq.bin')[0]
        jrclust_path = glob.glob(data_folder + '*imec2.csv')[0]
        analyzer_path = glob.glob(params_path[:-4] + '/*.analyzer')[0]

        self.sp = functions.jrclust_csv(jrclust_path)
        self.trials = functions.get_params(params_path, analyzer_path)

        if not os.path.exists(data_folder + os.path.basename(params_path)[:-4] + '/stim_samples.npy'):
            self.trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
            np.save(data_folder + os.path.basename(params_path)[:-4] + '/stim_samples.npy', self.trials['stim_sample'])
        else:
            self.trials['stim_sample'] = np.load(data_folder + os.path.basename(params_path)[:-4] + '/stim_samples.npy')
        self.trials['stim_time'] = self.trials.stim_sample / 25000
        self.trials['stim_time'] += start_time


class MplCanvas(FigureCanvas):
    def __init__(self, *args, **kwargs):
        width = kwargs.pop('width', None)
        height = kwargs.pop('height', None)
        dpi = kwargs.pop('dpi', None)

        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class RFMplCanvas(MplCanvas):
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)

        param_path = kwargs.pop('param_path', None)
        self.recording_info = kwargs.pop('recording_info', None)
        self.revcorr_images = None
        self.revcorr_images_f = None

        parent_folder = os.path.dirname(param_path) + '/'
        self.data_folder = parent_folder + os.path.basename(param_path)[:-4] + '/'

        analyzer_path = glob.glob(self.data_folder + '*.analyzer')[0]

        if not os.path.exists(self.data_folder + 'revcorr_image_array.npy'):
            self.cond, self.image_array = functions.build_hartley(analyzer_path)
            self.image_array = self.image_array[:, :, 0:self.cond.shape[0]]
            self.cond.to_pickle(self.data_folder + 'revcorr_image_cond.p')
            np.save(self.data_folder + 'revcorr_image_array.npy', self.image_array)
        else:
            self.cond = pd.read_pickle(self.data_folder + 'revcorr_image_cond.p')
            self.image_array = np.load(self.data_folder + 'revcorr_image_array.npy')

        self.xN, self.yN = functions.stim_size_pixels(analyzer_path)

        self.revcorr_center = None
        self.point_plot = None
        self.slider_ax = None
        self.slider = None

        self.tau_range = np.arange(-0.2, 0, 0.001)

        self.slider_ax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(self.slider_ax, 'Tau', 0.001, 0.20, valinit=0.001)

    def compute_initial_figure(self, cluster=None, point_plot=None, plot=True, stim=None, *args, **kwargs):
        self.point_plot = point_plot

        spike_times = self.recording_info.sp.time[self.recording_info.sp.cluster == cluster].as_matrix()
        spike_times = spike_times[(spike_times > self.recording_info.trials.stim_time.min() + 0.2) &
                                  (spike_times < self.recording_info.trials.stim_time.max())]

        for i in np.where(np.diff(self.recording_info.trials.stim_time) > 0.3)[0]:
            beginning = self.recording_info.trials.loc[i, 'stim_time']
            end = self.recording_info.trials.loc[i+1, 'stim_time'] + 0.2
            spike_times = spike_times[(spike_times < beginning) |
                                      (spike_times > end)]

        if not os.path.exists(self.data_folder + 'revcorr_images_%d.npy' % cluster):
            revcorr_images = np.zeros([self.xN.astype(int), self.yN.astype(int), self.tau_range.size])
            revcorr_results, self.revcorr_images_f = functions.revcorr(self.tau_range, spike_times,
                                                                       self.recording_info.trials,
                                                                       self.image_array, self.cond)

            for im in range(len(revcorr_results)):
                revcorr_images[:, :, im] = revcorr_results[im]

            np.save(self.data_folder + 'revcorr_images_%d.npy' % cluster, revcorr_images)
            np.save(self.data_folder + 'revcorr_images_f_%d.npy' % cluster, self.revcorr_images_f)
        else:
            if not os.path.exists(self.data_folder + 'revcorr_images_f_%d.npy' % cluster):
                self.revcorr_images_f = functions.revcorr_f(self.tau_range, spike_times, self.recording_info.trials)
            else:
                self.revcorr_images_f = np.load(self.data_folder + 'revcorr_images_f_%d.npy' % cluster, mmap_mode='r')
                np.save(self.data_folder + 'revcorr_images_f_%d.npy' % cluster, self.revcorr_images_f)
            revcorr_images = np.load(self.data_folder + 'revcorr_images_%d.npy' % cluster, mmap_mode='r')

        if plot:
            self.ax.cla()

            self.ax.axis('off')
            self.revcorr_center = revcorr_images[np.round(self.xN/4).astype(int):np.round(self.xN*3/4).astype(int),
                                                 np.round(self.yN/4).astype(int):np.round(self.yN*3/4).astype(int),
                                                 :]

            starting_flat = np.argmax(np.abs(self.revcorr_center))
            starting_ind = np.unravel_index(starting_flat, self.revcorr_center.shape)

            plot_lim = functions.plot_lim(self.revcorr_center.min(), self.revcorr_center.max())
            self.l = self.ax.imshow(self.revcorr_center[:, :, starting_ind[2]],
                                    cmap='jet', picker=True, interpolation='bilinear',
                                    vmin=-plot_lim, vmax=plot_lim)

            self.slider.set_val(-self.tau_range[starting_ind[2]])
            self.slider.valtext.set_text('{:03.3f}'.format(-self.tau_range[starting_ind[2]]))
            self.slider.on_changed(self.update_slider)
            self.slider.drawon = False

            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.set_title(stim)

            self.fig.canvas.draw()
            self.fig.canvas.mpl_connect('pick_event', self.onpick)

            self.point_plot.compute_point_plot(self.revcorr_center, self.tau_range,
                                               starting_ind[1],
                                               starting_ind[0])
            return plot_lim

    def plot_revcorr_images_f(self):
        self.ax.cla()

        self.ax.axis('on')
        self.ax.xaxis.set_visible(True)
        self.ax.yaxis.set_visible(True)

        starting_flat = np.argmax(np.abs(self.revcorr_center))
        starting_ind = np.unravel_index(starting_flat, self.revcorr_center.shape)
        plot_lim = functions.plot_lim(self.revcorr_images_f.min(), self.revcorr_images_f.max())

        sfx_vals = np.sort(np.unique(self.recording_info.trials.sf_x))
        sfy_vals = np.sort(np.unique(self.recording_info.trials.sf_y))

        self.ax.set_xlim([0, 3])
        self.ax.set_ylim([-3, 3])

        self.ax.set_xlabel(r'$\omega_x$')
        self.ax.set_ylabel(r'$\omega_y$')

        self.ax.imshow(self.revcorr_images_f[:, :, starting_ind[2]].T, cmap='jet', interpolation='spline36',
                       extent=[sfx_vals.min(), sfx_vals.max(),
                               sfy_vals.min(), sfy_vals.max()])

        self.slider.set_val(-self.tau_range[starting_ind[2]])
        self.slider.valtext.set_text('{:03.3f}'.format(-self.tau_range[starting_ind[2]]))
        self.slider.on_changed(self.update_slider_f)
        self.slider.drawon = False

    def update_vlim(self, plot_lim):
        self.l.set_clim(vmin=-plot_lim, vmax=plot_lim)
        self.fig.canvas.draw()

    def update_slider(self, value):
        """

        :param value: Float. Value from slider on image.
        :return: New image presented containing shifted time window.
        """

        self.l.set_data(self.revcorr_center[:, :, np.searchsorted(self.tau_range, -value)])
        self.slider.valtext.set_text('{:03.3f}'.format(value))
        self.fig.canvas.draw()

    def update_slider_f(self, value):
        self.l.set_data(self.revcorr_center[:, :, np.searchsorted(self.tau_range, -value)])
        self.slider.valtext.set_text('{:03.3f}'.format(value))
        self.fig.canvas.draw()

    def onpick(self, event):
        """

        :param event: Mouse click on image.
        :return: New data on figure l2 displaying average contrast over time at point clicked during event.
        """
        mouse_event = event.mouseevent
        x = int(np.round(mouse_event.xdata))
        y = int(np.round(mouse_event.ydata))
        self.point_plot.compute_point_plot(self.revcorr_center, self.tau_range, x, y)


class PointMplCanvas(MplCanvas):
    def compute_point_plot(self, revcorr_center, t, x, y):
        y_data = revcorr_center[y, x, :]
        self.ax.cla()
        self.ax.plot(t, y_data)
        self.ax.set_title('Spike-Triggered Average for Point [%d, %d]' % (y, x), fontsize=12)
        self.ax.set_xlabel('Time Before Spike (seconds)', fontsize=10)
        self.ax.set_ylabel('Spike-Triggered Average', fontsize=10)
        plot_lim = functions.plot_lim(revcorr_center.min(), revcorr_center.max())
        self.ax.set_ylim([-plot_lim, plot_lim])
        self.ax.set_xlim([np.min(t), 0])
        self.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        self.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        self.ax.xaxis.set_tick_params(labelsize=6)
        self.ax.yaxis.set_tick_params(labelsize=6)

        self.fig.tight_layout()
        self.fig.canvas.draw()


class AppWindowParent(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.l = QtWidgets.QGridLayout(self.main_widget)


class ApplicationWindow(AppWindowParent):
    def __init__(self):
        AppWindowParent.__init__(self)
        self.setWindowTitle("Hartley Receptive Field")

        self.xls_path = None
        self.trial_info = None
        self.cluster = 1
        self.load_click_flag = 0
        self.process_click_flag = 0

        self.plot_lim = []
        self.rf_dict = {}
        self.rf_point_dict = {}
        self.rec_info_dict = {}
        self.navi_toolbar_dict = {}

        self.bar = self.menuBar()
        file_menu = self.bar.addMenu('File')
        file_menu.addAction('&Load', self.load_clicked, QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        file_menu.addAction('&Quit', self.file_quit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        load_button = QtWidgets.QPushButton('Load File')
        self.l.addWidget(load_button, 0, 0, 1, 1)
        load_button.clicked.connect(self.load_clicked)

        self.process_button = QtWidgets.QPushButton('Process All Clusters')
        self.l.addWidget(self.process_button, 0, 1, 1, 1)
        self.process_button.setEnabled(False)

        self.cluster_label = QtWidgets.QLabel()
        self.cluster_label.setText('Cluster:')
        self.cluster_edit = QtWidgets.QLineEdit()

        self.normalize_button = QtWidgets.QPushButton('Normalize')
        self.normalize_button.clicked.connect(self.normalize_axes)
        self.l.addWidget(self.normalize_button, 4, 2, 1, 1)
        self.normalize_button.setEnabled(False)

        self.frequency_button = QtWidgets.QPushButton('Plot Frequency Domain')
        self.frequency_button.clicked.connect(self.plot_frequency)
        self.l.addWidget(self.frequency_button, 0, 2, 1, 1)
        self.frequency_button.setEnabled(False)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.showMaximized()

        self.statusBar().showMessage('Please load .mat file.', 2000)

    def load_clicked(self):
        self.xls_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', '*.csv')
        if self.xls_path[0] == "":
            return
        self.statusBar().showMessage('Loaded %s' % self.xls_path[0])
        data_folder = os.path.dirname(self.xls_path[0]) + '/'

        self.exp_list = pd.read_csv(self.xls_path[0])

        for i in range(self.exp_list.shape[0]):
            if not os.path.exists(data_folder + self.exp_list.data_path[i][:-4]):
                os.makedirs(data_folder + self.exp_list.data_path[i][:-4])
            self.rec_info_dict[self.exp_list.stim[i]] = RecordingInfo(params_path=data_folder + self.exp_list.data_path[i],
                                                                      start_time=self.exp_list.start_time[i],
                                                                      end_time=self.exp_list.end_time[i])

            self.rf_dict[self.exp_list.stim[i]] = RFMplCanvas(self.main_widget, width=5, height=5, dpi=100,
                                                              param_path=data_folder + self.exp_list.data_path[i],
                                                              recording_info=self.rec_info_dict[self.exp_list.stim[i]])
            self.l.addWidget(self.rf_dict[self.exp_list.stim[i]], 1, i, 1, 1)

            self.rf_point_dict[self.exp_list.stim[i]] = PointMplCanvas(self.main_widget, width=5, height=5, dpi=100)
            self.l.addWidget(self.rf_point_dict[self.exp_list.stim[i]], 2, i, 1, 1)

            self.navi_toolbar_dict[self.exp_list.stim[i]] = NavigationToolbar(self.rf_dict[self.exp_list.stim[i]], self)
            self.l.addWidget(self.navi_toolbar_dict[self.exp_list.stim[i]], 3, i, 1, 1)

        self.l.addWidget(self.cluster_label, 4, 0, 1, 1)
        self.l.addWidget(self.cluster_edit, 4, 1, 1, 1)

        self.cluster_edit.returnPressed.connect(self.enter_press)
        self.process_button.setEnabled(True)
        self.process_button.clicked.connect(self.process_clicked)

    def enter_press(self):
        self.cluster = int(self.cluster_edit.text())
        self.update_cluster()
        self.normalize_button.setEnabled(True)
        self.frequency_button.setEnabled(True)

    def update_cluster(self):
        self.statusBar().showMessage('Selected cluster %d.' % self.cluster)
        self.plot_lim = []
        for i in range(self.exp_list.shape[0]):
            self.plot_lim.append(self.rf_dict[self.exp_list.stim[i]].compute_initial_figure(cluster=self.cluster,
                                                                                            point_plot=self.
                                                                                            rf_point_dict[
                                                                                                self.exp_list.stim[i]],
                                                                                            plot=True,
                                                                                            stim=self.exp_list.stim[i]))

        self.cluster_edit.clear()

    def normalize_axes(self):
        for i in range(self.exp_list.shape[0]):
            self.rf_dict[self.exp_list.stim[i]].update_vlim(np.max(self.plot_lim))

    def process_clicked(self):
        self.statusBar().showMessage('Processing all clusters.')
        clusters = np.unique(self.rec_info_dict[list(self.rec_info_dict.keys())[0]].sp.cluster[self.rec_info_dict[list(self.rec_info_dict.keys())[0]].sp.cluster > 0])
        progdialog = QtWidgets.QProgressDialog("", 'Cancel', 0, np.size(clusters))
        progdialog.setWindowTitle('Progress')
        progdialog.setWindowModality(QtCore.Qt.WindowModal)
        progdialog.show()
        for i, clust in enumerate(clusters):
            progdialog.setLabelText('Analyzing cluster %d' % clust)
            progdialog.setValue(i + 1)
            for j in range(self.exp_list.shape[0]):
                self.rf_dict[self.exp_list.stim[j]].compute_initial_figure(cluster=clust,
                                                                           point_plot=self.rf_point_dict[self.exp_list.stim[j]],
                                                                           plot=False)
        progdialog.close()

    def file_quit(self):
        self.close()

    def plot_frequency(self):
        for i in range(self.exp_list.shape[0]):
            self.rf_dict[self.exp_list.stim[i]].plot_revcorr_images_f()

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('%s' % progname)
    aw.show()
    sys.exit(qApp.exec_())
