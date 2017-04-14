import glob
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.ticker as mtick
import functions
# import boto3

matplotlib.use('Qt5Agg')
progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class RecordingInfo(object):
    def __init__(self, params_path=None):
        data_folder = os.path.dirname(params_path) + '/'

        start_time = 0  # seconds
        end_time = None  # seconds

        # if type(self.params_paths) != str:
        #     excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
        #     exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time', 'stim'])

        data_path = glob.glob(data_folder + '*nidq.bin')[0]
        jrclust_path = glob.glob(data_folder + '*.csv')[0]
        analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

        self.sp = functions.jrclust_csv(jrclust_path)
        self.trials = functions.get_params(params_path, analyzer_path)

        if not os.path.exists(data_folder + '/stim_samples.npy'):
            self.trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
        else:
            self.trials['stim_sample'] = np.load(data_folder + 'stim_samples.npy')
        self.trials['stim_time'] = self.trials.stim_sample / 25000


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100, RecordingInfo=None):
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
        self.recording_info = None
        self.cluster = None
        self.spike_times = None
        self.revcorr_images = None
        self.revcorr_center = None
        self.point_plot = None
        self.slider_ax = None
        self.slider = None

        self.tau_range = np.arange(-0.2, 0, 0.001)

    def compute_initial_figure(self, recording_info=None, cluster=None, param_path=None,
                               point_plot=None, plot=True, *args, **kwargs):
        self.point_plot = point_plot
        self.recording_info = recording_info
        self.cluster = cluster

        data_folder = os.path.dirname(param_path) + '/'
        analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

        xN, yN = functions.stim_size_pixels(analyzer_path)
        if not os.path.exists(data_folder + 'revcorr_image_array.npy'):
            cond, image_array = functions.build_hartley(analyzer_path)
            image_array = image_array[:, :, 0:cond.shape[0]]
            cond.to_pickle(data_folder + 'revcorr_image_cond.p')
            np.save(data_folder + 'revcorr_image_array.npy', image_array)
        else:
            cond = pd.read_pickle(data_folder + 'revcorr_image_cond.p')
            image_array = np.load(data_folder + 'revcorr_image_array.npy')

        self.spike_times = self.recording_info.sp.time[self.recording_info.sp.cluster == self.cluster].as_matrix()
        self.spike_times = self.spike_times[(self.spike_times > recording_info.trials.stim_time.min() + 0.2) &
                                            (self.spike_times < recording_info.trials.stim_time.max())]

        for i in np.where(np.diff(self.recording_info.trials.stim_time) > 0.3)[0]:
            beginning = self.recording_info.trials.loc[i, 'stim_time']
            end = self.recording_info.trials.loc[i+1, 'stim_time'] + 0.2
            self.spike_times = self.spike_times[(self.spike_times < beginning) |
                                                (self.spike_times > end)]

        if not os.path.exists(data_folder + 'revcorr_images_%d.npy' % cluster):
            # progdialog = QtWidgets.QProgressDialog('Processing cluster %d' % cluster, 'Cancel',
            #                                        0, np.size(self.tau_range))
            # progdialog.setWindowTitle('Progress')
            # progdialog.setWindowModality(QtCore.Qt.WindowModal)
            # progdialog.show()

            self.revcorr_images = np.zeros([xN.astype(int), yN.astype(int), self.tau_range.size])
            revcorr_results = functions.revcorr(self.tau_range, self.spike_times, self.recording_info.trials,
                                                self.revcorr_images, image_array, cond)
            # progdialog.close()

            for im in range(len(revcorr_results)):
                self.revcorr_images[:, :, im] = revcorr_results[im]

            np.save(data_folder + 'revcorr_images_%d.npy' % cluster, self.revcorr_images)
        else:
            self.revcorr_images = np.load(data_folder + 'revcorr_images_%d.npy' % cluster)

        if plot:
            self.revcorr_center = self.revcorr_images[np.round(xN/4).astype(int):np.round(xN*3/4).astype(int),
                                                      np.round(yN/4).astype(int):np.round(yN*3/4).astype(int),
                                                      :]

            starting_flat = np.argmax(np.abs(self.revcorr_center))
            starting_ind = np.unravel_index(starting_flat, self.revcorr_center.shape)

            plot_lim = functions.plot_lim(self.revcorr_center.min(), self.revcorr_center.max())
            self.l = self.ax.imshow(self.revcorr_center[:, :, starting_ind[2]],
                                    cmap='jet', picker=True, interpolation='bilinear',
                                    vmin=-plot_lim, vmax=plot_lim)

            self.slider_ax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
            self.slider_ax.cla()
            self.slider = Slider(self.slider_ax, 'Tau', 0.001, 0.20, valinit=-self.tau_range[starting_ind[2]])
            self.slider.valtext.set_text('{:03.3f}'.format(-self.tau_range[starting_ind[2]]))
            self.slider.on_changed(self.update_slider)
            self.slider.drawon = False

            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)

            self.fig.canvas.draw()
            self.fig.canvas.mpl_connect('pick_event', self.onpick)

            self.point_plot.compute_point_plot(self.revcorr_center, self.tau_range,
                                               starting_ind[1],
                                               starting_ind[0])

    def update_slider(self, value):
        """

        :param value: Float. Value from slider on image.
        :return: New image presented containing shifted time window.
        """

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


class LatencyMplCanvas(MplCanvas):
    def compute_latencies(self, clusters, latencies):
        pass


class AppWindowParent(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.l = QtWidgets.QGridLayout(self.main_widget)


class LatencyWindow(AppWindowParent):
    def __init__(self, app_window=None):
        AppWindowParent.__init__(self)
        self.app_window = app_window
        self.latency = LatencyMplCanvas(self.main_widget, width=5, height=5, dpi=100)
        self.l.addWidget(self.latency)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


class ApplicationWindow(AppWindowParent):
    def __init__(self):
        AppWindowParent.__init__(self)
        self.setWindowTitle("Hartley Receptive Field")

        self.param_path = None
        self.trial_info = None
        self.cluster = 1
        self.load_click_flag = 0
        self.process_click_flag = 0

        self.bar = self.menuBar()
        file_menu = self.bar.addMenu('File')
        file_menu.addAction('&Load', self.load_clicked, QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        file_menu.addAction('&Quit', self.file_quit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        self.receptive_field = RFMplCanvas(self.main_widget, width=5, height=5, dpi=100)
        self.rf_point = PointMplCanvas(self.main_widget, width=5, height=5, dpi=100)
        self.navi_toolbar1 = NavigationToolbar(self.receptive_field, self)
        self.l.addWidget(self.navi_toolbar1, 1, 0, 1, 6)

        self.l.addWidget(self.receptive_field, 2, 0, 1, 6)
        self.l.addWidget(self.rf_point, 3, 0, 1, 6)

        load_button = QtWidgets.QPushButton('Load File')
        self.l.addWidget(load_button, 0, 0, 1, 3)
        load_button.clicked.connect(self.load_clicked)

        self.process_button = QtWidgets.QPushButton('Process All Clusters')
        self.l.addWidget(self.process_button, 0, 3, 1, 3)
        self.process_button.setEnabled(False)

        cluster_label = QtWidgets.QLabel()
        cluster_label.setText('Cluster:')
        self.l.addWidget(cluster_label, 4, 0, 1, 1)

        self.cluster_edit = QtWidgets.QLineEdit()
        self.l.addWidget(self.cluster_edit, 4, 1, 1, 2)
        self.cluster_edit.returnPressed.connect(self.enter_press)

        self.cluster_button = QtWidgets.QPushButton('Process')
        self.l.addWidget(self.cluster_button, 4, 3, 1, 1)
        self.cluster_button.setEnabled(False)

        self.last_button = QtWidgets.QPushButton('Last')
        self.l.addWidget(self.last_button, 4, 4, 1, 1)
        self.last_button.clicked.connect(self.last_press)
        self.last_button.setEnabled(False)

        self.next_button = QtWidgets.QPushButton('Next')
        self.l.addWidget(self.next_button, 4, 5, 1, 1)
        self.next_button.clicked.connect(self.next_press)
        self.next_button.setEnabled(False)

        self.exp_details = QtWidgets.QTextEdit()
        self.exp_details.setReadOnly(True)
        self.l.addWidget(self.exp_details, 1, 6, 2, 1)

        self.cluster_details = QtWidgets.QTextEdit()
        self.cluster_details.setReadOnly(True)
        self.l.addWidget(self.cluster_details, 3, 6, 2, 1)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage('Please load .mat file.', 2000)
        self.latency_window = LatencyWindow(self)

    def load_clicked(self):
        self.param_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', '*.mat')
        if self.param_path[0] == "":
            return
        self.statusBar().showMessage('Loaded %s' % self.param_path[0])
        self.trial_info = RecordingInfo(self.param_path[0])
        data_folder = os.path.dirname(self.param_path[0])

        analyzer_path = glob.glob(data_folder + '/*.analyzer')[0]
        import scipy.io as sio
        analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
        analyzer = analyzer_complete['Analyzer']

        x_size = float(analyzer.P.param[5][2])
        y_size = float(analyzer.P.param[6][2])
        min_sf = float(analyzer.P.param[17][2])
        max_sf = float(analyzer.P.param[18][2])
        screen_dist = int(analyzer.M.screenDist)

        self.exp_details.clear()
        self.exp_details.append("<html><b>Recording Details</b></html>")
        self.exp_details.append('Date: \t%s' % os.path.dirname(data_folder)[-8:])
        self.exp_details.append('Recording: \t%s' % os.path.split(data_folder)[1])
        self.exp_details.append('Clusters: \t%d' %
                                np.unique(self.trial_info.sp.cluster[self.trial_info.sp.cluster > 0]).size)
        self.exp_details.append('X size: \t%.2f degrees' % (x_size / 2))
        self.exp_details.append('Y size: \t%.2f degrees' % (y_size / 2))
        self.exp_details.append('Distance: \t%d cm' % screen_dist)
        self.exp_details.append('Min. SF: \t%.1f cpd' % min_sf)
        self.exp_details.append('Max. SF: \t%.1f cpd' % max_sf)

        self.process_button.setEnabled(True)
        self.process_button.clicked.connect(self.process_clicked)

        self.cluster_button.setEnabled(True)
        self.cluster_button.clicked.connect(self.enter_press)

        if self.load_click_flag == 0:
            self.process_menu = self.bar.addMenu('Process')
            self.process_menu.addAction('Process All Clusters', self.process_clicked, QtCore.Qt.CTRL + QtCore.Qt.Key_P)
            self.load_click_flag += 1

    def process_clicked(self):
        clusters = np.unique(self.trial_info.sp.cluster[self.trial_info.sp.cluster > 0])
        progdialog = QtWidgets.QProgressDialog("", 'Cancel', 0, np.size(clusters))
        progdialog.setWindowTitle('Progress')
        progdialog.setWindowModality(QtCore.Qt.WindowModal)
        progdialog.show()
        for i, clust in enumerate(clusters):
            progdialog.setLabelText('Analyzing cluster %d' % clust)
            progdialog.setValue(i + 1)
            self.receptive_field.compute_initial_figure(recording_info=self.trial_info,
                                                        cluster=clust,
                                                        param_path=self.param_path[0],
                                                        point_plot=self.rf_point, plot=False)
        progdialog.close()
        if self.process_click_flag == 0:
            self.process_menu.addAction('Compute Latencies', self.latency_clicked)
            self.process_click_flag += 1

    def latency_clicked(self):
        self.latency_window.show()

    def enter_press(self):
        self.cluster = int(self.cluster_edit.text())
        self.update_cluster()

    def last_press(self):
        self.cluster -= 1
        self.update_cluster()

    def next_press(self):
        self.cluster += 1
        self.update_cluster()

    def update_cluster(self):
        self.statusBar().showMessage('Selected cluster %d.' % self.cluster)
        self.receptive_field.compute_initial_figure(recording_info=self.trial_info,
                                                    cluster=self.cluster,
                                                    param_path=self.param_path[0],
                                                    point_plot=self.rf_point)
        self.cluster_details.clear()
        self.cluster_details.append("<html><b>Cluster %d Details</b></html>" % self.cluster)
        self.cluster_details.append('Max Site: \t%d' %
                                    self.trial_info.sp[self.trial_info.sp.cluster == self.cluster].max_site.mode()[0])
        self.cluster_details.append('Spikes: \t%d' %
                                    self.trial_info.sp[self.trial_info.sp.cluster == self.cluster].time.size)
        self.cluster_edit.clear()
        if self.cluster != 1:
            self.last_button.setEnabled(True)

        if self.cluster != np.unique(self.trial_info.sp.cluster[self.trial_info.sp.cluster > 0]).size:
            self.next_button.setEnabled(True)

    def file_quit(self):
        self.close()

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('%s' % progname)
    aw.show()
    sys.exit(qApp.exec_())
