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
    def __init__(self, params_path=None, samples=None):
        data_folder = os.path.dirname(params_path) + '/'
        kilosort_path = os.path.dirname(os.path.dirname(data_folder)) + '/'

        im_fs = 30000
        analyzer_path = glob.glob(data_folder + '*.analyzer')[0]
        analyzer_name = os.path.splitext(os.path.basename(analyzer_path))[0]

        self.sp = functions.kilosort_info(kilosort_path, im_fs)
        self.trials = functions.get_params(params_path, analyzer_path)

        self.start_sample, self.end_sample = samples
        self.sp['time'] = self.sp['time'] - (self.start_sample / im_fs)

        print('Loading sync channels and calculating timing info.')
        im_flips = np.load(data_folder + analyzer_name + '_im_flips.npy')
        ni_flips = np.load(data_folder + analyzer_name + '_ni_flips.npy')

        im_flips_z = im_flips - im_flips[0, 0]
        ni_flips_z = ni_flips - ni_flips[0, 0]
        ni_drift = np.mean(im_flips_z.ravel()[1:] / ni_flips_z.ravel()[1:])

        # if not os.path.exists(data_folder + '/stim_samples.npy'):
        #     self.trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
        # else:
        #     self.trials['stim_sample'] = np.load(data_folder + 'stim_samples.npy')
        trial_samples = np.load(data_folder + analyzer_name + '_trial_samples.npy')
        trial_samples_z = trial_samples - ni_flips[0, 0]
        trial_samples_corrected = (trial_samples_z * ni_drift) + im_flips[0, 0]

        self.trials['stim_sample'] = trial_samples_corrected
        self.trials['stim_time'] = self.trials['stim_sample'] / im_fs


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
        self.starting_latency = None

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
            end = self.recording_info.trials.loc[i + 1, 'stim_time'] + 0.2
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
                                                image_array, cond)
            # progdialog.close()

            for im in range(len(revcorr_results[0])):
                self.revcorr_images[:, :, im] = revcorr_results[0][im]

            np.save(data_folder + 'revcorr_images_%d.npy' % cluster, self.revcorr_images)
        else:
            self.revcorr_images = np.load(data_folder + 'revcorr_images_%d.npy' % cluster)

        if plot:
            self.revcorr_center = self.revcorr_images[np.round(xN / 4).astype(int):np.round(xN * 3 / 4).astype(int),
                                  np.round(yN / 4).astype(int):np.round(yN * 3 / 4).astype(int),
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
            self.starting_latency = self.tau_range[starting_ind[2]]
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


class SummaryMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, sharey='row', figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_selectivity(self, summary_df, rf_list):
        bin_width = 5
        bins = np.arange(summary_df.relative_depth.min(), summary_df.relative_depth.max() + bin_width, bin_width)

        summary_df[summary_df.rf == 'On']['relative_depth'].plot(kind='hist', ax=self.ax1, color='#E24A33',
                                                                 orientation='horizontal', bins=bins)
        summary_df[summary_df.rf == 'Off']['relative_depth'].plot(kind='hist', ax=self.ax2, color='#348ABD',
                                                                  orientation='horizontal', bins=bins)
        summary_df[summary_df.rf == 'Oriented']['relative_depth'].plot(kind='hist', ax=self.ax3, color='#777777',
                                                                       orientation='horizontal', bins=bins)
        self.ax1.set_xlabel('ON cells')
        self.ax2.set_xlabel('OFF cells')
        self.ax3.set_xlabel('ON/OFF cells')
        self.ax1.set_ylabel('Depth from Bottom of Layer 4C')
        self.ax1.xaxis.set_tick_params(labelsize=6)
        self.ax1.yaxis.set_tick_params(labelsize=6)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        pass


class AppWindowParent(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.main_widget = QtWidgets.QWidget(self)
        self.l = QtWidgets.QGridLayout(self.main_widget)


class SummaryWindow(AppWindowParent):
    def __init__(self, app_window=None):
        AppWindowParent.__init__(self)
        self.app_window = app_window
        self.selectivity = SummaryMplCanvas(self.main_widget, width=15, height=10, dpi=100)
        self.l.addWidget(self.selectivity)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


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
        self.params = None
        self.cluster_info = None
        self.cluster = 1
        self.load_click_flag = 0
        self.process_click_flag = 0

        self.bar = self.menuBar()
        file_menu = self.bar.addMenu('File')
        file_menu.addAction('&Load', self.load_clicked, QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        file_menu.addAction('&Quit', self.file_quit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        self.receptive_field = RFMplCanvas(self.main_widget, width=30, height=30, dpi=100)
        self.rf_point = PointMplCanvas(self.main_widget, width=30, height=30, dpi=100)
        self.navi_toolbar1 = NavigationToolbar(self.receptive_field, self)
        self.l.addWidget(self.navi_toolbar1, 1, 0, 1, 6)

        self.l.addWidget(self.receptive_field, 2, 0, 1, 6)
        self.l.addWidget(self.rf_point, 3, 0, 1, 6)

        load_button = QtWidgets.QPushButton('Load File')
        self.l.addWidget(load_button, 0, 0, 1, 2)
        load_button.clicked.connect(self.load_clicked)

        self.process_button = QtWidgets.QPushButton('Process All Clusters')
        self.l.addWidget(self.process_button, 0, 2, 1, 2)
        self.process_button.setEnabled(False)

        self.process_menu = self.bar.addMenu('Process')
        self.process_menu.addAction('Plot Summary', self.summary_clicked)

        self.save_button = QtWidgets.QPushButton('Save RF Info')
        self.l.addWidget(self.save_button, 0, 4, 1, 2)
        self.save_button.clicked.connect(self.save_clicked)
        self.save_button.setEnabled(False)

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

        start_label = QtWidgets.QLabel()
        start_label.setText('Start Sample:')
        self.l.addWidget(start_label, 5, 0, 1, 1)

        self.start_edit = QtWidgets.QLineEdit()
        self.l.addWidget(self.start_edit, 5, 1, 1, 1)

        end_label = QtWidgets.QLabel()
        end_label.setText('End Sample:')
        self.l.addWidget(end_label, 5, 2, 1, 1)

        self.end_edit = QtWidgets.QLineEdit()
        self.l.addWidget(self.end_edit, 5, 3, 1, 1)

        self.exp_details = QtWidgets.QTextEdit()
        self.exp_details.setReadOnly(True)
        self.l.addWidget(self.exp_details, 1, 6, 2, 1)

        self.cluster_details = QtWidgets.QTextEdit()
        self.cluster_details.setReadOnly(True)
        self.l.addWidget(self.cluster_details, 3, 6, 2, 1)

        self.cluster_options = QtWidgets.QVBoxLayout()

        self.sc_list = ['Simple', 'Complex']
        self.rf_list = ['On', 'Off', 'Oriented', 'None']

        self.sc_box, self.sc_button_group = self.create_radiobutton_group('Select cell type:', self.sc_list)
        self.sc_button_group.buttonClicked.connect(self.sc_clicked)
        self.rf_box, self.rf_button_group = self.create_radiobutton_group('Select RF organization:', self.rf_list)
        self.rf_button_group.buttonClicked.connect(self.rf_clicked)
        latency_label = QtWidgets.QLabel()
        latency_label.setText('Latency (ms):')
        self.latency_edit = QtWidgets.QLineEdit()

        self.cluster_options.addWidget(self.sc_box)
        self.cluster_options.addWidget(self.rf_box)
        self.cluster_options.addWidget(latency_label)
        self.cluster_options.addWidget(self.latency_edit)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage('Please load .mat file.', 2000)
        self.latency_window = LatencyWindow(self)

    def load_clicked(self):
        self.param_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', '*.mat')
        if self.param_path[0] == "":
            return
        self.statusBar().showMessage('Loaded %s' % self.param_path[0])
        data_folder = os.path.dirname(self.param_path[0])

        self.trial_info = RecordingInfo(self.param_path[0], samples=(float(self.start_edit.text()),
                                                                     float(self.end_edit.text())))

        # self.cluster_info = pd.DataFrame(self.trial_info.sp[self.trial_info.sp.cluster > 0].groupby('cluster').
        #                                  agg(lambda x: x.value_counts().index[0]).max_site, columns=['max_site'])

        self.cluster_info = pd.DataFrame()

        self.cluster_info.insert(0, 'date', os.path.dirname(data_folder)[-8:])
        self.cluster_info.insert(1, 'exp_name', os.path.split(data_folder)[1])
        # csd_info = pd.read_csv("csd_borders.csv")
        probe_geo = functions.get_probe_geo()

        # date = int(os.path.dirname(data_folder)[-8:])
        # self.cluster_info['csd_bottom'] = csd_info[csd_info.date == date].bottom.values[0]
        # self.cluster_info['csd_top'] = csd_info[csd_info.date == date].top.values[0]
        # self.cluster_info['relative_depth'] = probe_geo[:, 1][self.cluster_info.max_site - 1] - \
        #                                       csd_info[csd_info.date == date].bottom.values[0]

        # self.cluster_info['sc'] = 'Complex'
        # self.cluster_info['rf'] = 'None'
        self.cluster_info['latency'] = 0.0

        analyzer_path = glob.glob(data_folder + '/*.analyzer')[0]
        analyzer = functions.load_analyzer(analyzer_path)

        self.params = functions.analyzer_params(analyzer_path)
        x_size = self.params['x_size']
        y_size = self.params['y_size']
        min_sf = self.params['min_sf']
        max_sf = self.params['max_sf']
        colorspace = self.params['colorspace']
        self.cluster_info['colorspace'] = colorspace
        screen_dist = int(analyzer.M.screenDist)

        self.exp_details.clear()
        self.exp_details.append("<html><b>Recording Details</b></html>")
        self.exp_details.append('Date: \t\t%s' % os.path.dirname(data_folder)[-8:])
        self.exp_details.append('Recording: \t%s' % os.path.split(data_folder)[1])
        self.exp_details.append('Clusters: \t\t%d' %
                                np.unique(self.trial_info.sp.cluster[self.trial_info.sp.cluster > 0]).size)
        self.exp_details.append('X size: \t\t%.2f deg' % (x_size / 2))
        self.exp_details.append('Y size: \t\t%.2f deg' % (y_size / 2))
        self.exp_details.append('Distance: \t\t%d cm' % screen_dist)
        self.exp_details.append('Min. SF: \t\t%.1f cpd' % min_sf)
        self.exp_details.append('Max. SF: \t\t%.1f cpd' % max_sf)

        self.process_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.process_button.clicked.connect(self.process_clicked)

        self.cluster_button.setEnabled(True)
        self.cluster_button.clicked.connect(self.enter_press)
        self.l.addLayout(self.cluster_options, 1, 7, 2, 1)

        if self.load_click_flag == 0:
            self.process_menu.addAction('Process All Clusters', self.process_clicked, QtCore.Qt.CTRL + QtCore.Qt.Key_P)
            self.load_click_flag += 1

    def save_clicked(self):
        data_folder = os.path.dirname(self.param_path[0])
        date = int(os.path.dirname(data_folder)[-8:])
        self.cluster_info.to_csv(data_folder + '/cluster_info.csv')
        self.cluster_info.to_csv('F:/NHP/%s.csv' % date)

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
                                                    point_plot=self.rf_point,
                                                    )
        self.cluster_info.set_value(self.cluster, 'latency', -self.receptive_field.starting_latency)
        self.latency_edit.setText(str(-self.receptive_field.starting_latency))

        self.cluster_details.clear()
        self.cluster_details.append("<html><b>Cluster %d Details</b></html>" % self.cluster)
        # self.cluster_details.append('Max Site: \t%d' %
        #                             self.trial_info.sp[self.trial_info.sp.cluster == self.cluster].max_site.mode()[0])
        self.cluster_details.append('Spikes: \t%d' %
                                    self.trial_info.sp[self.trial_info.sp.cluster == self.cluster].time.size)
        self.cluster_edit.clear()
        if self.cluster != 1:
            self.last_button.setEnabled(True)

        if self.cluster != np.unique(self.trial_info.sp.cluster[self.trial_info.sp.cluster > 0]).size:
            self.next_button.setEnabled(True)

    def file_quit(self):
        self.close()

    def create_radiobutton_group(self, title, button_list):
        group_box = QtWidgets.QGroupBox(title)
        radio_button_group = QtWidgets.QButtonGroup()
        vbox = QtWidgets.QVBoxLayout()

        radios = []
        for button in button_list:
            radios.append(QtWidgets.QRadioButton(button))

        for i, each in enumerate(radios):
            vbox.addWidget(each)
            radio_button_group.addButton(each)
            radio_button_group.setId(each, i)

        group_box.setLayout(vbox)

        return group_box, radio_button_group

    def sc_clicked(self):
        self.cluster_info.set_value(self.cluster, 'sc', self.sc_button_group.checkedButton().text())

    def rf_clicked(self):
        self.cluster_info.set_value(self.cluster, 'rf', self.rf_button_group.checkedButton().text())

    def summary_clicked(self):
        csv_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder of CSV Files')
        try:
            csv_path
        except NameError:
            return

        csv_files = glob.glob(os.path.join(csv_path, '*.csv'))
        summary_df = pd.concat(pd.read_csv(f) for f in csv_files)

        self.selectivity_chart = SummaryWindow(self)
        self.selectivity_chart.show()

        self.selectivity_chart.selectivity.compute_selectivity(summary_df, self.rf_list)


if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('%s' % progname)
    aw.showMaximized()
    sys.exit(qApp.exec_())

