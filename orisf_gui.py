import sys
import os
import glob
import pandas as pd
import numpy as np
import functions
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets, QtGui

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe.
    """

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


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


class OriMplCanvas(MplCanvas):
    def compute_ori_tuning(self, clu, ori, resp, err, baseline):
        self.ax.cla()
        self.ax.plot(ori, resp)
        self.ax.fill_between(ori, resp-err, resp+err, alpha=0.5)
        self.ax.axhline(y=baseline, linestyle='--')
        self.ax.set_xlim([ori.min(), ori.max()])
        self.ax.set_title('Orientation Tuning for Cluster %d' % clu, fontsize=12)
        self.ax.set_xlabel('Direction (degrees)', fontsize=10)
        self.ax.set_ylabel('Spike Rate (spikes/second)', fontsize=10)
        self.ax.xaxis.set_tick_params(labelsize=6)
        self.ax.yaxis.set_tick_params(labelsize=6)
        self.fig.tight_layout()
        self.fig.canvas.draw()


class SFMplCanvas(MplCanvas):
    def compute_sf_tuning(self, clu, s_freq, resp, err):
        self.ax.cla()
        self.ax.plot(s_freq, resp)
        self.ax.fill_between(s_freq, resp-err, resp+err, alpha=0.5)
        self.ax.set_xlim([s_freq.min(), s_freq.max()])
        self.ax.set_title('Spatial Frequency Tuning for Cluster %d' % clu, fontsize=12)
        self.ax.set_xlabel('Spatial Frequency (cycles/degree)', fontsize=10)
        self.ax.set_ylabel('Spike Rate (spikes/second)', fontsize=10)
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


class SelectivityMplCanvas(MplCanvas):
    def compute_selectivity(self, depth, osi, param_string):
        self.ax.cla()
        self.ax.scatter(osi, depth, marker='o')
        self.ax.set_xlabel('OSI')
        self.ax.set_ylabel('Depth from Bottom of Layer 4C')
        self.ax.set_title(param_string + ' Selectivity')
        self.ax.xaxis.set_tick_params(labelsize=6)
        self.ax.yaxis.set_tick_params(labelsize=6)
        self.ax.set_xlim([0, 1])
        self.fig.tight_layout()
        self.fig.canvas.draw()
        pass


class SelectivityWindow(AppWindowParent):
    def __init__(self, app_window=None):
        AppWindowParent.__init__(self)
        self.app_window = app_window
        self.selectivity = SelectivityMplCanvas(self.main_widget, width=5, height=10, dpi=100)
        self.l.addWidget(self.selectivity)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


class ApplicationWindow(AppWindowParent):
    def __init__(self):
        AppWindowParent.__init__(self)
        self.cluster_info = None
        self.data_folder = None
        self.st = None
        self.trial_info = None
        self.stim_time = None
        self.sf_counts = {}
        self.sf_error = {}
        self.ori_counts = {}
        self.ori_error = {}
        self.dir_counts = {}
        self.dir_error = {}
        self.baseline = {}
        self.baseline_error = {}
        self.cluster = 0

        self.file_menu = self.menuBar().addMenu('File')
        self.file_menu.addAction('&Quit', self.file_quit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('Load Spike Time CSV', self.load_st_csv)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = self.menuBar().addMenu('Help')
        self.help_menu.addAction('&About', self.about)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.data_menu = self.menuBar().addMenu('Data')
        self.data_menu.addAction('Load Saved Data', self.load_saved)
        self.menuBar().addMenu(self.data_menu)

        self.ori_mpl = OriMplCanvas(self.main_widget, width=5, height=5, dpi=100)
        self.sf_mpl = SFMplCanvas(self.main_widget, width=5, height=5, dpi=100)
        self.navi_toolbar1 = NavigationToolbar(self.ori_mpl, self)
        self.l.addWidget(self.navi_toolbar1, 1, 0, 1, 6)

        self.l.addWidget(self.ori_mpl, 2, 0, 1, 6)
        self.l.addWidget(self.sf_mpl, 3, 0, 1, 6)

        load_button = QtWidgets.QPushButton('Load CSV')
        self.l.addWidget(load_button, 0, 0, 1, 3)
        load_button.clicked.connect(self.load_st_csv)

        self.process_button = QtWidgets.QPushButton('Process All Clusters')
        self.l.addWidget(self.process_button, 0, 3, 1, 2)
        self.process_button.setEnabled(False)

        self.save_button = QtWidgets.QPushButton('Save Cluster Info')
        self.l.addWidget(self.save_button, 0, 5, 1, 1)
        self.save_button.setEnabled(False)

        self.combo_box = QtWidgets.QComboBox()
        self.combo_box.addItem('Orientation')
        self.combo_box.addItem('Direction')
        self.combo_box.activated.connect(self.combo_activated)
        self.l.addWidget(self.combo_box, 0, 6, 1, 1)

        cluster_label = QtWidgets.QLabel()
        cluster_label.setText('Cluster:')
        self.l.addWidget(cluster_label, 4, 0, 1, 1)

        self.cluster_edit = QtWidgets.QLineEdit()
        self.l.addWidget(self.cluster_edit, 4, 1, 1, 3)
        self.cluster_edit.returnPressed.connect(self.enter_press)

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

        self.statusBar().showMessage("Welcome!")

    def combo_activated(self, index):
        return self.combo_box.itemText(index)

    def file_quit(self):
        self.close()

    def load_st_csv(self):
        csv_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open CSV File', '', '*.csv')
        if csv_path[0] == "":
            return

        data_folder = os.path.dirname(csv_path[0])
        self.data_folder = data_folder
        import glob
        analyzer_path = glob.glob(data_folder + '/*.analyzer')[0]

        self.st = functions.jrclust_csv(csv_path[0])
        self.trial_info, self.stim_time = functions.analyzer_pg_conds(analyzer_path)

        self.trial_info['direction'] = self.trial_info.ori
        self.trial_info.ori[(self.trial_info.ori >= 180) & (self.trial_info.ori != 256)] = \
            self.trial_info.ori[(self.trial_info.ori >= 180) & (self.trial_info.ori != 256)] - 180

        self.data_menu.addAction('Display Experiment Parameters', self.disp_trial_info)

        self.cluster_info = pd.DataFrame(self.st[self.st.cluster > 0].groupby('cluster').
                                         agg(lambda x: x.value_counts().index[0]).max_site, columns=['max_site'])

        self.cluster_info.insert(0, 'date', os.path.dirname(data_folder)[-8:])
        self.cluster_info.insert(1, 'exp_name', os.path.split(data_folder)[1])

        csd_info = pd.read_csv("csd_borders.csv")
        probe_geo = functions.get_probe_geo()

        date = int(os.path.dirname(data_folder)[-8:])
        self.cluster_info['csd_bottom'] = csd_info[csd_info.date == date].bottom.values[0]
        self.cluster_info['csd_top'] = csd_info[csd_info.date == date].top.values[0]
        self.cluster_info['relative_depth'] = probe_geo[:, 1][self.cluster_info.max_site - 1] - \
            csd_info[csd_info.date == date].bottom.values[0]

        self.process_button.clicked.connect(self.process_button_clicked)
        self.process_button.setEnabled(True)
        self.save_button.clicked.connect(self.save_button_clicked)
        self.save_button.setEnabled(True)
        self.statusBar().showMessage('Loaded spike times and analyzer.')

    def load_saved(self):
        csv_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder of CSV Files')
        try:
            csv_path
        except NameError:
            return

        csv_files = glob.glob(os.path.join(csv_path, '*.csv'))
        saved_df = pd.concat(pd.read_csv(f) for f in csv_files)

        self.selectivity_chart = SelectivityWindow(self)
        self.selectivity_chart.show()

        if str(self.combo_box.currentText()) == 'Orientation':
            self.selectivity_chart.selectivity.compute_selectivity(saved_df['relative_depth'], saved_df['osi'],
                                                                   str(self.combo_box.currentText()))
        else:
            self.selectivity_chart.selectivity.compute_selectivity(saved_df['relative_depth'], saved_df['dsi'],
                                                                   str(self.combo_box.currentText()))

    def process_button_clicked(self):
        if not os.path.exists(self.data_folder + '/stim_samples.npy'):
            data_path = glob.glob(self.data_folder + '/*.nidq.bin')[0]

            self.trial_info['stim_start'] = functions.get_stim_samples_pg(data_path, 0)[1::3] / 25000
            # self.trial_info['stim_start'] = functions.get_stim_samples_pg(data_path, 0) / 25000
            np.save(os.path.dirname(data_path) + '/stim_samples.npy', self.trial_num.stim_start * 25000)
        else:
            try:
                self.trial_info['stim_start'] = np.load(self.data_folder + '/stim_samples.npy')[1::3] / 25000
            except ValueError:
                self.trial_info['stim_start'] = np.load(self.data_folder + '/stim_samples.npy') / 25000
        self.trial_info['stim_end'] = self.trial_info.stim_start + self.stim_time[2]

        osi_array = np.zeros(self.st.cluster.max())
        dsi_array = np.zeros(self.st.cluster.max())

        for i in range(1, self.st.cluster.max() + 1):
            self.sf_counts[i-1], self.sf_error[i-1], self.ori_counts[i-1], self.ori_error[i-1], \
            self.dir_counts[i-1], self.dir_error[i-1], self.baseline[i-1], self.baseline_error[i-1], osi, dsi = \
                functions.orisf_counts(self.st, i, self.trial_info, self.stim_time)
            osi_array[i-1] = osi
            dsi_array[i-1] = dsi

        self.cluster_info['osi'] = osi_array
        self.cluster_info['dsi'] = dsi_array

        self.cluster = 1
        self.plot_cluster()
        return

    def save_button_clicked(self):
        save_filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Filename', '', '*.csv')
        self.cluster_info.to_csv(save_filename[0])
        return

    def enter_press(self):
        self.cluster = int(self.cluster_edit.text())
        self.plot_cluster()
        return

    def last_press(self):
        return

    def next_press(self):
        return

    def plot_cluster(self):
        if str(self.combo_box.currentText()) == 'Orientation':
            self.ori_mpl.compute_ori_tuning(self.cluster,
                                            self.ori_counts[self.cluster - 1].index.values,
                                            self.ori_counts[self.cluster - 1].values,
                                            self.ori_error[self.cluster - 1].values,
                                            self.baseline[self.cluster - 1][256])
        else:
            self.ori_mpl.compute_ori_tuning(self.cluster,
                                            self.dir_counts[self.cluster - 1].index.values,
                                            self.dir_counts[self.cluster - 1].values,
                                            self.dir_error[self.cluster - 1].values,
                                            self.baseline[self.cluster - 1][256])

        self.sf_mpl.compute_sf_tuning(self.cluster,
                                      self.sf_counts[self.cluster - 1].index.values,
                                      self.sf_counts[self.cluster - 1].values,
                                      self.sf_error[self.cluster - 1].values)

    def disp_trial_info(self):
        self.view = QtWidgets.QTableView()
        model = PandasModel(self.trial_info)
        self.view.setModel(model)
        self.view.verticalHeader().setVisible(True)
        self.view.horizontalHeader().setVisible(True)
        self.view.show()

    def closeEvent(self, ce):
        self.file_quit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """GUI for analyzing orientation and spatial frequency tuning. 
                                    Created by Anupam Garg, SNL-C, anupam@salk.edu."""
                                    )

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('%s' % progname)
    aw.show()
    sys.exit(qApp.exec_())
