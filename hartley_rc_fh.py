import os
import glob
import functions
import pandas as pd
import numpy as np

image_path = 'E:/color_vision/images/'
image_array = np.load(image_path + 'image_array.npy')

params_paths = 'F:/NHP/AD8/Ephys/20161205/hartley_gray/2016.12.6_AD8_001_009.mat'
data_folder = os.path.dirname(params_paths) + '/'

start_time = 20  # seconds
end_time = None  # seconds

if type(params_paths) != str:
    excel_path = list(glob.iglob(data_folder + '*.xlsx'))[0]
    exp_info = pd.read_excel(excel_path, header=None, names=['analyzer', 'start_time', 'end_time', 'stimulus'])

data_path = glob.glob(data_folder + '*.bin')[0]
jrclust_path = glob.glob(data_folder + '*.csv')[0]
analyzer_path = glob.glob(data_folder + '*.analyzer')[0]

sp = functions.jrclust_csv(jrclust_path)
trials = functions.get_params(params_paths, analyzer_path)

if not os.path.exists(data_folder + '/stim_samples.npy'):
    trials['stim_sample'] = functions.get_stim_samples_fh(data_path, start_time, end_time)
else:
    trials['stim_sample'] = np.load(data_folder + 'stim_samples.npy')
trials['stim_time'] = trials.stim_sample / 25000

# xcorr = functions.xcorr_spiketime_all(data_folder, sp, maxlag=50, spacing=1)

p = functions.PlotRF(trials, sp.time[sp.cluster == 52].as_matrix(), analyzer_path)
p.show()


# import sys
# from Pyqt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import matplotlib
# from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
#
# class PlotRFqt(QMainWindow):
#     def __init__(self, parent=None):
#         QMainWindow.__init__(self, parent)
#         self.setWindowTitle('Hartley Receptive Field')
#         self.create_menu()
#         self.create_main_frame()
#         self.create_status_bar()
#         self.textbox.setText('1')
#         self.on_draw()
#
#     def on_pick(self, event):
#         box_points = event.artist.get_bbox().get_points()
#         msg = "You've clicked on a bar with coords:\n %s" % box_points
#         QMessageBox.information(self, 'Click!', msg)
#
