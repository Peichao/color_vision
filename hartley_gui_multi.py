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

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage('Please load .mat file.', 2000)

    def load_clicked(self):
        self.xls_path = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open File', '', '*.xlsx')
        if self.xls_path[0] == "":
            return
        self.statusBar().showMessage('Loaded %s' % self.xls_path[0])
        print(self.xls_path[0][0])

        exp_list = pd.read_excel(self.xls_path[0], engine='xlrd')
        print(exp_list)

    def file_quit(self):
        self.close()

if __name__ == '__main__':
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle('%s' % progname)
    aw.show()
    sys.exit(qApp.exec_())
