import glob
import tables
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
plt.style.use('ggplot')
plt.ioff()


def waveforms(data_folder):
    waveform_path = glob.glob(data_folder + '*spkwav.mat')[0]
    waveform_h5 = tables.openFile(waveform_path)
    waveform = waveform_h5.root.trSpkWav[:]
    waveform_h5.close()

    return waveform


def waveform_analysis(data_folder):
    waveform = waveforms(data_folder)
    min_index = np.argmin(np.min(waveform, axis=2), axis=1)
    peak_waveforms = waveform[np.arange(waveform.shape[0]), min_index, :]  # all clusters max deflection

    # calculate shift of each cluster to cluster that is farther to left (minimum min peak)
    shift = np.argmin(peak_waveforms, axis=1) - np.min(np.argmin(peak_waveforms, axis=1))

    def shifted(shift, x, s):
        """
        Shifts waveform by constant.
        :param shift: amount (samples) to shift each cluster waveform by
        :param x: x axis (samples)
        :param s: waveform amplitude
        :return: shifted x axis, shifted waveform
        """
        if shift == 0:
            return x, s
        else:
            return x[:-1*shift], s[shift:]

    half_max_width = np.zeros(waveform.shape[0])
    peak_trough_width = np.zeros(waveform.shape[0])
    x = np.arange(waveform.shape[2])
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in np.arange(waveform.shape[0]):
        px, py = shifted(shift[i], x, peak_waveforms[i, :])
        # normalize to lowest point so that trough is set to zero
        py_norm = py - py[np.argmin(py)]

        # calculate half max by calculating last point where waveform is below half of peak
        half_max_width[i] = (np.where(py_norm < (np.max(py_norm)/2)))[0][-1] - (np.argmin(py_norm)) / 25
        # calculate trough-peak width
        peak_trough_width[i] = (np.argmax(py_norm[10:]) + 10) - np.argmin(py_norm) / 25

        ax1.plot(px / 25, py_norm)
        ax2.scatter(half_max_width[i], peak_trough_width[i])

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (Normalized)')
    ax1.set_xlim([0, 1.0])
    ax2.set_xlabel('Half-Max Width (ms)')
    ax2.set_ylabel('Peak-Trough Width (ms)')

    plt.savefig(data_folder + 'waveform.pdf', format='pdf')
    plt.close()

    return waveform, min_index
