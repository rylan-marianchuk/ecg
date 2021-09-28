"""
transforms.py

classes implementing common digital signal transformations:
Fourier, WindowedFourier, Wavelet, Median pass filter

@author Rylan Marianchuk
September 2021
"""
import plotly.graph_objs as go
import torch
import pywt
import math

class MedianPass(object):
    def __init__(self, radius, T=10, fs=500):
        """
        param radius: window size to extract median from
        """
        self.r = radius
        self.fs = fs
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # For display/figure generation in plotly
        self.domain = torch.linspace(0, self.T, self.fs * self.T)  # Generic time domain

    def __call__(self, sample):
        median_passed = torch.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            median_passed[i] = torch.median(sample[max(0, i - self.r): min(sample.shape[0], i + self.r)])
        return median_passed


class Fourier(object):
    def __init__(self, T=10, fs=500):
        self.fs = fs
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # Only for display/figure generation
        self.domain = {0: torch.fft.rfftfreq(T*fs, 1/fs)}
        return

    def __call__(self, signal):
        return torch.fft.rfft(signal)


class InvFourier(object):
    def __init__(self, T=10, fs=500):
        self.fs = fs
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # Only for display/figure generation
        self.domain = {0: torch.linspace(0, T, T*fs)}
        return

    def __call__(self, signal):
        return torch.fft.irfft(signal)

class STFourier(object):
    def __init__(self, window_size, T=10, fs=500):
        assert (window_size < T*fs), "Specified window size greater than the signal itself"
        self.fs = fs
        self.T = T
        self.win = window_size
        self.overlap_frac = 0.5
        self.jump = math.floor(window_size * self.overlap_frac)
        self.n_windows = int(math.ceil(self.T*self.fs / self.jump))
        self.domain = {0: torch.tensor(range(0, T*fs, self.jump)) / fs,
                       1: torch.fft.rfftfreq(self.win, 1/fs)}
        self.domain_shape = (len(self.domain[0]), len(self.domain[1]))
        self.hanning = torch.hann_window(self.win)

    def __call__(self, signal):
        assert (signal.shape[0] == self.T*self.fs), "The signal is not corresponding to the specified time length and" \
                                                    "sample frequency"
        result = torch.zeros(self.domain_shape[0], self.domain_shape[1])
        signal_pad = torch.concat((signal, torch.zeros(self.win)))
        for i, L_edge in enumerate((range(0, signal.shape[0], self.jump))):
            dampened = self.hanning * signal_pad[L_edge: L_edge + self.win]
            F = torch.fft.rfft(dampened)
            result[i] = torch.abs(F)
        return result

    def view(self, signal, hyperparam):
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm,
                                        x=self.domain[1],
                                        y=self.domain[0]))
        fig.update_layout(title="Short Time Fourier Transform  -  Window size: " + str(self.win) + hyperparam)
        fig.update_yaxes(title_text="Window start time (seconds)", type='category')
        fig.update_xaxes(title_text="Power Spectrum of Window (frequency)", type='category')
        fig.show()


class PowerSpec(object):
    def __init__(self, T=10, fs=500):
        self.fs = fs
        self.T = T
        self.domain = {0: torch.fft.rfftfreq(T*fs, 1/fs)}

    def __call__(self, signal):
        return torch.pow(torch.abs(torch.fft.rfft(signal)), 2)

    def view(self, signal):
        trfm = self(signal)
        fig = go.Figure(go.Scatter(x=self.domain[0], y=trfm))
        fig.update_layout(title="PowerSpectrum Transform"), fig.update_xaxes(title_text="Frequency")
        fig.show()


class Wavelet(object):
    def __init__(self, widths, wavelet='mexh', T=10, fs=500):
        self.fs = fs
        self.T = T
        self.wavelet = wavelet
        self.widths = widths
        self.domain = { 0 : widths,
                        1: torch.linspace(0, self.T, self.T * self.fs)}
        return

    def __call__(self, signal):
        return torch.tensor(pywt.cwt(signal.numpy(), self.widths, self.wavelet)[0])

    def seeAvailableWavelets(self):
        print(pywt.wavelist(kind='continuous'))

    def view(self, signal):
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm,
                                        x=self.domain[1],
                                        y=self.domain[0]))
        fig.update_layout(title="Wavelet Transform  -  Wavelet: " + str(self.wavelet))
        fig.update_yaxes(title_text="Wavelet scale", type='category')
        fig.update_xaxes(title_text="Time (seconds)", type='category')
        fig.show()


class GramianAngularField(object):
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return

class MarkovTransitionField(object):
    def __init__(self, T=10, fs=500):
        self.T = T
        self.fs = fs

    def __call__(self):
        return

class RecurrencePlot(object):
    def __init__(self, epsilon=0.1, T=10, fs=500):
        self.fs = fs
        self.T = T

        self.epsilon = epsilon
        self.domain = {0: torch.linspace(0, T, T*fs),
                       1: torch.linspace(0, T, T*fs)}
        return

    def __call__(self, signal):
        R = torch.zeros(self.T*self.fs, self.T*self.fs, dtype=torch.int8)
        for i in range(self.T*self.fs):
            for j in range(i):
                if torch.abs(signal[i] - signal[j]) < self.epsilon:
                    R[i, j] = 1
                    R[j, i] = 1
        return R

    def view(self, signal):
        trfm = self(signal)
        fig = go.Figure(data=go.Heatmap(z=trfm))
        fig.update_layout(title="Recurrence Plot")
        fig['layout']['yaxis']['scaleanchor'] = 'x'
        fig.show()
