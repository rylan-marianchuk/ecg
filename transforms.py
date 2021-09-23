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
        fig.update_layout(title="Wavelet Transform\nWavelet: " + str(self.wavelet))
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
