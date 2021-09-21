"""
transforms.py

classes implementing common digital signal transformations:
Fourier, WindowedFourier, Wavelet, Median pass filter

@author Rylan Marianchuk
September 2021
"""
import numpy as np
import torch
from scipy.signal import periodogram, cwt
from scipy.signal import ricker, morlet

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
        self.domain = np.linspace(0, self.T, self.fs * self.T)  # Generic time domain

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
        self.domain = { 0 : periodogram(torch.zeros(self.fs * self.T), self.fs)[0] } # Frequency domain
        return

    def __call__(self, signal):
        return torch.tensor(periodogram(signal, self.fs)[1])



class FourierWindowed(object):
    def __init__(self, window, step, T=10, fs=500):
        self.window = window
        self.step = step
        self.fs = fs
        self.T = T

        # Hacky but correct for now until I find a closed form
        l = []
        for i in range(0, self.T * self.fs, step):
            if i + window >= self.T * self.fs: break
            l.append(i)
        # 2D domain (image)
        self.domain = {0: periodogram(torch.zeros(self.window), self.fs)[0].shape[0],
                       1: len(l)}
        return

    def __call__(self, signal):
        image = torch.zeros(self.domain[0], self.domain[1])
        for i,p in enumerate((range(0, self.T * self.fs, self.step))):
            if p + self.window >= self.T * self.fs: break
            fourier = periodogram(signal[p:p+self.window], self.fs)[1]
            image[:, i] = torch.tensor(fourier)
        return image


class Wavelet(object):
    def __init__(self, widths, T=10, fs=500):
        self.fs = fs
        self.T = T
        self.wavelet = ricker
        self.widths = widths
        self.domain = { 0 : widths,
                        1: np.linspace(0, self.T, self.T * self.fs)}
        return

    def __call__(self, signal):
        return torch.tensor(cwt(signal, ricker, self.widths, dtype=np.float32))

