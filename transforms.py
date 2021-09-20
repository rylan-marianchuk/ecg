"""
transforms.py

classes implementing common digital signal transformations:
Fourier, WindowedFourier, Wavelet

@author Rylan Marianchuk
September 2021
"""
import numpy as np
import torch
from scipy.signal import periodogram

class MedianPass(object):
    def __init__(self, radius, T=10, ts=500):
        """
        param radius: window size to extract median from
        """
        self.r = radius
        self.ts = ts
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # For display/figure generation in plotly
        self.domain = np.linspace(0, self.T, self.ts * self.T)  # Generic time domain

    def __call__(self, sample):
        median_passed = torch.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            median_passed[i] = torch.median(sample[max(0, i - self.r): min(sample.shape[0], i + self.r)])
        return median_passed



class Fourier(object):
    def __init__(self, T=10, ts=500):
        self.ts = ts
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # For display/figure generation
        self.domain = periodogram(torch.zeros(self.ts * self.T), self.ts)[0]  # Frequency domain
        return

    def __call__(self, signal):
        return torch.tensor(periodogram(signal, self.ts)[1])



class FourierWindowed(object):
    def __init__(self, window, step, T=10, ts=500):
        self.window = window
        self.step = step
        self.ts = ts
        self.T = T

        # Hacky but correct for now until I find a closed form
        l = []
        for i in range(0, 5000, step):
            if i + window >= 5000: break
            l.append(i)
        # 2D domain (image)
        self.domain = periodogram(torch.zeros(self.ts * self.T), self.ts)[0].shape[0], len(l)
        return

    def __call__(self, signal):
        image = torch.zeros(self.domain)
        for i in range(0, self.T * self.ts, self.step):
            fourier = periodogram(signal[i:i+self.window], self.ts)[1]
            image[:, i] = fourier
        return image


class Wavelet(object):
    def __init__(self):
        pass

    def __call__(self, signal):
        return