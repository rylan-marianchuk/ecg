"""
transforms.py



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
    def __init__(self, radius, step, T=10, ts=500):
        self.radius = radius
        self.step = step
        self.ts = ts
        self.T = T
        # X values that couple with the return structure of __call(sample)__
        # For display/figure generation
        self.domain = periodogram(torch.zeros(self.ts * self.T), self.ts)[0]  # Frequency domain
        return

    def __call__(self, signal):
        return torch.fft.rfftfreq(signal)


class Wavelet(object):
    def __init__(self):
        pass

    def __call__(self, signal):
        return