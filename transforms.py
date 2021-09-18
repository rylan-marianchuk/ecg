"""
transforms.py



@author Rylan Marianchuk
September 2021
"""

import torch

class MedianPass(object):
    def __init__(self, radius):
        """
        param radius: window size to extract median from
        """
        self.r = radius

    def __call__(self, sample):
        median_passed = torch.zeros(sample.shape[0])
        for i in range(sample.shape[0]):
            median_passed[i] = torch.median(sample[max(0, i - self.r): min(sample.shape[0], i + self.r)])
        return median_passed

class Fourier(object):
    def __init__(self):
        return

    def __call__(self, signal):
        FT = torch.fft.rfft(signal)
        freqs = torch.fft.rfftfreq(signal.shape[0], 1 / 500)
        PS =  torch.abs(FT)**2
        return PS[torch.argsort(freqs)]




class FourierWindowed(object):
    def __init__(self, radius, step):
        self.radius = radius
        self.step = step
        return

    def __call__(self, signal):
        return torch.fft.rfftfreq(signal)
