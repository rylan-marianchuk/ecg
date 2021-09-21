import numpy as np

from transforms import *
from generate import SinusoidDataSet

FW = FourierWindowed(window=50, step=20, T=1)
WVLT = Wavelet(np.linspace(0.1, 20, 35), T=1)
ds = SinusoidDataSet(6, 3, T=1, transform=FW)
ds.viewSignal(5)
ds.viewTrueSignal(5)
