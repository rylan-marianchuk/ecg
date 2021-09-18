from generate import SinusoidDataSet
from transforms import Fourier
ds = SinusoidDataSet(6, transform=Fourier())
ds.viewSignal(5)


