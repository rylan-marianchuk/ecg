import torch

from generate import SinusoidDataSet
from models import MLP
from transforms import Fourier, MedianPass, FourierWindowed, Wavelet
from torch.utils.data import DataLoader
import torch.nn as nn

sinusoids = 3
epochs = 12
# Since in generation I capped the frequencies to 15.5, fourier will not spike after the 160th element, thus redundant
input_size = 160


# ------------------ All to be replaced by new ECG dataset -------------------------------------
dataset_tr = SinusoidDataSet(50000, 3, load=True, transform=Fourier())
dataset_te = SinusoidDataSet(20000, 3, load=True, transform=Fourier())

dataset_tr.data = torch.load("./data-sins-3-len-70000.pt")[:50000]
dataset_tr.data_params_freq = torch.load("./freq-sins-3-maxfreq-15_5-len-70000.pt")[:50000]
dataset_tr.data_params_amp = torch.load("./amp-sins-3-len-70000.pt")[:50000]

dataset_te.data = torch.load("./data-sins-3-len-70000.pt")[50000:]
dataset_te.data_params_freq = torch.load("./freq-sins-3-maxfreq-15_5-len-70000.pt")[50000:]
dataset_te.data_params_amp = torch.load("./amp-sins-3-len-70000.pt")[50000:]
# ------------------ ------------------------------------- -------------------------------------


X_tr = DataLoader(dataset_tr, batch_size=64, num_workers=3)
X_te = DataLoader(dataset_te, batch_size=64, num_workers=3)

model = MLP(input_size, sinusoids)
print(model)

L = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.001)


for _ in range(epochs):
    training_loss = 0
    for X, y in X_tr:
        prediction = model(X[:, :input_size])
        loss = L(prediction, y)
        training_loss += loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("Done epoch " + str(_))
    print("Training loss: " + str(training_loss))

L_performance = 0
with torch.no_grad():
    for X,y in X_te:
        prediction = model(X[:, :input_size])
        L_performance += L(prediction, y).item()
print(L_performance)


print()
