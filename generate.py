import numpy as np
import plotly.graph_objs as go
import torch
from torch.utils.data import Dataset

class SinusoidDataSet(Dataset):

    def __init__(self, size, T=50, fs=500, transform=None):
        """
        Initialize a experimental sinusoid signal dataset to test transforms with
        """
        self.transform = transform

        self.n = size
        # Time of signal in seconds
        self.T = T
        # How are these seconds sampled, Hz?
        self.fs = fs

        # Matrix (the dataset) to generate, each row holds a signal
        self.data = torch.zeros(size, self.T * self.fs)

        # The parameters used to generate each signal (row)
        # self.data_params[i] holds a two row np array:
            # First row: frequency coefs of sinusoid
            # Second row: amplitude coefs of sinusoid
            # Columns follow possion distribution (number of sinusoids summed in signal)
        self.data_params = dict()

        # Generate the signals
        for i in range(size):
            self.data[i], self.data_params[i] = self.getSignal()


    def __len__(self):
        return self.n


    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx])
        return self.data[idx]


    def viewTrueSignal(self, idx):
        """
        Generate a 2D plot of the non-transformed signal using web browser and plotly
        Add text revealing the true frequency and amplitude coefficients of the signal
        :param item: the index of signal to view
        """
        fig = go.Figure(go.Scatter(y=self.data[idx]))
        fig.update_layout(title="freq:" + str(self.data_params[idx][0])
                                + "\n\namp:" + str(self.data_params[idx][1]))
        fig.show()


    def viewSignal(self, idx):
        """
        Generate a 2D plot of the non-transformed signal using web browser and plotly
        Add text revealing the true frequency and amplitude coefficients of the signal
        :param item: the index of signal to view
        """
        if self.transform is not None:
            freqs = torch.fft.rfftfreq(self.data[idx].shape[0], 1 / 500)
            fig = go.Figure(go.Scatter(x=freqs[torch.argsort(freqs)], y=self.transform(self.data[idx])))
        else:
            fig = go.Figure(go.Scatter(y=self.data[idx]))

        fig.update_layout(title="freq:" + str(self.data_params[idx][0])
                                + "\n\namp:" + str(self.data_params[idx][1]))
        fig.show()

    def getSignal(self):
        """

        :return:
        """
        signal = torch.zeros(self.T * self.fs)

        # How many sin functions to sum in the signal follows a poission distribution, +1 to shift support over 1 not 0
        sinusoids = np.random.poisson(1.25) + 1

        # Obtain random scale an amplitude coefficients for each sinusoid
        freq_coefs = np.abs(np.random.normal(0, 1.75, sinusoids))
        amp_coefs = np.random.normal(0, 1.75, sinusoids)

        # Obtain the signal by evaluating the sinusoids
        for i,v in enumerate((np.linspace(0, self.T, self.T * self.fs))):
            signal[i] = sum(amp_coefs[j] * np.sin(freq_coefs[j] * v) for j in range(sinusoids))
        return signal, np.vstack((freq_coefs, amp_coefs))
