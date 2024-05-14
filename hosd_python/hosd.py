from hosd_python.hosobject import *
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def apply_hosd(data_in, window_size):
	lowpass = 90 # Low pass filter cutoff frequency in Hz
	Fs = 1000 # Sampling frequency
	N = window_size 
	hos = hosobject(3)
	hos.initialize(N, Fs, lowpass)

	hos.get_block(data_in, maxiter=25)

	output_1 = hos.waveform[:, None]
	output_2 = csr_matrix(hos.apply_filter(data_in)[0]).toarray().T
	beat = np.array([item for sublist in output_1 for item in sublist])
	beat = np.roll(beat, int(len(beat) / 2))
	detect = np.array([item for sublist in output_2 for item in sublist])
	return beat, detect

 