from hosobject import *
from scipy.io import loadmat
from scipy.sparse import csr_matrix


lowpass = 90 # Low pass filter cutoff frequency in Hz
Fs = 1000 # Sampling frequency
N = 1000 
mat_data = loadmat('serden/matlab_analysis/HOSD-master/P44_R1_ICA_channel9.mat')['data'].T
mat_data_hosd = loadmat('serden/matlab_analysis/HOSD-master/P44_R1_ICA_channel9_hosd.mat')['output_2'].T
hos = hosobject(3)
hos.initialize(N, Fs, lowpass)

hos.get_block(mat_data, maxiter=25)

output_1 = hos.waveform[:, None]
output_2 = csr_matrix(hos.apply_filter(mat_data)[0]).toarray().T

beat = np.array([item for sublist in output_1 for item in sublist])
beat = np.roll(beat, int(len(beat) / 2))
detect = np.array([item for sublist in output_2 for item in sublist])


