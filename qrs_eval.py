import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import functions
import hosd_python.hosd as hos
import interactive_analysis

data=np.load('patient_ICA/P045_S01_R002_ICA.npy')
window_size=int(1400)
time_axis=np.linspace(0,int(window_size)-1,num=int(window_size))

# These indexes should be chosen manually from ICA data everytime
mother_index = 7 # manually selected for P45 R002
fetus_index = 6 # manually selected for P45 R002

# calls matlab engine and executes a matlab script with the given data
# beat, detect = functions.matlab_hos(data[:,fetus_index], window_size)

# python equivalent of the matlab script
beat_py, detect_py = hos.apply_hosd(data[:,fetus_index], window_size)
RR, sigma_RR, peaks, popt_RRs, yy, xx = functions.detect_beats(detect_py)

try:
    savgol_median = savgol_filter(np.diff(peaks), 41, 3)
    avg_peaks_list = peaks[np.where(np.square((np.diff(peaks) - savgol_median) / sigma_RR) < 2)]
except:
    avg_peaks_list = peaks

peaks = avg_peaks_list
if len(peaks)<3:
    print('skipped (low peak count)')
else:
    slice_box, avg_mean,worked = functions.avg_based_QRScomplex(data[:,fetus_index],
                                                                peaks,sigma_RR, box_size=window_size)
    segmentation_data=interactive_analysis.interactive_ploter(np.array(slice_box),
                                                                time_axis-window_size/2,
                                                                add_lines=[avg_mean,beat_py],
                                                                sideplot=[xx,yy,popt_RRs,RR,sigma_RR,0])
    