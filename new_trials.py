import functions
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import functions
import hosd_python.hosd as hos
import interactive_analysis

data = functions.array_from_TDMSgroup("patients/P044/P044_S01_D2024-04-18_G36.tdms", "R001")
data = data[:,17]
window_size=int(1400)
time_axis=np.linspace(0,int(window_size)-1,num=int(window_size))

# python equivalent of the matlab script

beat_py, detect_py = hos.apply_hosd(data, window_size)


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
    slice_box, avg_mean,worked = functions.avg_based_QRScomplex(data,peaks,sigma_RR, box_size=window_size)
    segmentation_data=interactive_analysis.interactive_ploter(np.array(slice_box),
                                                                time_axis-window_size/2,
                                                                add_lines=[avg_mean,beat_py],
                                                                sideplot=[xx,yy,popt_RRs,RR,sigma_RR,0])