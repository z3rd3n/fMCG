
import sys
import os
sys.path.append(os.path.dirname(__file__))

import FUNCTIONS
from DIRECTORIES import _base_path,log_path,data_path,analysis_path
import re
import numpy as np
# import os
import Interactive_plot_base
from scipy.signal import savgol_filter
from datetime import datetime
import matplotlib.pyplot as plt
# import tkinter as tk
from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()

# print('Select .tdms file to watch: ')
# tdms_path = filedialog.askopenfilename()
# print('TDMS selected.')

print('Select directory to analyze (...P000_S00_Dxxxx-xx-xx-Gxx_evaluation):')
root_path=filedialog.askdirectory()
# print(root_path)
# root_path= str(input("Select directory to analyze (...P000_S00_Dxxxx-xx-xx-Gxx_evaluation):"))
# root_path=root_path.replace("\\","/")
file_ID=root_path.strip('/').split('/')[-1].split('_evaluation')[0]
run_list=[name for name in os.listdir(root_path)]
# print(run_list)

QRS_analyis_pic_path=root_path +'/QRS_pics'
if not os.path.exists(QRS_analyis_pic_path):
    os.makedirs(QRS_analyis_pic_path)
    
QRS_path=root_path
QRS_path+='_QRS_times.txt'

if not os.path.exists(QRS_path):
    f=open(QRS_path,mode='w')
    f.write('slice ID \t P onset x \t P onset y\t P peak x \t P peak y \t P offset x \t P offset y \t')
    f.write('Q peak x \t Q peak y \t')
    f.write('R peak x \t R peak y \t')
    f.write('S peak x \t S peak y \t')
    f.write('T onset x \t T onset y\t T peak x \t T peak y \t T offset x \t T offset y \t RR \t sigma_RR \n')
    f.close()

selected_indices={}

for run_name in run_list:
# ICA_channel = input("Which ICA channel number should be processed? ")
    try:
        if os.path.exists(root_path+'/'+run_name+'/'+run_name+'_ICA.npy'):
            data=np.load(root_path+'/'+run_name+'/'+run_name+'_ICA.npy')
            # print(data.shape)
            no_ica_chans = int(data.shape[1])
            span = 15
            offset = 100
            fs=1000
            ms2s = lambda x, _:f'{x/1000:g}'
    
            fig, ax = plt.subplots(no_ica_chans, 1, sharex=True, figsize=(18 / 2.54, 24 / 2.54), dpi=250)
            fig.subplots_adjust(hspace=0)
    
            for i in range(no_ica_chans):
                ax[i].plot(data[int(offset * fs):int(fs * (offset + span)), i], label=str(i+1))
                ax[i].legend(loc=1)
            ax[-1].xaxis.set_major_formatter(ms2s)
            ax[-1].set_xlabel('Time [s]')
            fig.suptitle(file_ID + ", " + run_name + ', ICA outputs', y=0.9)
            fname = root_path+'/'+run_name+'/'+run_name+'_ICA-picture.png'
            fig.savefig(fname, format='png')
            plt.show()
            # plt.close()
            
            
            while True:
                mother_index=input('Provide index of mother: [0, 1, etc.]: ')
                try:
                    mother_index = int(mother_index)-1
                    break
                except ValueError:
                    print('no valid entry')
            while True:
                
                fetus_index=input('Provide index of fetus: [0, 1, etc.]: ')
                try:
                    fetus_index = int(fetus_index)-1
                    break
                except ValueError:
                    print('no valid entry')
                    
            selected_indices[run_name]={'mother_index':mother_index,'fetus_index':fetus_index}
    
    
            window_size=int(1400)
            time_axis=np.linspace(0,int(window_size)-1,num=int(window_size))
            
            beat, detect = FUNCTIONS.matlab_hos(data[:,selected_indices[run_name]['fetus_index']], window_size)
            RR, sigma_RR, peaks, popt_RRs, yy, xx = FUNCTIONS.detect_beats(detect)
            try:
                savgol_median = savgol_filter(np.diff(peaks), 41, 3)
                avg_peaks_list = peaks[np.where(np.square((np.diff(peaks) - savgol_median) / sigma_RR) < 2)]
            except:
                avg_peaks_list = peaks
                # print('except on savgol')
            peaks = avg_peaks_list
            if len(peaks)<3:
                print(run_name+' - skipped (low peak count)')
            else:
                slice_box, avg_mean,worked = FUNCTIONS.avg_based_QRScomplex(data[:,selected_indices[run_name]['fetus_index']],
                                                                           peaks,sigma_RR, box_size=window_size)
                segmentation_data=Interactive_plot_base.interactive_ploter(np.array(slice_box),
                                                                           time_axis-window_size/2,
                                                                           add_lines=[avg_mean,beat],
                                                                           sideplot=[xx,yy,popt_RRs,RR,sigma_RR,0],
                                                                           savename=[QRS_analyis_pic_path,
                                                                                     run_name])
                f=open(QRS_path,mode='a')
                str_buffer = run_name
                for key in ['P','Q','R','S','T']:
                    for pos in segmentation_data[key]:
                        for entryy in pos:
                            str_buffer += '\t' + str(entryy)
                f.write(str_buffer + '\t'+str(RR)+'\t'+str(sigma_RR)+'\n')
                f.close()
                print(run_name+' - done')

    except:
        continue
