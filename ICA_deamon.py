# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:33:54 2023

@author: Qspin
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))


import FUNCTIONS
from functools import partial
import tkinter as tk
from tkinter import filedialog
from nptdms import TdmsFile
# import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import GAplot_single_patient



# import time
# root = tk.Tk()
# root.withdraw()

'''
große Karte (32 Kanäle): P012,
kleine Platte (16 Kanäle, unterschiedlich ausgerichtet): P011 R001, R002, P013, P014
Patienten ab 15 (einschließlich) mit großer Karte (NI 9205) aufgenommen. Patienten vorher (bis 14) mit kleiner Karte aufgenommen.
ab Patientin 15 sicher im default Modus aufgenommen. d.h. hier gilt: Kanäle 0-7, 16-23 differetiell aufgenommen, Kanäle 8-15, 24-31 gegen Erde
bis Patientin 14 (NI 62..) 0-7 differentiell, 8-15 gegen Erde im default Modus. Nicht sicher, welche Patienten im default Modus gemessen wurden
'''

def LSD(root_path, tdms_path, groupname, selected_idx, i):
    print("Start")
    
    bpfc=[4,5] # bandstop filter
    bsfc_new=3 # the critical frequency for lowpass filter
    lpfc_new=70 # the critical frequency for lowpass filter
    lpo_new=2 # order of the lowpass filter
    bso_new=3 # order of the lowpass filter
    bsto_new=2 # bandstop filter
    Freqencytoremove_new=50 # Frequency to remove for notch filter
    Qualityfactor_new=40

    data_array=FUNCTIONS.array_from_TDMSgroup(tdms_path,groupname)
    
    #Place here the current version of desired ICA treatment
    #################
    data_array = FUNCTIONS.butter_filter(data_array, lpfc=lpfc_new, lpo=lpo_new, bsfc=bsfc_new)#, bstfc=bpfc
    # data_array = FUNCTIONS.butter_filter(data_array, lpo=2) dd, lpfc=70, lpo=3, bsfc=1, bso=3, Freqencytoremove=50, Qualityfactor=40

    data_select = np.zeros((len(data_array[:,0]),len(selected_idx)))


    
    for number, channel in enumerate(selected_idx):
        data_select[:,number] = data_array[:,channel]
   
    ms2s = lambda x, _:f'{x/1000:g}' 
   
    #Plot raw data after filtering
    
    # fig1, ax = plt.subplots(len(selected_idx), 1, sharex=True, figsize=(18 / 2.54, 24 / 2.54), dpi=250)
    # for n, idx in enumerate(selected_idx):
    #     ax[n].plot(data_array[5000:,idx], label=f'{idx}')
    #     ax[n].legend(loc=1)
    # ax[-1].xaxis.set_major_formatter(ms2s)
    # ax[-1].set_xlabel('Time [s]')
    # fname =  root_path+'/'+groupname+'/'+f'ICA_{i}_raw_after_filter.png'
    # fig1.savefig(fname, format='png')
    # plt.show()
    # plt.close()
    
   
    print("Filter finished")
    S_ = FUNCTIONS.FASTICA(data_array[5000:,:],i)
    print("ICA finished")
    # ######################################
    np.save(root_path+'/' + groupname + '/'+groupname+'_ICA',S_)
    
    no_ica_chans = int(i)
    span = 5
    offset = 100 
    fs=1000
    
    
    fig, ax = plt.subplots(no_ica_chans, 1, sharex=True, figsize=(18 / 2.54, 24 / 2.54), dpi=250) #
    fig.subplots_adjust(hspace=0)
    
    for j in range(no_ica_chans):
        ax[j].plot(S_[int(offset * fs):int(fs * (offset + span)), j], label=str(j+1))
        ax[j].legend(loc=1)
    ax[-1].xaxis.set_major_formatter(ms2s)
    ax[-1].set_xlabel('Time [s]')
    fig.suptitle( f'{i} ICA outputs, bandpass:{bpfc} Hz with {bsto_new} order,\n 1.lowpass: {lpfc_new} Hz with {lpo_new} order, 2.lowpass: {bsfc_new} Hz with {bso_new} order, \
                 \n Cutoff F:{Freqencytoremove_new} Hz with Qualityfactor: {Qualityfactor_new}', y=0.95)
    fname =  root_path+'/'+groupname+'/'+f'{i}_ICA-picture.png'
    fig.savefig(fname, format='png')
    plt.show()
    plt.close()
    print(f'{i} ICA outputs, bandpass:{bpfc} Hz with {bsto_new} order,\n 1.lowpass: {lpfc_new} Hz with {lpo_new} order, 2.lowpass: {bsfc_new} Hz with {bso_new} order, \
                 \n Cutoff F:{Freqencytoremove_new} Hz with Qualityfactor: {Qualityfactor_new}')


def main():
    tdms_path= str(input("Select .tdms file to watch: "))
    tdms_path=tdms_path.replace("\\","/")
    tdms_file = TdmsFile.read(tdms_path)
    file_name=os.path.basename(tdms_path).split('.tdms')[0]
    # print('Select directory to place intermdiate data and results:')
    # root_path=filedialog.askdirectory()
    root_path = str(input("Select directory to place intermediate data and results:"))
    root_path=root_path.replace("\\","/")
    root_path+='/'+file_name+'_evaluation/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    print('Results path: '+str(root_path))

    default_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

    while True:
        selected_idx = input("Enter the inidices : ").split()
        try:
            if selected_idx == ['default']:
                selected_idx = default_idx
                print("Using default indices: ", selected_idx) 
                break
            elif [int(item) for item in selected_idx]:
                selected_idx = [int(item) for item in selected_idx]
                selected_idx.sort()
                print("Indices that will be used: ", selected_idx)
                break
        except:
            print("Please insert numbers or type default")
        
    while True:
        try:
            
            ICA_channels = int(input("Number of channels for ICA: "))
            break
        except:
            print("Please insert a number")
            
    print('Start main routine:')
     
    #Start the main daemon loop
    processed_groups=[name for name in os.listdir(os.path.dirname(root_path)) if os.path.isdir(root_path+name)]
    for group in tdms_file.groups():
        if group.name not in processed_groups:
            while True:
                process_toggle = input("Process run: "+group.name +' ? [y/n]')
                if process_toggle not in ('y', 'n'):
                    print("Please respond with y for yes or n for no")
                else:
                    break
                
            os.makedirs(root_path + '/' + group.name + '/')
            if process_toggle=='y':
                process_toggle=True
                groupname= group.name
                LSD(root_path, tdms_path, groupname, selected_idx, ICA_channels)
                #Test different numbers of ICA channels in a Loop
                # pool = Pool(5)
                # pool.map(partial(LSD, root_path, tdms_path, groupname, selected_idx), range(ICA_channels,))
                # pool.join()
                # pool.close()
            
        else:
            process_toggle=False
            f=open(root_path+'/'+group.name+'/This_is_a_placeholder_delete_folder_to_reprocess_this_run.txt',mode='w')
            f.close()
        
    #Plot different parameters over gestation week
    # GAplot_single_patient.make_GA_plot(interval_times_path,root_path+'/'+file_ID+'_GAplot.png',file_ID)

if __name__ == '__main__':
    main()
    
    
    

