# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:25:42 2023

@author: Qspin
"""
import os
import re
import numpy as np
import sys
from datetime import datetime
from nptdms import TdmsFile
from sklearn.decomposition import FastICA as skFastICA
from scipy.signal import butter,periodogram,sosfilt,iirnotch,filtfilt,welch,find_peaks, correlate,savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings
import neurokit2 as nk
import matplotlib.pyplot as plt
from copy import deepcopy


def array_from_TDMSgroup(file_dir,group_name,log_dict=None,cuts=None,omits=False):
    tdms_file = TdmsFile.read(file_dir)

    if log_dict==None:
        scaling=2.7e-3 # V/pT  from 2.7 #V/nT
    else:
        scaling=log_dict['base_infos']['scaling']


    data_dict={}

    group=tdms_file[group_name]
    group_name = group.name #This is redundant, should be used to test input instead
    for i,channel in enumerate(group.channels()):
        channel_name = channel.name
        # Access dictionary of properties:
        properties = channel.properties
        #print(properties)
        # Access numpy array of data for channel:
        data_dict[i] = channel.read_data(scaled=True)
    no_of_channels=len(data_dict)
    #Create common time axis using the scaling from the file
    data_dict[-1]=np.linspace(0,(len(data_dict[0])-1)*properties['wf_increment'],len(data_dict[0]))
    fs=1/data_dict[-1][1]

    #Create an array with the content as the dict:
    data_array=np.zeros((len(data_dict[-1]),no_of_channels+1))
    data_array[:,0]=data_dict[-1]

    for i in range(no_of_channels):
        data_array[:,i+1]=data_dict[i]/scaling

    # Apply omits if they exists
    if omits:
        omits = log_dict['group_info'][group_name]['omits']
        chans_to_remove = omits  # List of chan no from 1 to 16!!
        chans_to_pick = list(range(0, no_of_channels+1))
        for elem in chans_to_remove:
            try:
                chans_to_pick.remove(elem)
            except:
                print('tried to remove a channel which was not there')
        data_array = data_array[:, chans_to_pick]
    # Apply cuts

    if cuts!=None:
        data_array=data_array[int(cuts[0]*fs):int(cuts[1]*fs),:]

    return data_array

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def detect_beats(inp):
    #print(inp)
    #print(np.shape(inp))
    # bin the distribution of the filter detection data so we know the contrast of noise to peaks
    y, bins = np.histogram(inp, bins=40)
    x = (bins[1:] + bins[:-1]) * 0.5

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(gaus, x, y, p0=[np.max(y), 0, 1])
    except:
        popt = [np.max(y), 0, np.std(inp)]

    # detect heart beats
    peaks, properties = find_peaks(inp,
                                   height=np.abs(popt[2]),
                                   distance=300, prominence=0.6)
    # fit guassion to get RR
    y, bins = np.histogram(np.diff(peaks), bins=np.linspace(300, 1200, num=int((1200 - 300) / 2 + 1)))
    x = (bins[1:] + bins[:-1]) * 0.5

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(gaus, x, y, p0=[np.max(y), 450, 80])
    except:
        popt = [np.max(y), np.mean(np.diff(peaks)), np.std(np.diff(peaks))]

    # plt.plot(x,y)
    # plt.plot(x,gaus(x,popt[0],popt[1],popt[2]))
    return (popt[1], np.abs(popt[2]), peaks, popt, y, x)



def avg_based_QRScomplex(ica_data,avg_peaks_list, sigma_RR, box_size=1000):
    #print('len peaks' +str(len(avg_peaks_list)))
    # peaks......array with inideces of R-peak positions
    # RR......RR time in ms
    # sigma_RR ..... uncertainty of RR time/ average spread
    # box_size ....... size of the QRS complex feature use: box_size=len(beat)!

    # Outputs

    # array with all available single beats from peaks
    # mean total beat

    box_size = int(box_size)

    # Remove outliers which are more than sigma_RR away from peaks list
    slice_box = []

    for j, val in enumerate(avg_peaks_list):
        if int(val - box_size / 2) > 0 and int(val + box_size / 2) < len(ica_data):
            s = ica_data[int(val - box_size / 2):int(val + box_size / 2)]
            x = np.linspace(0, box_size - 1, num=box_size)
            slope, intercept, r, p, se = linregress(x, s)
            slice_box.append(s - x * slope - intercept)
    slice_box = np.array(slice_box)

    return slice_box.T - slice_box.T.mean(axis=0, keepdims=True), np.mean(slice_box, axis=0),True

def butter_filter(dd, lpfc=70, lpo=3, bsfc=3, bso=3, bstfc=[4,6], bsto=2, Freqencytoremove=50, Qualityfactor=40):
    #Filter data
    ''' lpfc: the critical frequency for lowpass filter
        lpo: order of the lowpass filter
        bsfc: the critical frequency for lowpass filter
        bso: order of the lowpass filter
        Freqencytoremove: Frequency to remove for notch filter
        '''
    fs=1000
    # lpfc=70
    # lpo=3
    # bsfc=1
    # bso=3
    band_stop=butter(bsto,bstfc,btype='bandstop',output='sos',fs=fs)
    lowpass=butter(lpo,lpfc,btype='low',output='sos',fs=fs)
    wavy_baseline=butter(bso,bsfc,btype='low',output='sos',fs=fs)
    b_notch, a_notch = iirnotch(Freqencytoremove, Qualityfactor, fs)
    filtered = deepcopy(dd)
    for i in range(1,np.shape(dd)[1]):
        filtered[:,i]=filtfilt(b_notch, a_notch, filtered[:,i])
        filtered[:,i]=sosfilt(lowpass,filtered[:,i])
        filtered[:,i]=sosfilt(band_stop,filtered[:,i])
        filtered[:,i]-=sosfilt(wavy_baseline,filtered[:,i])
        # dd[:,i]=denoise_wavelet(dd[:,i],method='BayesShrink',
        #                         mode='soft',wavelet_levels=3,
        #                         wavelet='sym8',rescale_sigma='True')
    # dd = dd[2000:12000, :]      ##we only pick 12 s of each for now
    return filtered

def FASTICA(dd,n_comp=None,random=0):
    if n_comp==None:
        n_comp=int(np.shape(dd)[1]-1)
    #print('Number of components Fast ICA: '+str(n_comp))
    ica = skFastICA(n_components=n_comp,algorithm='deflation',max_iter=4000,fun='logcosh',tol=1e-5,random_state=random)
    S_ = ica.fit_transform(dd[:,1:])
    return S_

def RR2BPM(RR):
    return 6e4/(RR+1e-10)

def BPM2RR(BPM):
    return 6e4/(BPM+1e-10)

def waveform_plot(container, time_axis, add_lines=None, sideplot=None, savename=None):
    ##### Create the base plot:
    fig, (ax,bx) = plt.subplots(1, 2, figsize=(30 / 2.54, 18 / 2.54), gridspec_kw={'width_ratios': [3, 1]}, dpi=120)
    ax.grid(alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.7)
    ax.set_ylabel("Signal [A.U.]")
    ax.set_xlabel("Time [ms]")

    if sideplot!=None:
        xx, yy, popt_RRs, RR, sigma_RR,hos=sideplot
        bx.grid(alpha=0.7)
        bx.minorticks_on()
        bx.grid(True, which='minor', linestyle='dotted', alpha=0.7)
        bx.plot(xx,yy)
        bx.plot(xx,gaus(xx,popt_RRs[0],popt_RRs[1],popt_RRs[2]))
        secaxb = bx.secondary_xaxis('top', functions=(BPM2RR, RR2BPM))
        secaxb.set_xlabel('BPM')
        secaxb.set_xticks(ticks=[200,150,120,100,80,75,60])
        bx.axvline(x=RR, color='r',linestyle='dashed')
        bx.text(RR*1.05,np.max(yy)*0.9,'RR: %i $\pm$ %i'% (int(RR),int(sigma_RR)),color='r')
        bx.set_xlabel('RR [ms]')
        bx.set_xlim([300,1000])
        bx.text(700,np.max(yy)*0.85,'# Peaks: %i' % (int(np.sum(yy))))
        bx.text(700, np.max(yy) * 0.8, '# HOS: %f' % (float(hos)))
    ax.set_ylabel("Signal [A.U.]")
    ax.set_xlabel("Time [ms]")
    fig.tight_layout()

    # Plot the container data
    x = time_axis
    container = container.T  # Transpose if necessary
    for j in range(container.shape[0]):
        ax.plot(x, container[j, :], color='b', alpha=0.05, lw=1)
    
    if add_lines is not None:
        ax.plot(x, add_lines[0], color='r', label='AVGs')
        ax.plot(x, add_lines[1]/np.std(add_lines[1])*np.std(add_lines[0]), color='k', label='Bispectral Filters')
        if len(add_lines) == 3:
            ax.plot(x, add_lines[2], color='g', label='Theory')
        ax.legend(loc='upper right')

    fig.tight_layout()

    if savename is not None:
        plt.savefig(savename)

    plt.show()

    return fig, ax  # Optional, in case you want to further manipulate the figure outside the function