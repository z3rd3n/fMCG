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
from scipy.stats import linregress, kurtosis
from scipy.optimize import curve_fit
import warnings
import neurokit2 as nk
import matplotlib.pyplot as plt
from copy import deepcopy
import pywt

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

def bandpass_filter(dd, order=5, low = 0.5, high = 50):
    bandpass = butter(order, [low, high], btype='band', output='sos', fs=1000)
    b_notch, a_notch = iirnotch(50, 40, fs=1000)

    # the first channel is uncessary and the first 5000 samples are skipped for transient sensor effects
    data = deepcopy(dd[5000:, 1:])
    for i in range(data.shape[1]):
        data[:,i] = sosfilt(bandpass, data[:,i])
        data[:,i] = filtfilt(b_notch, a_notch, data[:,i])
    return data

def FASTICA(dd,n_comp=None, algo = "parallel"):
    ica = skFastICA(n_components=n_comp,algorithm=algo,max_iter=500,fun='logcosh',tol=1e-5)
    S_ = ica.fit_transform(dd)
    return S_, ica

from scipy.interpolate import CubicSpline

def upDownSample(maternal_matrix, maternal_indices, downsample_factor_qrs=5, downsample_factor_other=20):

    for index in maternal_indices:
        candidate = maternal_matrix[:, index]
        peak_indices, _ = find_peaks(np.abs(candidate), distance=400, prominence=2)
        
        # Create a mask for downsampling
        downsample_mask = np.zeros_like(candidate, dtype=bool)
        
        for peak in peak_indices:
            start_qrs = max(0, peak - 50)
            end_qrs = min(len(candidate), peak + 50)
            downsample_mask[start_qrs:end_qrs] = True
        
        # Downsample
        downsampled_signal = []
        indices = []
        for i in range(len(candidate)):
            if downsample_mask[i]:
                if i % downsample_factor_qrs == 0:
                    downsampled_signal.append(candidate[i])
                    indices.append(i)
            else:
                if i % downsample_factor_other == 0:
                    downsampled_signal.append(candidate[i])
                    indices.append(i)
        
        # Convert to numpy arrays
        downsampled_signal = np.array(downsampled_signal)
        indices = np.array(indices)
        
        # Perform cubic spline interpolation
        cs = CubicSpline(indices, downsampled_signal)
        upsampled_signal = cs(np.arange(len(candidate)))
        
        # Replace the original signal with the upsampled signal
        maternal_matrix[:, index] = upsampled_signal
    
    return maternal_matrix

def plotICA(sources, ica = None, kurt_threshold=2, offset=10000, span=3000, fs=1000, mse_th=1):
    kurtosis_values = [kurtosis(sources[:, j]) for j in range(sources.shape[1])]
    kurt_list = [j for j, kurt in enumerate(kurtosis_values) if kurt_threshold < kurt < 200]
    channels_to_plot = []
    for chan in kurt_list:
        signal = sources[offset:offset+span, chan]
        peaks, _ = find_peaks(np.abs(signal), distance=400, prominence=2)
        peak_heights = signal[peaks]
        mean_height = np.mean(peak_heights)
        mse = np.mean((peak_heights - mean_height)**2)
        
        # Normalize MSE by the square of the mean height to make it scale-invariant
        normalized_mse = mse / (mean_height**2)
        
        # Check if normalized MSE is below threshold (smaller is better)
        if normalized_mse < mse_th:
            channels_to_plot.append(chan)
    
    fig, axes = plt.subplots(len(channels_to_plot), 1, sharex=True, 
                             figsize=(12, 2 * len(channels_to_plot)), dpi=150)
    fig.subplots_adjust(hspace=0.3)
    
    if len(channels_to_plot) == 1:
        axes = [axes]
    
    maxPeaks = 0
    compF = -1
    for i, j in enumerate(channels_to_plot):
        signal = sources[offset:offset+span, j]
        peaks, _ = find_peaks(np.abs(signal), distance=400, prominence=2)
        if len(peaks) > maxPeaks:
            maxPeaks = len(peaks)
            compF = j
        
        axes[i].plot(np.arange(span) / fs, signal, label=f'Channel {j}')
        axes[i].plot(peaks / fs, signal[peaks], "rx", markersize=8)
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(loc='upper right')
        axes[i].text(0.02, 0.95, f'Kurtosis: {kurtosis_values[j]:.2f}\nPeaks: {len(peaks)}', 
                     transform=axes[i].transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    axes[-1].set_xlabel('Time [s]')
    params = ica.get_params()
    plt.suptitle(f'ICA {len(channels_to_plot)} channels with kurtosis > {kurt_threshold}, algo:{params['algorithm']}, comps:{params['n_components']}', fontsize=16)

    
    plt.tight_layout()
    plt.show()
    return compF, channels_to_plot

def wavelet_denoise(averaged_waveform, wavelet='db4', level=5):
    coeffs = pywt.wavedec(averaged_waveform, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=0.7, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def identify_hr_range_segments(heart_rates, min_hr, max_hr, plot=False, min_segment_length=5):
    range_segments = []
    start_index = None
    
    for i, hr in enumerate(heart_rates):
        if min_hr <= hr <= max_hr:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                if i - start_index >= min_segment_length:
                    range_segments.append((start_index, i))
                start_index = None
    
    # Check if the last segment extends to the end
    if start_index is not None and len(heart_rates) - start_index >= min_segment_length:
        range_segments.append((start_index, len(heart_rates)))
    if plot:
        plt.figure(figsize=(15, 6))
        plt.plot(heart_rates, 'b-', label='Heart Rate')
        for start, end in range_segments:
            plt.axvspan(start, end, color='green', alpha=0.3)
        plt.axhline(y=60, color='r', linestyle='--', label='Min HR')
        plt.axhline(y=150, color='r', linestyle='--', label='Max HR')
        plt.title(f"Heart Rate with {min_hr}-{max_hr} BPM Range Segments Highlighted")
        plt.xlabel("Beat Number")
        plt.ylabel("Heart Rate (BPM)")
        plt.legend()
        plt.show()
    return range_segments

def __avg_hr_range(data, peaks, heart_rates, window_size, min_hr=0, max_hr=200):
    half_window = window_size // 2
    range_segments = identify_hr_range_segments(heart_rates, min_hr, max_hr)
    
    summed_waveforms = []
    for start, end in range_segments:
        segment_peaks = peaks[start:end]
        for peak in segment_peaks:
            if peak - half_window < 0 or peak + half_window >= len(data):
                continue
            window = data[peak - half_window : peak + half_window]
            summed_waveforms.append(window)
    
    if not summed_waveforms:
        return None
    
    averaged_waveform = np.mean(summed_waveforms, axis=0)
    return averaged_waveform

def avg_channels_hr_range(data, peaks, heart_rates, window_size, denoise=False, plot=False, min_hr=0, max_hr=200):
    if plot:
        plt.figure(figsize=[20,10])

    for ch in range(1, data.shape[1]):
        averaged_waveform = __avg_hr_range(data[:, ch], peaks, heart_rates, window_size=1200, min_hr=min_hr, max_hr=max_hr)
        if denoise:
            averaged_waveform = wavelet_denoise(averaged_waveform)
        if averaged_waveform is not None and plot:
            plt.plot(averaged_waveform, color='k')
    if plot:
        plt.title(f"Averaged Waveform from Heart Rate Range {min_hr}-{max_hr} BPM")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
    return averaged_waveform

def plotHR(signal, h=2, d=400, fs=1000, minBpm=-1, maxBpm=-1):
    # Find all peaks
    peaks, _ = find_peaks(np.abs(signal), prominence=h, distance=d)
    
    # Calculate intervals and heart rates for all peaks
    all_intervals = np.diff(peaks) / fs
    heart_rates = 60 / all_intervals
    
    # Filter peaks based on heart rate criteria
    if (minBpm != -1 and maxBpm != -1):
        valid_indices = np.where((heart_rates >= minBpm) & (heart_rates <= maxBpm))[0]
        peaks = peaks[:-1][valid_indices]  # Exclude the last peak as it doesn't have a corresponding interval
        heart_rates = heart_rates[valid_indices]
    
    avg = np.mean(heart_rates)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot vertical lines
    x_positions = np.arange(len(heart_rates))
    plt.vlines(x_positions, ymin=0, ymax=heart_rates, colors='b', linewidth=1)
    
    # Customize the plot
    plt.xlabel('Beat Number')
    plt.ylabel(f'Heart Rate (BPM)')
    plt.title(f'Heart Rate avg = {avg:.2f} - Fetal')
    plt.grid(True, linestyle='--', alpha=0.9)
    
    num_ticks = 30  # Adjust this number to control how many ticks you want
    tick_locations = np.linspace(0, len(heart_rates) - 1, num_ticks, dtype=int)
    plt.xticks(tick_locations, tick_locations)
    plt.yticks(np.linspace(60, 150, 20, dtype=int), np.linspace(60, 150, 20, dtype=int))
    
    # Set y-axis to start from 100
    plt.ylim(bottom=60)
    
    # Add markers at the top of each line
    plt.plot(x_positions, heart_rates, 'ro', markersize=2)
    
    # Plot all data points (they're all between minBpm and maxBpm now)
    plt.plot(x_positions, heart_rates, 'go', markersize=4, label='Between Min and Max BPM')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return heart_rates, peaks

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
        ax.plot(x, container[j, :], color='k', alpha=0.05, lw=1)
    
    if add_lines is not None:
        ax.plot(x, add_lines[0], color='r', label='AVGs')
        ax.plot(x, add_lines[1]/np.std(add_lines[1])*np.std(add_lines[0]), color='b', label='Bispectral Filters')
        if len(add_lines) == 3:
            ax.plot(x, add_lines[2], color='g', label='Theory')
        ax.legend(loc='upper right')

    fig.tight_layout()

    if savename is not None:
        plt.savefig(savename)

    plt.show()

    return fig, ax  # Optional, in case you want to further manipulate the figure outside the function