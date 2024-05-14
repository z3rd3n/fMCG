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
import matlab
import matlab.engine
from sklearn.decomposition import FastICA as skFastICA
from scipy.signal import butter,periodogram,sosfilt,iirnotch,filtfilt,welch,find_peaks, correlate,savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings
import neurokit2 as nk
import matplotlib.pyplot as plt
#import statsmodels.api as sm
#from skimage.restoration import denoise_wavelet
#import pywt
#import shutil
#from fpdf import FPDF
#from scipy.stats import linregress




def log2dict_v1(log_path):
    #This is for all older log files until 21 March 2023
    f=open(log_path)
    a=f.readlines()
    f.close()
    
    coll_dict={}
    
    #Get header information
    no_of_base_infos=7
    
    ############
    base_dict={}
    for i in range(no_of_base_infos):
        parts=a[i].strip().split('\t')
        base_dict[parts[0].strip().strip(':')]=parts[1].strip()
        
        
    base_dict['scaling']=float(base_dict['scaling'])
    
    no_of_channels=int(base_dict['number_of_channels'])
    
    ### Make information from the file name available in the base dict
    name_string_split=base_dict['file_ID'].split('_')

    #print(name_string_split)
    base_dict['patient_ID']=int(name_string_split[0].split('P')[1])
    base_dict['session_ID']=int(name_string_split[1].split('S')[1])
    base_dict['GA']=int(name_string_split[3].split('G')[1])
    base_dict['session_date_str']=str(name_string_split[2].split('D')[1])
    date_string_cutup=name_string_split[2].split('D')[1].split('-')
    base_dict['session_datetime_obj']=datetime(int(date_string_cutup[0]),
                                               int(date_string_cutup[1]),
                                               int(date_string_cutup[2]),
                                               )
    
    
    coll_dict['base_infos']=base_dict
    
    for j in range(5):
        dict_name=a[no_of_base_infos+j*(no_of_channels+2)].strip()
        buf={}
        for i in range(no_of_channels):
            parts=a[no_of_base_infos+j*(no_of_channels+2)+i+2].strip().split('\t')
            buf[int(parts[0])+1]=parts[1]
        coll_dict[dict_name]=buf
        
    offset=no_of_base_infos+j*(no_of_channels+2)+i+3
    
    
    group_dict={}
    cat_list=[]
    for elem in a[offset+1].split('\t'):
        cat_list.append(elem.strip())
    
    i=0
    running=True
    while running:
        if offset+2+i >= len(a):
            running=False
        elif len(a[offset+2+i].strip().split('\t'))==len(cat_list):
            group={}
            for j,elem in enumerate(cat_list):
                group[elem]=a[offset+2+i].strip().split('\t')[j]
            group['fs']=float(group['fs'])
            group['no_of_samples']=int(group['no_of_samples'])
            group['cuts']=list(map(float, group['cuts_string'].split(',')))
            if group['omit_string']=='-1':
                group['omits']=[]
            else:
                group['omits']=list(map(int,group['omit_string'].split(',')))
            group_dict[group['group_name']]=group
            i+=1
        else:
            running=False
        
    coll_dict['group_info']=group_dict
    
    return coll_dict


def log2dict_v2(log_path):
    #Still missing!
    print('log2dict_v2 still missing.')
    return None

def metadatadictionaryfromfilelist():
    # Walks through log_path and collects all logs and created dictonary 
    #Collect all txt files
    log_list=[]
    for file in os.listdir(_base_path+log_path):
        if file.endswith(".txt"):
            log_list.append((os.path.join(_base_path+log_path, file)))

    f_list={}

    #### convert to dict of dicts:
    for entry in log_list:
        match=re.match('.*?_D(\d{4})-(\d{2})-(\d{2})_G\d{2}(_\d)?_log.txt$', entry)
        if match:
            session_date=datetime(int(match[1]),int(match[2]),int(match[3]))
        else:
            pass
            print('Entry could not be parsed in metadatadictionaryfromfilelist: '+entry)
        if session_date<datetime(2023,3,21):
            a=log2dict_v1(entry)
        else:
            a=log2dict_v2(entry)
            
        if a!=None:
            #The if clause of this block can removed once v2 is done
            file_ID=a['base_infos']['file_ID']
            P=a['base_infos']['patient_ID']
            tut_path=_base_path + data_path + 'P' + str(P).zfill(3) + '/' + file_ID + '.tdms'
            if not os.path.exists(tut_path):
                print('No file found for file_ID:'+file_ID)
            a['base_infos']['data_file_path'] = tut_path
            a['base_infos']['base_path_str'] = _base_path
            a['base_infos']['log_path_str'] = _base_path+log_path
            a['base_infos']['data_path_str'] = _base_path+data_path
            a['base_infos']['analysis_path_str'] = _base_path+analysis_path
    
            f_list[file_ID]=a
    return f_list

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

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def ICA_HOS_scoring(S_,file_ID,group_name,cuts):
    ICA_len = np.shape(S_)[1]
    # ICA_len=2
    fig, ax = plt.subplots(ICA_len, 2, sharex=False, figsize=(16 / 2.54, 5 * ICA_len / 2.54), dpi=150)
    qualifiers = []
    for k in range(ICA_len):
        window_size = int(600)
        beat, detect = matlab_hos(S_[:,k], window_size)
        RR, sigma_RR, peaks, popt_RRs, yy, xx = detect_beats(detect)
        try:
            savgol_median = savgol_filter(np.diff(peaks), 41, 3)
            avg_peaks_list = peaks[np.where(np.square((np.diff(peaks) - savgol_median) / sigma_RR) < 2)]
        except:
            avg_peaks_list = peaks
            #print('except on savgol')
        peaks=avg_peaks_list
        if len(peaks)<3:
            used_avgs=0
            coverage=0
            ma=0
        else:
            slice_box, avg_mean,worked = avg_based_QRScomplex(S_[:,k],peaks, sigma_RR, box_size=window_size)
            ma, argma = correlate_with_reference(beat, RR)
            used_avgs=np.shape(slice_box)[1]
            if np.sum(yy)>0:
                coverage =np.sum(yy[np.where(np.abs(xx-RR)<sigma_RR)])/np.sum(yy)
            else:
                coverage=0
            ax[k, 0].plot(avg_mean, label='# avg: %i' % (used_avgs), color='b', linestyle='--')
            ax[k, 0].plot(beat, label='HOS: %.2f' % (ma), color='r')
            ax[k, 0].legend(loc=1)
        ax[k, 1].plot(xx, yy)
        ax[k, 1].plot(xx, gaus(xx, popt_RRs[0], popt_RRs[1], popt_RRs[2]),
                      label='BPM: %i $\pm$ %i \n coverage: %.2f' % (6e4 / RR, 6e4 * sigma_RR / RR ** 2, coverage))
        ax[k, 1].legend(loc=1)
        for j in range(2):
            ax[k, j].grid(alpha=0.7)
            ax[k, j].minorticks_on()
            ax[k, j].grid(True, which='minor', linestyle='dotted', alpha=0.7)
        ax[k, 0].set_xlabel("Time [ms]")
        ax[k, 1].set_xlabel("RR time [ms]")
        secaxb = ax[0, 1].secondary_xaxis('top', functions=(BPM2RR, RR2BPM))
        secaxb.set_xlabel('BPM')
        qualifiers.append([k, RR, sigma_RR, used_avgs, ma, coverage])

    fname= file_ID + '--' + group_name + '--' + str(int(cuts[0])) + '-' + str(int(cuts[1]))
    fname+='--'+str(int(ICA_len)).zfill(2)+'_ICA_score.png'
    fig.suptitle(fname,y=0.95)
    fig.subplots_adjust(hspace=0.001)

    fname = _base_path+analysis_path + file_ID + '/'+fname
    fig.savefig(fname,format='png')
    #print('save fig ausgeführt')
    plt.close()
    #print(qualifiers)
    return fname,qualifiers
def RR2BPM(RR):
    return 6e4/(RR+1e-10)

def BPM2RR(BPM):
    return 6e4/(BPM+1e-10)
def autocorr(buf):
    # FIlter the signal so that auto-corr is cleaner
    bandpass=butter(4,(5,70),btype='bandpass',output='sos',fs=1000)
    buf=sosfilt(bandpass, buf)
    # Wavelet convolution for HF detection
    #[phi, psi, x] = pywt.Wavelet('haar').wavefun(level=2)
    #buf = np.convolve(buf, psi, 'same')

    # Normalized data
    buf -=np.mean(buf)

    acorr = correlate(buf, buf, 'same')
    acorr = acorr /(np.var(buf) *len(buf))
    return acorr[int(len(buf)/2):]


def matlab_hos(data_input_to_matlab, window_size):
    input_1 = matlab.double(list(data_input_to_matlab))
    input_2 = matlab.double([int(window_size)])

    # Directory with matlab script (MacOS style)
    # script_dir = '/Users/<your_user_name>/Documents/python_projects'
    # script_dir = 'C:/Users/Qspin/Desktop/Manual_analysis_2022/analysis_portable'
    script_dir = r'/home/serden/OneDrive/TUM/WS23/thesis/internship/fMCG/matlab_analysis/HOSD-master'
    # script_dir = r'C:\Users\Herzzentrum\Desktop\Manual_analysis_2023\analysis_portable'
    # Start matlab engine
    eng = matlab.engine.start_matlab()
    # Update path to script in matlab with path to directory with your script
    eng.addpath(script_dir)

    # Call matlab script with 2 input and two output arguments
    output_1, output_2 = eng.my_matlab_script(input_1, input_2, nargout=2)

    # Print results
    beat = np.array([item for sublist in output_1 for item in sublist])
    beat = np.roll(beat, int(len(beat) / 2))
    detect = np.array([item for sublist in output_2 for item in sublist])
    return beat, detect


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

def correlate_with_reference(beat_in, RR,use_fMCG_reference=True):
    ##### Inputs ######
    # beat_in ..... array of list of QRS complex waveform from HOS
    # RR... RR time in ms
    if RR > 800 or RR < 300 or np.isnan(RR):
        return 0, 0
    if use_fMCG_reference:
        protobeat = np.array([-1.20870072e-01, -1.62894563e-01, -2.06511559e-01, -2.05867736e-01,
                              -2.05931273e-01, -2.07056073e-01, -2.09378708e-01, -2.12715677e-01,
                              -2.16496549e-01, -2.19999429e-01, -2.22644629e-01, -2.24006347e-01,
                              -2.23766622e-01, -2.21872470e-01, -2.18616830e-01, -2.14473228e-01,
                              -2.09977030e-01, -2.05741256e-01, -2.02338385e-01, -2.00033039e-01,
                              -1.98695092e-01, -1.97981045e-01, -1.97477591e-01, -1.96671144e-01,
                              -1.95002868e-01, -1.92118281e-01, -1.87989399e-01, -1.82771775e-01,
                              -1.76744785e-01, -1.70414924e-01, -1.64392462e-01, -1.59070784e-01,
                              -1.54568154e-01, -1.50873030e-01, -1.47787703e-01, -1.44825221e-01,
                              -1.41430272e-01, -1.37265933e-01, -1.32171273e-01, -1.26052852e-01,
                              -1.19032434e-01, -1.11566665e-01, -1.04279399e-01, -9.77383484e-02,
                              -9.23899693e-02, -8.85278748e-02, -8.61427836e-02, -8.48222193e-02,
                              -8.39477600e-02, -8.29861926e-02, -8.15241176e-02, -7.92260769e-02,
                              -7.60000680e-02, -7.20957523e-02, -6.79031588e-02, -6.38047771e-02,
                              -6.02659225e-02, -5.77858108e-02, -5.66022784e-02, -5.65784237e-02,
                              -5.73740167e-02, -5.85273185e-02, -5.94043932e-02, -5.93470105e-02,
                              -5.79841626e-02, -5.53234815e-02, -5.15927750e-02, -4.71452331e-02,
                              -4.24762815e-02, -3.81054545e-02, -3.43367075e-02, -3.12271106e-02,
                              -2.87404208e-02, -2.67127063e-02, -2.47195020e-02, -2.22321413e-02,
                              -1.89052741e-02, -1.45873475e-02, -9.20609584e-03, -2.88336306e-03,
                              3.93275741e-03, 1.06448120e-02, 1.67360082e-02, 2.17637406e-02,
                              2.53669851e-02, 2.74921679e-02, 2.85259326e-02, 2.90879202e-02,
                              2.97520831e-02, 3.09743430e-02, 3.30528291e-02, 3.59548426e-02,
                              3.92967168e-02, 4.26194893e-02, 4.55507539e-02, 4.76842605e-02,
                              4.85975657e-02, 4.81242090e-02, 4.64371374e-02, 4.38546055e-02,
                              4.07609723e-02, 3.76576031e-02, 3.50268293e-02, 3.30900749e-02,
                              3.17831806e-02, 3.08751341e-02, 2.99619313e-02, 2.84584935e-02,
                              2.58160924e-02, 2.17818105e-02, 1.64094723e-02, 9.93384251e-03,
                              2.79680182e-03, -4.31574514e-03, -1.07262517e-02, -1.60526077e-02,
                              -2.01626430e-02, -2.30572099e-02, -2.49989768e-02, -2.65666832e-02,
                              -2.83556725e-02, -3.07270690e-02, -3.39198951e-02, -3.81593670e-02,
                              -4.34860146e-02, -4.96444383e-02, -5.62471714e-02, -6.29045075e-02,
                              -6.91965250e-02, -7.47694274e-02, -7.95814493e-02, -8.39479710e-02,
                              -8.82956689e-02, -9.29915948e-02, -9.84185238e-02, -1.04962466e-01,
                              -1.12776887e-01, -1.21712074e-01, -1.31503586e-01, -1.41808075e-01,
                              -1.52113757e-01, -1.61955562e-01, -1.71269895e-01, -1.80401687e-01,
                              -1.89888754e-01, -2.00414407e-01, -2.12799205e-01, -2.27758854e-01,
                              -2.45653629e-01, -2.66499883e-01, -2.90032467e-01, -3.15610669e-01,
                              -3.42219709e-01, -3.68776550e-01, -3.94418826e-01, -4.18481491e-01,
                              -4.40448612e-01, -4.60154392e-01, -4.77892659e-01, -4.94148184e-01,
                              -5.09353969e-01, -5.23955003e-01, -5.38372995e-01, -5.52721217e-01,
                              -5.66724175e-01, -5.79955781e-01, -5.91977509e-01, -6.02286871e-01,
                              -6.10418411e-01, -6.16178501e-01, -6.19642688e-01, -6.20937385e-01,
                              -6.20155340e-01, -6.17407436e-01, -6.12724924e-01, -6.05861888e-01,
                              -5.96321691e-01, -5.83573598e-01, -5.67112683e-01, -5.46392986e-01,
                              -5.20988770e-01, -4.90875788e-01, -4.56435458e-01, -4.18320742e-01,
                              -3.77495634e-01, -3.35210092e-01, -2.92666798e-01, -2.50725094e-01,
                              -2.09916181e-01, -1.70473141e-01, -1.32202593e-01, -9.45278620e-02,
                              -5.68200251e-02, -1.86321204e-02, 2.03140162e-02, 6.00730396e-02,
                              1.00266884e-01, 1.40060323e-01, 1.78453812e-01, 2.14524370e-01,
                              2.47381600e-01, 2.76172248e-01, 3.00302095e-01, 3.19476586e-01,
                              3.33459459e-01, 3.42024457e-01, 3.45168162e-01, 3.43086057e-01,
                              3.35911030e-01, 3.23732067e-01, 3.06857411e-01, 2.85866549e-01,
                              2.61529065e-01, 2.34933425e-01, 2.07632311e-01, 1.81448113e-01,
                              1.58117777e-01, 1.39115606e-01, 1.25572460e-01, 1.18065244e-01,
                              1.16422587e-01, 1.19811185e-01, 1.26911226e-01, 1.35920398e-01,
                              1.44669848e-01, 1.51095946e-01, 1.53657218e-01, 1.51376166e-01,
                              1.43852820e-01, 1.31433577e-01, 1.15181832e-01, 9.65912974e-02,
                              7.74091453e-02, 5.96027319e-02, 4.51365211e-02, 3.55764649e-02,
                              3.19107437e-02, 3.46312190e-02, 4.37432935e-02, 5.86483920e-02,
                              7.81883112e-02, 1.00873798e-01, 1.24962225e-01, 1.48360061e-01,
                              1.68726297e-01, 1.83805532e-01, 1.91602639e-01, 1.90424306e-01,
                              1.79182066e-01, 1.57802305e-01, 1.27279236e-01, 8.95201780e-02,
                              4.73528683e-02, 4.49800723e-03, -3.48015034e-02, -6.63018912e-02,
                              -8.61478266e-02, -9.12142848e-02, -7.95324798e-02, -5.06334267e-02,
                              -5.63661831e-03, 5.27536518e-02, 1.20176844e-01, 1.90712026e-01,
                              2.57398566e-01, 3.12898057e-01, 3.49934273e-01, 3.61719198e-01,
                              3.42566612e-01, 2.88322402e-01, 1.96423388e-01, 6.59819868e-02,
                              -1.01993257e-01, -3.04630895e-01, -5.37465243e-01, -7.94468574e-01,
                              -1.06778041e+00, -1.34747507e+00, -1.62143009e+00, -1.87515753e+00,
                              -2.09185822e+00, -2.25316278e+00, -2.34048885e+00, -2.33650202e+00,
                              -2.22650403e+00, -1.99992384e+00, -1.65168541e+00, -1.18290065e+00,
                              -6.00923271e-01, 8.07780110e-02, 8.42988804e-01, 1.66168136e+00,
                              2.50926968e+00, 3.35577767e+00, 4.17010504e+00, 4.92165194e+00,
                              5.58194554e+00, 6.12591123e+00, 6.53298381e+00, 6.78821964e+00,
                              6.88311948e+00, 6.81592672e+00, 6.59156058e+00, 6.22130205e+00,
                              5.72206718e+00, 5.11524090e+00, 4.42541081e+00, 3.67921765e+00,
                              2.90408625e+00, 2.12677625e+00, 1.37220550e+00, 6.62725308e-01,
                              1.74921250e-02, -5.48140404e-01, -1.02288522e+00, -1.39949477e+00,
                              -1.67478579e+00, -1.84961198e+00, -1.92849193e+00, -1.91906480e+00,
                              -1.83169371e+00, -1.67909483e+00, -1.47569334e+00, -1.23679787e+00,
                              -9.77899996e-01, -7.14075402e-01, -4.59232416e-01, -2.25305750e-01,
                              -2.17790481e-02, 1.44516886e-01, 2.69689662e-01, 3.52963087e-01,
                              3.96328403e-01, 4.03940740e-01, 3.81678393e-01, 3.36606182e-01,
                              2.76128023e-01, 2.07247976e-01, 1.36196700e-01, 6.81460151e-02,
                              6.91817200e-03, -4.50433705e-02, -8.64421853e-02, -1.16949828e-01,
                              -1.37128843e-01, -1.48220329e-01, -1.51786644e-01, -1.49503621e-01,
                              -1.43173247e-01, -1.34635030e-01, -1.25497413e-01, -1.17035424e-01,
                              -1.10280464e-01, -1.05930803e-01, -1.04142215e-01, -1.04608872e-01,
                              -1.06807315e-01, -1.09989705e-01, -1.13099115e-01, -1.14987040e-01,
                              -1.14739436e-01, -1.11773782e-01, -1.05866965e-01, -9.73222423e-02,
                              -8.70498458e-02, -7.63519028e-02, -6.66279873e-02, -5.92467123e-02,
                              -5.54284855e-02, -5.59642230e-02, -6.10150543e-02, -7.02153317e-02,
                              -8.28306619e-02, -9.77351483e-02, -1.13482575e-01, -1.28679401e-01,
                              -1.42267364e-01, -1.53478150e-01, -1.61829745e-01, -1.67288940e-01,
                              -1.70214728e-01, -1.71061007e-01, -1.70252606e-01, -1.68254255e-01,
                              -1.65514060e-01, -1.62308500e-01, -1.58773805e-01, -1.55099197e-01,
                              -1.51588824e-01, -1.48576415e-01, -1.46448984e-01, -1.45757082e-01,
                              -1.47093249e-01, -1.50775707e-01, -1.56776773e-01, -1.64875510e-01,
                              -1.74599636e-01, -1.85079850e-01, -1.95236499e-01, -2.04100518e-01,
                              -2.10877673e-01, -2.14974229e-01, -2.16247049e-01, -2.15130634e-01,
                              -2.12400776e-01, -2.08920035e-01, -2.05582764e-01, -2.03229568e-01,
                              -2.02430784e-01, -2.03384507e-01, -2.06017476e-01, -2.10045911e-01,
                              -2.14917197e-01, -2.19894953e-01, -2.24358129e-01, -2.27975384e-01,
                              -2.30626277e-01, -2.32368337e-01, -2.33505545e-01, -2.34456733e-01,
                              -2.35473443e-01, -2.36600437e-01, -2.37822746e-01, -2.39021722e-01,
                              -2.39854488e-01, -2.39879410e-01, -2.38763780e-01, -2.36309941e-01,
                              -2.32460414e-01, -2.27435776e-01, -2.21776016e-01, -2.16124957e-01,
                              -2.10986944e-01, -2.06698776e-01, -2.03481103e-01, -2.01333603e-01,
                              -1.99938341e-01, -1.98811366e-01, -1.97479334e-01, -1.95435805e-01,
                              -1.92141926e-01, -1.87301993e-01, -1.81063783e-01, -1.73870313e-01,
                              -1.66281292e-01, -1.58970863e-01, -1.52632702e-01, -1.47717668e-01,
                              -1.44309764e-01, -1.42215571e-01, -1.41041511e-01, -1.40212613e-01,
                              -1.39085060e-01, -1.37159556e-01, -1.34194020e-01, -1.30143460e-01,
                              -1.25140865e-01, -1.19574715e-01, -1.13983380e-01, -1.08753913e-01,
                              -1.04015933e-01, -9.97751769e-02, -9.59201300e-02, -9.21231394e-02,
                              -8.79579177e-02, -8.31224183e-02, -7.74754488e-02, -7.10246185e-02,
                              -6.40569382e-02, -5.71786415e-02, -5.10953624e-02, -4.63824110e-02,
                              -4.34220666e-02, -4.23609870e-02, -4.29953945e-02, -4.47585081e-02,
                              -4.69350517e-02, -4.88989438e-02, -5.01604825e-02, -5.03767869e-02,
                              -4.94850861e-02, -4.77325853e-02, -4.54747168e-02, -4.30433048e-02,
                              -4.08105741e-02, -3.91443850e-02, -3.81718345e-02, -3.76882830e-02,
                              -3.73056410e-02, -3.65596765e-02, -3.49338218e-02, -3.19802672e-02,
                              -2.74806781e-02, -2.14765766e-02, -1.42050188e-02, -6.08301986e-03,
                              2.27790891e-03, 1.01748338e-02, 1.70397910e-02, 2.26063117e-02,
                              2.69202861e-02, 3.03098445e-02, 3.33325986e-02, 3.65780204e-02,
                              4.04377357e-02, 4.50649230e-02, 5.04308500e-02, 5.62984656e-02,
                              6.22300251e-02, 6.77269691e-02, 7.23156599e-02, 7.55379018e-02,
                              7.70486084e-02, 7.68072390e-02, 7.51207660e-02, 7.24826807e-02,
                              6.94045528e-02, 6.63581372e-02, 6.37271961e-02, 6.16811201e-02,
                              6.01073622e-02, 5.87124051e-02, 5.71052921e-02, 5.47878180e-02,
                              5.12886827e-02, 4.64364269e-02, 4.04390548e-02, 3.37094561e-02,
                              2.67193668e-02, 1.99596332e-02, 1.38120481e-02, 8.37645295e-03,
                              3.47983767e-03, -1.20264145e-03, -6.11360701e-03, -1.18081782e-02,
                              -1.88503142e-02, -2.76784186e-02, -3.85299081e-02, -5.13928211e-02,
                              -6.59635070e-02, -8.16830680e-02, -9.79200301e-02, -1.14171685e-01,
                              -1.30087750e-01, -1.45424842e-01, -1.60131395e-01, -1.74415873e-01,
                              -1.88629456e-01, -2.03142845e-01, -2.18327520e-01, -2.34438411e-01,
                              -2.51387760e-01, -2.68703709e-01, -2.85711174e-01, -3.01697511e-01,
                              -3.16019486e-01, -3.28246791e-01, -3.38281658e-01, -3.46337590e-01,
                              -3.52804780e-01, -3.58151304e-01, -3.62911800e-01, -3.67602358e-01,
                              -3.72497862e-01, -3.77520154e-01, -3.82370714e-01, -3.86669249e-01,
                              -3.89979537e-01, -3.91919203e-01, -3.92350901e-01, -3.91404249e-01,
                              -3.89356071e-01, -3.86607958e-01, -3.83715605e-01, -3.75587558e-01,
                              -3.73901578e-01, -3.67852856e-01, -3.66777894e-01, -3.66078784e-01,
                              -3.65336953e-01, -3.64052529e-01, -3.09762424e-01, -2.74156412e-01,
                              -2.72202492e-01, -2.26272326e-01, -1.92704124e-01, -1.53070400e-01,
                              -1.39940949e-01, -1.36901212e-01, -1.00137928e-01, -9.46436828e-02,
                              -8.85481470e-02, -8.22856519e-02, -7.64133709e-02, -7.13658929e-02,
                              -6.73060233e-02, -6.42058459e-02, -6.18602281e-02, -1.09430936e-01])
        if len(protobeat) < len(beat_in):
            syn = np.pad(protobeat,(int((len(beat_in) - int(len(protobeat))) / 2),int((len(beat_in) - int(protobeat)) / 2)), constant_values=0)
        else:
            syn = protobeat[int(len(beat_in) - int(len(protobeat)) / 2):-int(len(beat_in) - int(len(protobeat)) / 2)]

    else:
        # Create ECG like reference
        syn = nk.ecg_simulate(duration=1, length=int(RR),
                              sampling_rate=1000, noise=1e-10, heart_rate=int(6e4 / RR),
                              heart_rate_std=1e-10, method='ecgsyn', random_state=0)
        # Shift it to have the R peak in the center
        syn = np.roll(syn, int(len(syn) / 2))
        # Add zeros on the sides to get of the size of 'beat_in'
        if int(RR) < len(beat_in):
            syn = np.pad(syn, (int((len(beat_in) - int(RR)) / 2), int((len(beat_in) - int(RR)) / 2)), constant_values=0)
        else:
            syn = syn[int(len(beat_in) - int(RR) / 2):-int(len(beat_in) - int(RR) / 2)]
        # Squareing makes it more resemble an fMCG and also removes the sign
        syn = np.square(syn)
    # Normalize to variance unity
    syn = syn / (np.std(syn) * np.sqrt(len(syn)))
    score = np.correlate(beat_in / np.sqrt(np.var(beat_in) * len(beat_in)), syn, mode='same')
    score -= np.mean(score)
    #score *= np.power([np.cos(np.pi * (0.5 + x / len(beat_in))) for x in range(len(beat_in))], 4)
    return np.max(score), np.argmax(score) / len(score)


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

def butter_filter(dd):
    #Filter data, lpfc=70, lpo=3, bsfc=3, bso=3, bstfc=[4,6], bsto=2, Freqencytoremove=50, Qualityfactor=40
    ''' lpfc: the critical frequency for lowpass filter
        lpo: order of the lowpass filter
        bsfc: the critical frequency for lowpass filter
        bso: order of the lowpass filter
        Freqencytoremove: Frequency to remove for notch filter
        '''
    
    fs = 1000
    band_stop = butter(2, [49.5, 50.5], btype='bandstop',output='sos',fs=fs)
    lowpass = butter(3,99,btype='low',output='sos',fs=fs)
    for i in range(1,np.shape(dd)[1]):
        dd[:,i]=sosfilt(band_stop,dd[:,i])
        dd[:,i]=sosfilt(lowpass,dd[:,i])

    # fs=1000
    # # lpfc=70
    # # lpo=3
    # # bsfc=1
    # # bso=3
    # band_stop=butter(bsto,bstfc,btype='bandstop',output='sos',fs=fs)
    # lowpass=butter(lpo,lpfc,btype='low',output='sos',fs=fs)
    # wavy_baseline=butter(bso,bsfc,btype='low',output='sos',fs=fs)
    # b_notch, a_notch = iirnotch(Freqencytoremove, Qualityfactor, fs)
    # for i in range(1,np.shape(dd)[1]):
    #     dd[:,i]=filtfilt(b_notch, a_notch, dd[:,i])
    #     dd[:,i]=sosfilt(lowpass,dd[:,i])
    #     dd[:,i]=sosfilt(band_stop,dd[:,i])
    #     dd[:,i]-=sosfilt(wavy_baseline,dd[:,i])
    #     # dd[:,i]=denoise_wavelet(dd[:,i],method='BayesShrink',
    #     #                         mode='soft',wavelet_levels=3,
    #     #                         wavelet='sym8',rescale_sigma='True')
    # # dd = dd[2000:12000, :]      ##we only pick 12 s of each for now
    return dd

def butter_filter_flexible(dd): #, lpfc=70, lpo=3, bsfc=3, bso=3, bstfc=[4,6], bsto=2, Freqencytoremove=50, Qualityfactor=40)
    #Filter data
    ''' lpfc: the critical frequency for lowpass filter
        lpo: order of the lowpass filter
        bsfc: the critical frequency for lowpass filter
        bso: order of the lowpass filter
        Freqencytoremove: Frequency to remove for notch filter
        '''
    fs=1000
    band_stop = butter(2,[48,52],btype='bandstop',output='sos',fs=fs)
    band_stop2 = butter(2,[148,152],btype='bandstop',output='sos',fs=fs)

    lowpass_filter = butter(3,90,btype='low',output='sos',fs=fs)
    highpass_filter = butter(2,4,btype='high',output='sos',fs=fs)
    wavy_baseline = butter(3,0.5,btype='low',output='sos',fs=fs)
    for i in range(1,np.shape(dd)[1]):
        dd[:,i]-=sosfilt(wavy_baseline, dd[:,i])
        dd[:,i]=sosfilt(band_stop,dd[:,i])
        dd[:,i]=sosfilt(band_stop2,dd[:,i])
        dd[:,i]=sosfilt(lowpass_filter, dd[:,i])
        dd[:,i]=sosfilt(highpass_filter, dd[:,i])
        
    # lpfc=70
    # lpo=3
    # bsfc=1
    # bso=3
    # band_stop=butter(bsto,bstfc,btype='bandstop',output='sos',fs=fs)
    # lowpass=butter(lpo,lpfc,btype='low',output='sos',fs=fs)
    # wavy_baseline=butter(bso,bsfc,btype='low',output='sos',fs=fs)
    # b_notch, a_notch = iirnotch(Freqencytoremove, Qualityfactor, fs)
    # for i in range(1,np.shape(dd)[1]):
    #     dd[:,i]=filtfilt(b_notch, a_notch, dd[:,i])
    #     dd[:,i]=sosfilt(lowpass,dd[:,i])
    #     dd[:,i]=sosfilt(band_stop,dd[:,i])
    #     dd[:,i]-=sosfilt(wavy_baseline,dd[:,i])
        # dd[:,i]=denoise_wavelet(dd[:,i],method='BayesShrink',
        #                         mode='soft',wavelet_levels=3,
        #                         wavelet='sym8',rescale_sigma='True')
    # dd = dd[2000:12000, :]      ##we only pick 12 s of each for now
    return dd

def plot_raw_signals(dd,file_ID,log_dict,group_name):
    fig, ax = plt.subplots(1,1,sharex=True,figsize=(18/2.54,8/2.54),dpi=150)

    for i in range(np.shape(dd)[1]):
        ax.plot(dd[:,0],dd[:,i+1])
    ax.set_ylabel("B [pT]")
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.7)
    ax.minorticks_on()
    ax.grid(True,which='minor',linestyle='dotted',alpha=0.7)
    fig.suptitle(file_ID+", "+group_name+'\n Raw OPM signals')
    fig.tight_layout()
    fig.savefig(log_dict['base_infos']['analysis_path_str']+"RAW_OPM.png",format='png')
    fig.close()
    return log_dict['base_infos']['analysis_path_str']+"RAW_OPM.png"

def FASTICA(dd,n_comp=None,random=0):
    if n_comp==None:
        n_comp=int(np.shape(dd)[1]-1)
    #print('Number of components Fast ICA: '+str(n_comp))
    ica = skFastICA(n_components=n_comp,algorithm='deflation',max_iter=4000,fun='logcosh',tol=1e-5,random_state=random)
    S_ = ica.fit_transform(dd[:,1:])
    return S_

def plot_ICA_section(dd,S_,file_ID,log_dict,group_name,cuts):
    no_ica_chans=np.shape(S_)[1]

    fs=log_dict['group_info'][group_name]['fs']
    span=10
    offset=2

    fig, ax = plt.subplots(no_ica_chans,1,sharex=True,figsize=(18/2.54,24/2.54),dpi=250)
    fig.subplots_adjust(hspace=0)

    for i in range(no_ica_chans):
        ax[i].plot(dd[int(offset*fs):int(fs*(offset+span)),0],S_[int(offset*fs):int(fs*(offset+span)),i],label=str(i))
        ax[i].legend(loc=1)
    fig.suptitle(file_ID+", "+group_name+', ICA outputs',y=0.9)
    fname = log_dict['base_infos']['analysis_path_str'] + file_ID + '/'
    fname += file_ID + '--' + group_name + '--' + str(int(cuts[0])) + '-' + str(int(cuts[1]))
    fname+='--'+str(int(no_ica_chans)).zfill(2)+'_ICA.png'

    fig.savefig(fname,format='png')
    plt.close()
    return fname

def save_ICA_to_npy(S_,file_ID,log_dict,group_name,cuts,n_comp):
    fname=log_dict['base_infos']['analysis_path_str']+file_ID+'/'
    fname+=file_ID+'--'+group_name+'--'+str(int(cuts[0]))+'-'+str(int(cuts[1]))
    fname += '--' + str(int(n_comp)).zfill(2) + '_ICA.npy'
    np.save(fname,S_)
    return fname

def pick_signals_from_ICA():
    return {'fetus':0,'mother':4}

def getBPM(S_,mother, fetus,file_ID,group_name,fname=None,plot=False):
    sampling_rate = 1000
    label_dict = {fetus: 'Fetus', mother: 'Mother'}

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(18 / 2.54, 8 / 2.54), dpi=150)
    for elem in [mother, fetus]:
        raw_data = np.copy(S_[:, elem])
        cleaned = nk.ecg_clean(raw_data, sampling_rate=sampling_rate,method='engzeemod2012')
        rpeaks = nk.ecg_peaks(cleaned, method='nk2', sampling_rate=sampling_rate, correct_artifacts=True)
        rate = nk.signal.signal_rate(rpeaks[1], sampling_rate=sampling_rate, desired_length=len(cleaned))
        ax.plot(np.linspace(0,len(S_[:,elem])-1,num=len(S_[:,elem]))/1000, rate, label=label_dict[elem])

    ax.grid(alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.7)
    ax.set_ylabel("RR rate [BPM]")
    ax.set_xlabel("Time [s]")
    fig.legend()
    fig.suptitle(file_ID+ ", " + group_name)
    if plot:
        if fname==None:
            fname=_base_path+analysis_path+file_ID+'--'+group_name+'BPM_pick.png'
        fig.savefig(fname,format='png')
    plt.close()
    return [np.linspace(0,len(S_[:,elem])-1,num=len(S_[:,elem]))/1000, rate]

def segment(channel,avgs=10,linear_regression_removal=False,plot=True,fname=None):

    avg_no = avgs
    lin_reg_on = linear_regression_removal

    raw_data = np.copy(channel)
    cleaned = nk.ecg_clean(raw_data, sampling_rate=1000)
    segments = nk.ecg_segment(cleaned, sampling_rate=1000, show=False)
    a = int(len(segments) / avg_no)
    b = len(segments['1'])
    container = np.zeros((a, b))

    if plot:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(18 / 2.54, 8 / 2.54), dpi=150)
        ax.grid(alpha=0.7)
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='dotted', alpha=0.7)
        ax.set_ylabel("Signal [A.U.]")
        ax.set_xlabel("Time [s]")

    for i in range(a):
        meaner = np.zeros((avg_no, b))
        for j in range(avg_no):
            s = np.array(segments[str(1 + i + j)]['Signal'])
            x = np.array(segments[str(1 + i + j)].axes[0])
            if lin_reg_on:
                slope, intercept, r, p, se = linregress(x, s)
                meaner[j, :] = s - x * slope - intercept
            else:
                meaner[j, :] = s
        container[i, :] = np.mean(meaner, axis=0)
        if plot:
            ax.plot(x, np.mean(meaner, axis=0), color='b', alpha=0.05)
    if plot:
        fig.savefig(fname,format='png')
        plt.close()
    return container,segments

def make_NPY_list():
    # make helper function
    def walker(walk_this_path):
        file_set = set()
        folder_set = set()
        abs_set = set()
    
    
        for dir_, _, fs in os.walk(walk_this_path):
            for file_name in fs:
                rel_dir = os.path.relpath(dir_, walk_this_path)
                rel_file = os.path.join(rel_dir, file_name)
                abs_file = os.path.join(walk_this_path, rel_file)
                abs_set.add(abs_file)
                file_set.add(rel_file)
                if rel_dir != '.':
                    folder_set.add(rel_dir)
        return folder_set, file_set, abs_set
    
    folders, files, absfiles = walker(_base_path+analysis_path)
    out_put_path=_base_path+analysis_path+'ICA_selection.txt'
    f=open(out_put_path,mode='w')
    f.write('filepath \t selected? \t mother \tfetus \n')

    for file in sorted(list(absfiles)):
        if file.endswith('.npy'):
            f.write(file+'\t no \t -1 \t -1 \n')
    f.close()
    return out_put_path
