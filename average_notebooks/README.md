### Averaging in temporal domain

**NOTE**: This averaging method may not work as expected for only maternal signal by averaging, since the maternal signal is present in the data indirectly and this might be causing a delay depending on the sensor position.

0- Reading TDMS data:
    - 'functions.array_from_TDMSgroup("patients/P050/P050_S01_D2024-06-13_G39.tdms", "R001")'
    - It will return a matrix with sizes (number_of_time_samples x number_of_channels)

1- First of all, we need to determine the heart rate, this will allow us to take the QRS-peaks from where it's periodicity does not change much. For example, I generally take the QRS-peaks in range (mean heart rate - 2 bpm, mean heart rate + 2 bpm)

2- To determine the heart rates, we need to find the fetal component (or maternal component for averaging), to do this we utilize the spatial filtering, ICA (Independent Component Analysis)

3- But, unfortunately ICA is not able to extract the fetal/maternal component unless we apply a BPF to remove high frequency noise and low-frequency DC power. Thus I used a BPF with critical frequencies 0.5 Hz, 80 Hz. I also observed that removing the 50 Hz power-line interference makes the process faster.
    - functions.bandpass_filter

4- After basic filtering, I apply FastICA with parallel algorithm for 30 components mainly due to 2 reasons:
    - Higher number of components produces better components (experimantally observed)
    - Parallel algorithm instead of deflation algorithm (It is almost 100-fold faster, and it produces similar results, I did not observe too much difference)
    - 'sources, ica = functions.FASTICA(bp_data, n_comp=30, algo='parallel')'
    - I am returning ica, because afterwards I use it in plotting

5-  After ICA, we need to detect the fetal component, to do this, I utilize two methods to automatically detect the fetal component 
    - 5.0: **Downside**: it will only detect one fetal component, in the maternal cancellation method we need to eleminate all fetal components from the mixing matrix, therefore if there are more than one fetal component, unfortunately you need to eleminate them manually. But if you don't deal with fetal but maternal, than this part should be skippable.
    - 5.1: Kurtosis analysis: Just by looking at the kurtosis, we reduce the useful set of components. I observed that if we select the kurtosis threshold of 2, then the components are already in some peaky structure. (This might not be optimal, and different thresholds also can be used)
    - 5.2: Peaks analysis: This is the most important part for detection, I use the 'find_peaks' function from scipy. and the distance and prominence parameters are quite of importance, because that's how the heart rate is calculated. I use distance = 400, and prominence = 2, meaning that there will be a distance of at least 400 samples between two QRS complexes, this is for fetal but it also detect maternals with these distance and prominence argument tells us the distinction of the QRS peaks from its neighbors. These parameters can even be more improved for better peak detection.
    - Now that we detect the peaks, we still need to detect the fetal component, so I use MSE (mean square error) from the average peaks value, this tells me that if the peaks are aligned in a horizontal line, which should be true for heart signals, the MSE should be very small. Thus I put an argument as mse_th=1, which plots the components whose mse value is less than 1, this is quite open for improvement, since it will be changing depending on the quality of peak detection:
    - **NOTE**: Better the peak detection, better the averaging. It is the heart of averaging at the moment.
    
6- **THIS PART IS FOR MATERNAL COMPONENT CANCELLATION**: 
    - Now that, we know the components, first we need to zero out the columns that corresponds to the fetal components. Because when we go back to measurement space we don't want to see any fetal contribution. Otherwise when we subtract the maternal reconstruction, we would also destroy the fetal parts, which is not completely eleminated anyway and disrupting the P and T waves.
    - 'ica.mixing_[:, compF] = 0' and 'sources[:, compF] = 0'
    - Than, we remove the fetal contribution from the maternal components using first downsampling (qrs peaks are less downsampled than the rest of the signal) and upsampling using the cubic interpolation. This will further eleminate the fetal contribution from the maternal sources so that when we subtract maternal reconstruction we don't destroy the fetal signal.
    - 'maternal_corrected = functions.upDownSample(maternal_sources, maternals, downsample_factor_qrs=5, downsample_factor_other=30)'
    - 'maternal_meas = ica.inverse_transform(maternal_corrected, True)' will project back to measurement space
    - 'remaining_fetal = bp_data - maternal_meas' : removing maternal from filtered data

7- Now we plot the heart rates using the corresponding ICA component to see the change of fetal heart rate and calculation of the average heart rate

8- then we identify the range_segments, where do we want to take averages, I generally select the range meanBPM - 2, meanBPM + 2, but as long as they don't destory each other while averaging this range could be extended. 

9- Then using the peaks, returned by plotHR function, I take the averages by selecting a window and aligning the peaks in the middle of the window, if the heart rate doesn't change much under the averaged windows, then should eleminate the noise and show the final results.
    - I have a denoise argument, which takes the wavelet transform of the averages, and by zeroing the low energy coefficients, it eleminates the noise when we go back to time domain. I think it is useful to see the nodes where channels are joining each other for diagnosis.
    - window size is heuristic and it is very important to select the correct window size. I recommend trying several to include 3-peaks so that we can see P-T waves better and RR distance, also the reason why the other peaks are smaller than the middle one because they are not aligned and when the heart rate changes, they destructively interfere with each otther and make them small

