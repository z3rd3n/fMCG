### Averaging in Temporal Domain

**NOTE**: This averaging method may not work as expected for only maternal signal by averaging, since the maternal signal is present in the data indirectly and this might be causing a delay depending on the sensor position.

0. **Reading TDMS Data:**
    - `functions.array_from_TDMSgroup("patients/P050/P050_S01_D2024-06-13_G39.tdms", "R001")`
    - It will return a matrix with sizes (number_of_time_samples x number_of_channels).

1. **Determine the Heart Rate:**
    - This will allow us to take the QRS-peaks from where its periodicity does not change much. For example, generally take the QRS-peaks in the range (mean heart rate - 2 bpm, mean heart rate + 2 bpm).

2. **Find the Fetal Component:**
    - Utilize spatial filtering and ICA (Independent Component Analysis).

3. **Apply a Bandpass Filter (BPF):**
    - ICA is not able to extract the fetal/maternal component unless we apply a BPF to remove high frequency noise and low-frequency DC power. Use a BPF with critical frequencies 0.5 Hz, 80 Hz. Removing the 50 Hz power-line interference makes the process faster.
    - `functions.bandpass_filter`

4. **Apply FastICA with Parallel Algorithm:**
    - For 30 components, mainly due to:
        - Higher number of components produces better components (experimentally observed).
        - Parallel algorithm instead of deflation algorithm (almost 100-fold faster, with similar results).
    - `sources, ica = functions.FASTICA(bp_data, n_comp=30, algo='parallel')`
    - Return ica, because it is used in plotting.

5. **Detect the Fetal Component:**
    - 5.0: **Downside**: It will only detect one fetal component. In the maternal cancellation method, we need to eliminate all fetal components from the mixing matrix. If there are more than one fetal component, they need to be eliminated manually. If dealing with maternal, this part can be skipped.
    - 5.1: **Kurtosis Analysis:** Reduces the useful set of components. With a kurtosis threshold of 2, components are in some peaky structure. Different thresholds can also be used.
    - 5.2: **Peaks Analysis:** The most important part for detection. Use `find_peaks` from scipy. Distance and prominence parameters are crucial for heart rate calculation. Use `distance = 400` and `prominence = 2`. These parameters can be improved for better peak detection.
    - **MSE Analysis:** Use mean square error (MSE) from the average peaks value. If the peaks are aligned in a horizontal line (true for heart signals), MSE should be very small. Use `mse_th=1` to plot components whose MSE value is less than 1. This can be improved depending on the quality of peak detection.
    - **NOTE**: Better peak detection leads to better averaging. It is the heart of averaging at the moment.

6. **Maternal Component Cancellation:**
    - Zero out the columns that correspond to the fetal components. This prevents seeing any fetal contribution in the measurement space. Otherwise, subtracting the maternal reconstruction would destroy the fetal parts and disrupt the P and T waves.
    - `ica.mixing_[:, compF] = 0` and `sources[:, compF] = 0`
    - Remove the fetal contribution from the maternal components using downsampling (QRS peaks are less downsampled than the rest of the signal) and upsampling with cubic interpolation. This further eliminates the fetal contribution from the maternal sources.
    - `maternal_corrected = functions.upDownSample(maternal_sources, maternals, downsample_factor_qrs=5, downsample_factor_other=30)`
    - `maternal_meas = ica.inverse_transform(maternal_corrected, True)` projects back to measurement space.
    - `remaining_fetal = bp_data - maternal_meas` removes maternal from filtered data.

7. **Plot Heart Rates:**
    - Use the corresponding ICA component to see the change of fetal heart rate and calculate the average heart rate.

8. **Identify Range Segments:**
    - Select the range meanBPM - 2, meanBPM + 2. Ensure the selected range does not destroy each other while averaging.

9. **Take Averages:**
    - Use the peaks returned by `plotHR` function. Select a window and align the peaks in the middle of the window. If the heart rate does not change much under the averaged windows, it should eliminate the noise and show the final results.
    - **Denoise Argument:** Takes the wavelet transform of the averages. By zeroing the low energy coefficients, it eliminates the noise when going back to the time domain. Useful to see the nodes where channels join each other for diagnosis.
    - **Window Size:** Select the correct window size. Try several to include 3-peaks to better see P-T waves and RR distance. Other peaks may appear smaller than the middle one because they are not aligned and when the heart rate changes, they destructively interfere with each other, making them small.
