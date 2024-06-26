## repository directories
### [average_notebooks: Averaging in temporal domain notebooks](./average_notebooks/)
- These notebooks first do ICA and calculate the heart beats, then from the calculated peak locations they do averaging on the raw data
### [hosd_python: Higher Order Spectral (HOS) Filtering python code](./hosd_python/)
- This folder contains the python codes for HOS, priorly used in MATLAB
### [matlab_analysis: HOS in MATLAB code (more detailed functions)](./matlab_analysis/)
### [notebooks: Notebooks for readind data, importance of butter filtering etc.](./notebooks/)
### [notebooks_freq: some STFT analysis (not complete and unnecessary)](./notebooks_freq/)

### System Setup
In the given fMCG setup, we have 16 sensors. Most sensors have 2 channels measuring in different
directions (y and z-axis) excluding one or two with 3 channels. In total, 32 channels are present. Since
the signal amplitude is too small, the data is quite noisy.

These sensors inside the magnetic shielded box are placed such that the abdomen will directly lie above them. After the abdomen is placed upon the sensors, the nulling operation on the magnetic field of the sensors are applied (how Optically Pumped Magnetometer works) and the recording of the magnetic fields starts.

### Raw data stored in TDMS
These raw data is stored on the .tdms file, with sessions (Empty, R001, R002, etc.), here Empty means that there is no patient inside the box. Then these tdms files are read using nptdms library in Python and stored as numpy array in the form of 2D matrix (n_samples x n_channels)

### Number of patients and data
So far, 45 patients were measured and recorded with several sessions and group measurements (empty, prone, and on-back positions) with each session lasting 5 to 10 minutes. When ICA is applied, the maternal cardiac interval is clearly visible; however, on some measurements, fetal signals are not easy to see. This could stem from several issues (baby movements, not properly aligned abdomen, etc.) in addition to the fetal signals being much smaller than maternal signals. 

A sampling frequency of 1000 Hz is used, so for 5 minutes of recording, we have 300,000 samples. Knowing cardiac intervals are periodic with maternal heart frequency 1-2 Hz, and fetal with double frequency, we should be able to see the components at least every 1000 samples. This information useful later in the higher-spectral filtering part. 

### Basic filtering before ICA
In the code, Butterworth filters for low-pass with 70-99 Hz to eliminate high-level noise, and band-stop filter to attenuate the effects of 50 Hz power lines interference are used. These choices for critical frequencies of low-pass were already present in the code and somewhat experimental.

### ICA
ICA identifies statistically independent components primarily by leveraging higher-order statistics, such as kurtosis and negentropy, which are measures of non-Gaussianity. While there are various algorithms for implementing ICA, each aims to achieve similar objectives; however, their performance can vary depending on the specific application and data characteristics. FastICA is particularly noted for its efficiency and robustness, and it is readily accessible through the scikit-learn library. In this study, FastICA has been employed to extract both fetal and maternal cardiac signals. Experimentally, number of ICA channels are selected as 18 since it has been observed to perform well.

It takes a lot of time to compute ICA, since we have a lot of samples and channel, therefore computed ICA results are saved as .npy data to save time.
(n_samples x n_channels) -----> |FastICA| ----> (n_samples x n_independent_components)

### Critical filtering after ICA for fetal component: higher-order spectral analysis
https://github.com/ckovach/HOSD

C. K. Kovach and M. A. Howard, Decomposition of higher-order spectra for blind multiple-input deconvolution, pattern identification and separation, Signal Processing, 165 (2019), pp. 357 – 379.

See https://doi.org/10.1016/j.sigpro.2019.07.007

This is a special implementation of a filter recovery using the higher order spectral properties like bispectra, since in the higher order spectra the phase information is restored and the gaussian noise vanishes, they use these properties to construct a filter that performs the same as matched filter. I find the mathmetical analyis a bit complex but since the MATLAB code was already available, I used it to convert MATLAB code to Python.