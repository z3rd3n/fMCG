# fMCG Repository

## Overview

This repository contains various tools and scripts for processing fetal Magnetocardiography (fMCG) data. The data is collected using 16 sensors, each with 2 to 3 channels, resulting in a total of 32 channels. Due to the small amplitude of the signal, the data is quite noisy, requiring several preprocessing steps to extract meaningful information.

## Repository Directories

### [average_notebooks: Averaging in temporal domain notebooks](./average_notebooks/)
- These notebooks first perform ICA and calculate the heartbeats. From the calculated peak locations, they perform averaging on the raw data.

### [hosd_python: Higher Order Spectral (HOS) Filtering python code](./hosd_python/)
- This folder contains the Python codes for HOS, previously used in MATLAB.

### [matlab_analysis: HOS in MATLAB code (more detailed functions)](./matlab_analysis/)
- Contains detailed MATLAB functions for Higher Order Spectral Decomposition (HOSD).

### [notebooks: Notebooks for reading data, importance of butter filtering, etc.](./notebooks/)
- Various notebooks for data reading and analysis.

### [notebooks_freq: Some STFT analysis (not complete and unnecessary)](./notebooks_freq/)
- Contains incomplete and unnecessary STFT analysis.

## System Setup

The fMCG setup includes:
- 16 sensors, most with 2 channels (y and z-axis) and some with 3 channels.
- Total of 32 channels.
- Sensors are placed inside a magnetic shielded box, with the abdomen directly above them for measurements.
- Data is stored in .tdms files, read using the nptdms library.

## Data Collection

- 45 patients measured and recorded.
- Several sessions and group measurements (empty, prone, and on-back positions) lasting 5 to 10 minutes each.
- Sampling frequency of 1000 Hz, resulting in 300,000 samples for 5 minutes of recording.

## Data Processing

### Basic Filtering Before ICA
- Butterworth filters for low-pass (70-99 Hz) and band-stop to reduce noise.
- These filters help in eliminating high-level noise and 50 Hz power line interference.

### Independent Component Analysis (ICA)
- ICA identifies statistically independent components using higher-order statistics.
- Computed ICA results are saved as .npy data to save time.

### Critical Filtering After ICA
- Higher-order spectral analysis for fetal component recovery.
- Uses bispectra to restore phase information and address Gaussianity issues.

## Usage

### Averaging in Temporal Domain
- Read TDMS data, determine heart rate, find fetal component, apply bandpass filter, and perform FastICA with a parallel algorithm.
- Detect fetal component using kurtosis analysis, peaks analysis, and MSE analysis.
- Perform maternal component cancellation to remove fetal contributions from the measurement space.
- Plot heart rates and identify range segments for averaging.

### Higher Order Spectral Decomposition (HOSD)
- Implementation based on Kovach and Howard 2019.
- Software for evaluating higher-order spectra for blind multiple-input deconvolution and pattern identification.

## References

- [Higher Order Spectral Decomposition - Kovach and Howard 2019](https://doi.org/10.1016/j.sigpro.2019.07.007)

## License

This software is provided for evaluation only. No rights are granted to copy, modify, publish, use, compile, sell, or distribute this software either in source code form or executable form, for any purpose, commercial or non-commercial.

For a license, please contact The University of Iowa Research Foundation at uirf@uiowa.edu.

---

Feel free to add or modify any section according to your needs. You can access the current `README.md` [here](https://github.com/z3rd3n/fMCG/blob/bd201f6b98e0b24aaa1f0bd6df3246b34cad0c11/README.md).
