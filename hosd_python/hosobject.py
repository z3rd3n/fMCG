from hosd_python.utils import *
from scipy import signal

class hosobject:
    """
    Class implementing higher-order spectral filtering based on Kovach and Howard 2019.
    
    Usage:
        To create an HOS object:
            hos = hosobject(order, N, sampling_rate, lowpass)

        To fit the object to a block of data (offline mode):
            hos.get_block(data, maxiter=50)

        To add a segment of data in computing a running average (online mode):
            hos.get_input(data)

        To initialize an M component decomposition:
            hos = [hosobject() for _ in range(M)]
            for h in hos:
                h.initialize(N, sampling_rate, lowpass)
    """
    def __init__(self, order=None, *args):

        if order is None:
            return  
        
        self.order = 3
        self.freqs = None
        self.PSD = 0
        self.sampling_rate = 1 # Sampling rate of the input data
        self.normalization = 'awplv' # Normalization used in computing bi-(or poly-)coherence
        self.hos_learning_rate = 0.01 # Learning rate for online mode
        self.filter_adaptation_rate = 0.02 # Filter adaptation rate for online mode
        self.burnin = 20
        self.window_number = 0 # Nunmber of window processed
        self.poverlap = 0.5 # Default overlap between adjacent windows
        self.do_update = True
        self.do_bsp_update = True
        self.do_wave_update = True
        self.do_filter_update = True
        self.adjust_lag = True # Automatically apply a circular shift to the filter and waveforms to center the energy in both
        self.lag = 1  # A phasor representing the amount of circularshift added to the filter estimate (1 = no shift, +/-1i = max shift)
        self.thresh = 0
        self.threshtemp = 1
        self.threshold_type = 'hard'
        self.keepfreqs = None
        self.pdonly = True
        self.dat = []
        self.avg_delay = 1  # Average delay is stored as a phasor because averaging is in the circular domain.
        self.outlier_threshold = 5

        # Access protected members
        self._inputbuffer = []
        self._outputbuffer = []
        self._reconbuffer = []
        self._residualbuffer = []
        self._shiftbuffer = []
        self._thresholdbuffer = []
        self._freqindx = {
            "reduce": np.empty([2,2])
        }
        self._bufferPos = 0
        self._sumlr = 0
        self._sumlr2 = 0
        self._radw = []
        self._sampt = []
        self._delay = 0
        self._waveftlag = []

        # Private members
        self.__bufferN = 1024 
        self.__G = []
        self.__wintype = 'sasaki'
        self.__win = sasaki(1024)
        self.__BCpart = 0
        self.__highpassval = 0
        self.__lowpassval = 0.5 # Lowpass on the edges (max freq)
        self.__glowpassval = 0.5 # Global lowpass
        self.__BIASnum = 0
        self.__Bval = 0
        self.__Bpartval = {}
        self.__Dval = 1

        # we also have dependent members that should be seperately taken care of
        # # # buffersize
        # # # waveform
        # # # wavefft
        # # # filterfun 
        # # # filterfft 
        # # # current_learning_rate
        # # # filterftlag
        # # # window 
        # # # bicoh
        # # # H
        # # # EDF
        # # # highpass  
        # # # lowpass  
        # # # glowpass  
        # # # Bfull
        # # # BIAS
        # # # fullmap
        # # # feature
        # # # current_threshold 
        # # # B 
        # # # Bpart 
        # # # D 

        if isinstance(order, hosobject):
            obj = order
            fns = set(obj.__dict__.keys()) - {'BIAS', 'Bfull', 'H', 'bicoh', 'current_threshold', 'sampling_rate', 'freqindx',
                                          'highpass', 'lowpass', 'glowpass', 'buffersize', 'filterftlag', 'fullmap'}
            if len(obj) == 1:
                obj[1:len(self)-1] = obj
            elif len(obj) > len(self):
                self[len(obj)] = hosobject()
            
            self[0].order = obj[0].order #?look, are multiple selfs are viable?
            self.initialize(obj[0].__bufferN, obj[0].sampling_rate, obj[0].lowpass, obj[0].freqs, obj[0]._freqindx, *args)

            for k in fns:
                self[0].__dict__[k] = obj[0].__dict__[k]

            if len(self) > 1:
                for i in range(1, len(self)):
                    self[i] = hosobject(obj[i])
            return
        
        if order is None:
            return
        elif not args:  # Only order is given
            self.order = order
            return
        else:
            self.order = order
        
        self.initialize(*args)

    def initialize(self, N, sampling_rate=None, lowpass=None, freqs=None, freqindex=None, *args):

        if not isinstance(N, (int, float)):
            X = N
            N = X.shape[0]  # Assuming N is array-like, if it is list then it fails
            self.do_update = True

        else:
            X = []

        if N:
            self.buffersize = N

        self.__highpassval = 2 / N

        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        else:
            sampling_rate = self.sampling_rate

        if lowpass is not None:
            self.__lowpassval = lowpass / self.sampling_rate
        else:
            lowpass = self.lowpass

        if freqs is None:
            freqs = fftfreq(self.__bufferN) * self.sampling_rate
        if not isinstance(freqs, list):
            freqs = [freqs]
        if len(freqs) > self.order:
            self.order = len(freqs)
        if self.order > len(freqs):
            freqs = [freqs.copy() for _ in range(self.order - len(freqs) + 1)]
        self.freqs = freqs

        for k, v in zip(args[::2], args[1::2]):
            setattr(self, k, v)

        self.update_frequency_indexing(freqindex)
        self.reset()

        if X:
            self.get_block(X)

    def reset(self):
        """
        Resets the HOS object to its initial state.
        """
        self.window_number = 0
        self._sumlr = 0
        self._sumlr2 = 0
        z = np.zeros(self.__bufferN)  # Initialize as a NumPy array
        self._inputbuffer = z.copy()
        self._outputbuffer = z.copy()
        self.waveform = z.copy()
        self._shiftbuffer = z.copy()
        self._thresholdbuffer = z.copy()
        self.PSD = np.append(z, 0)  # Use np.append for consistency
        self._bufferPos = 0
        if isinstance(self.B, np.ndarray):
            self.B[:] = 0
        if isinstance(self.__G, np.ndarray):
            self.__G[:] = 1
        if isinstance(self.D, np.ndarray):
            self.D[:] = 1
        self.avg_delay = 1
        self.lag = 1

    def update_frequency_indexing(self, freqindx = None):
        lowpass = self.__lowpassval * self.sampling_rate
        highpass = self.__highpassval * self.sampling_rate

        order = self.order
        freqs = self.freqs

        if not isinstance(lowpass, list):
            lowpass = [lowpass]
        if not isinstance(highpass, list):
            highpass = [highpass]

        if len(lowpass) < order - 1:
            lowpass.extend([lowpass[-1]] * (order - 1 - len(lowpass)))
        if len(lowpass) < order:
            lowpass.append(self.__glowpassval * self.sampling_rate)

        if len(highpass) < order:
            highpass.extend([highpass[-1]] * (order - len(highpass)))

        keepfreqs = []
        for k in range(len(self.freqs)):
            keepfreqs.append(np.logical_and(np.abs(self.freqs[k]) <= lowpass[k], np.abs(self.freqs[k]) > highpass[k]))
        self.keepfreqs = keepfreqs

        if freqindx is None:
            freqindx = freq2index(freqs, order, lowpass, highpass, keepfreqs, self.pdonly)
        self._freqindx = freqindx

        Z = np.zeros(self._freqindx['Is'].shape[0] + 1)  # Initialize as a NumPy array
        self.B = Z.copy()
        self.Bpart = [Z.copy() for _ in range(order)]
        self.D = Z.copy()
        self.__BIASnum = Z.copy()
        self.__G = np.ones(np.sum(self.keepfreqs[0]))  # Initialize as a NumPy array 
        self.reset()

    def learningfunction(self, learningrate, m=1, burnin=None):
        """
        Calculates the learning rate and adjusted learning rate.
        """
        if burnin is None:
            burnin = self.burnin
        lr = np.exp(-self.window_number / burnin) / (self.window_number + 1) + \
             (1 - np.exp(-self.window_number / burnin)) * learningrate
        lradj = 1 - (1 - lr)**m
        return lradj, lr
    
    @property
    def Bfull(self):
        """
        Gets the full bispectrum with complex conjugate symmetry.
        """
        out = self.B[self._freqindx['remap']]
        out[self._freqindx['PDconj']] = np.conj(out[self._freqindx['PDconj']])
        return out
    
    @property
    def bicoh(self):
        """
        Gets the bicoherence (or polycoherence) with normalization and bias correction.
        """
        BC = self.B / self.D
        bias = np.sqrt(self.__BIASnum / (self.D**2 + np.finfo(float).eps))
        BC = (np.abs(BC) - bias) * BC / (np.abs(BC) + np.finfo(float).eps)
        BC = BC[self._freqindx['remap']]
        BC[self._freqindx['PDconj']] = np.conj(BC[self._freqindx['PDconj']])
        return BC
    
    @property
    def BIAS(self):
        """
        Gets the bias estimate for the bicoherence.
        """
        bias = np.sqrt(self.__BIASnum / (self.D**2 + np.finfo(float).eps))
        return bias[self._freqindx['remap']]
    
    @property
    def fullmap(self):
        """
        Gets the frequency index remapping for the full bispectrum.
        """
        return self._freqindx['remap']
    
    @property
    def lowpass(self):
        """
        Gets the lowpass cutoff frequency in Hz.
        """
        return self.__lowpassval * self.sampling_rate
    
    @lowpass.setter
    def lowpass(self, a):
        """
        Sets the lowpass cutoff frequency and updates frequency indexing.
        """
        self.__lowpassval = a / self.sampling_rate
        self.update_frequency_indexing()
    
    @property
    def glowpass(self):
        return self.__glowpassval * self.sampling_rate
    
    @glowpass.setter
    def glowpass(self, a):
        self.__glowpassval = a / self.sampling_rate
        self.update_frequency_indexing()
    
    @property
    def highpass(self):
        return self.__highpassval * self.sampling_rate
    
    @highpass.setter
    def highpass(self, a):
        self.__highpassval = a / self.sampling_rate
        self.update_frequency_indexing()

    @property
    def buffersize(self):
        return self.__bufferN
    
    @buffersize.setter
    def buffersize(self, N):
        """
        Sets the buffer size and updates related parameters. 
        """
        self.__bufferN = N
        self.__win = windowfunc(self.__wintype, N)
        self._radw = np.fft.ifftshift((np.arange(N) - np.floor(N / 2))) / N * 2 * np.pi
        self._sampt = np.fft.ifftshift((np.arange(N) - np.floor(N / 2)))
        self.reset()
        
    @property
    def window(self):
        return self.__wintype

    @window.setter
    def window(self, win):
        self.__wintype = win
        self.__win = windowfunc(win, self.__bufferN)

    @property
    def filterftlag(self):
        """
        Gets the filter FFT without circular shift adjustment.
        """
        out = np.zeros(self.__bufferN, dtype=complex) #?look why complex
        out[self.keepfreqs[0][0]] = self.__G
        return out
    
    @property
    def filterfft(self):
        """
        Gets the filter FFT with lag adjustment.
        """
        out = self.filterftlag
        dt = np.arctan2(np.imag(self.lag), np.real(self.lag)) / (2 * np.pi) * self.__bufferN
        delt = self._radw * dt
        delt[np.isnan(delt)] = 0  # Handle potential NaN values
        out = np.exp(-1j * delt) * out
        return out
    
    @filterfft.setter
    def filterfft(self, in_data):
        """
        Sets the filter FFT with lag adjustment.
        """
        dt = np.arctan2(np.imag(self.lag), np.real(self.lag)) / (2 * np.pi) * self.__bufferN
        delt = self._radw * dt
        F = np.exp(1j * delt) * in_data
        self.__G = F[self.keepfreqs[0]]

    @property
    def filterfun(self):
        """
        Gets the filter function with lag adjustment.
        """
        F = self.filterfft
        return np.fft.ifftshift(np.real(np.fft.ifft(F)))
    
    @filterfun.setter
    def filterfun(self, in_data):
        """
        Sets the filter function and computes its FFT.
        """
        if len(in_data) > self.__bufferN:
            print("Warning: Filter function size does not match current buffer. Filter will be truncated.")
            in_data = in_data[:self.__bufferN]
        elif len(in_data) < self.__bufferN:
            print("Warning: Filter function size does not match current buffer. Filter will be padded.")
            in_data = np.pad(in_data, (0, self.__bufferN - len(in_data)), 'constant')
        F = np.fft.fft(np.fft.fftshift(in_data))
        self.filterfft = F
    
    @property
    def wavefft(self):
        """
        Gets the waveform FFT with lag and centering adjustments.
        """
        F = self._waveftlag
        mxi = np.argmax(np.fft.ifft(F * self.filterftlag))
        dt = np.arctan2(np.imag(self.lag), np.real(self.lag)) / (2 * np.pi) * self.__bufferN + self._sampt[mxi]
        delt = self._radw * dt
        return np.exp(1j * delt) * F
    
    @wavefft.setter
    def wavefft(self, in_data):
        """
        Sets the waveform FFT with lag adjustment.
        """
        F = in_data
        dt = np.arctan2(np.imag(self.lag), np.real(self.lag)) / (2 * np.pi) * self.__bufferN
        delt = self._radw * dt
        if np.all(delt == 0):  # Check for empty delt
            delt = 0 
        self._waveftlag = np.exp(-1j * delt) * F

    @property
    def waveform(self):
        """
        Gets the waveform from its FFT.
        """
        return np.real(np.fft.ifft(self.wavefft))
    
    @waveform.setter
    def waveform(self, in_data):
        """
        Sets the waveform and computes its FFT. 
        """
        F = np.fft.fft(in_data)
        self.wavefft = F
    
    @property
    def EDF(self):
        return self._sumlr / self._sumlr2
    
    @EDF.setter
    def EDF(self, in_data):
        self._sumlr = 1
        self._sumlr2 = 1 / in_data
        self.window_number = in_data

    @property
    def feature(self):
        """
        Gets the feature with inverse fftshift.
        """
        return np.fft.ifftshift(self.waveform) 
    
    @feature.setter
    def feature(self, in_data):
        """ 
        Sets the feature with fftshift.
        """
        self.waveform = np.fft.fftshift(in_data)

    def apply_filter(self, X, apply_window=True, return_shifted=True):
        """
        Applies the filter to the input data.
        """
        FXshift = []
        sgn = 1
        if X.shape[0] == self.__bufferN:
            if apply_window:
                win = self.__win[:, None]
            else:
                win = np.ones((X.shape[0], 1))
            Xwin = np.fft.fftshift(np.tile(win, (1,X.shape[1])) * X, axes=0) 
            FXwin = np.fft.fft(Xwin, axis=0)
            Xfilt = np.real(np.fft.ifft(FXwin * np.tile(self.filterfft[:,None], (1, X.shape[1])), axis=0))
            mxi = np.argmax(Xfilt**self.order, axis=0)
            if self.order % 2 == 0: #?look no enter
                sgn = np.sign(Xfilt[mxi + np.arange(Xfilt.shape[1]) * Xfilt.shape[0]])
            if return_shifted:
                samptc = self._sampt
                dt = samptc[mxi]
                self._delay = dt
                delt = self._radw[:,None] * dt[None, :]
                FXshift = np.exp(1j * delt) * FXwin
            else:
                FXshift = FXwin
            if isinstance(sgn, int):
                FXshift = FXshift * sgn
            else:
                FXshift = FXshift * np.diag(sgn)
        else: 
            Xin = X.flatten().copy()
            Xin = np.pad(Xin, (0, self.__bufferN), 'constant')
            Xfilt = signal.lfilter(self.filterfun, 1, Xin)
            Xfilt = Xfilt[int(np.ceil(self.__bufferN/2)):-int(np.floor(self.__bufferN/2))]
        return Xfilt, FXshift, sgn
    
    @property
    def B(self):
        """
        Gets the bispectrum values.
        """
        return self.__Bval

    @B.setter
    def B(self, in_data):
        """
        Sets the bispectrum values.
        """
        if len(in_data.shape) == 1:
            self.__Bval = in_data
        else:
            self.__Bval = np.concatenate((in_data[self._freqindx['reduce']], [0]))

    @property
    def D(self):
        """
        Gets the normalization values.
        """
        return self.__Dval

    @D.setter
    def D(self, in_data):
        """
        Sets the normalization values.
        """
        if len(in_data.shape) == 1:
            self.__Dval = in_data
        else:
            self.__Dval = np.concatenate((in_data[self._freqindx['reduce']], [0]))

    @property
    def Bpart(self):
        """
        Gets the partial bispectrum values.
        """
        return self.__Bpartval

    @Bpart.setter
    def Bpart(self, in_data):
        """
        Sets the partial bispectrum values.
        """
        if in_data is None or len(in_data[0].shape) <= 1:
            self.__Bpartval = in_data
        else:
            for kk in range(len(in_data)):
                self.__Bpartval[kk] = np.concatenate((in_data[kk][self._freqindx['reduce']], [0]))

    @property
    def H(self):
        """
        Gets the complex conjugate of the normalized bispectrum with bias correction.
        """
        BC = self.B / (self.D + np.finfo(float).eps)
        bias = np.sqrt(self.__BIASnum / (self.D**2 + np.finfo(float).eps))
        bias[np.isnan(bias)] = 0  # Handle potential NaN values
        BC = (np.abs(BC) - bias) * BC / (np.abs(BC) + np.finfo(float).eps)
        H = BC / (self.D + np.finfo(float).eps)
        H = H[self._freqindx['remap']]
        H[self._freqindx['PDconj']] = np.conj(H[self._freqindx['PDconj']])
        return np.conj(H)
    
    def update_bispectrum(self, FX, initialize=False):
        """
        Updates the bispectrum and related quantities.
        """
        m = FX.shape[1]

        if not FX.size:
            return

        # Adjust for lag
        dt = np.arctan2(np.imag(self.lag), np.real(self.lag)) / (2 * np.pi) * self.__bufferN
        delt = self._radw * dt
        delt[np.isnan(delt)] = 0
        FX = np.tile(np.exp(-1j * delt)[:, None], (1, FX.shape[1])) * FX

        FFX = np.conj(FX[self._freqindx['Is'][:, self.order - 1], :])
        FFXpart = [FFX] * (self.order - 1)
        FFXpart.append(np.ones_like(FFX))

        for k in range(self.order - 2, -1, -1):
            FXk = FX[self._freqindx['Is'][:, k], :]
            for kk in range(self.order):
                if kk != k:
                    FFXpart[kk] = FFXpart[kk] * FXk
            FFX = FFX * FXk

        BX = np.mean(FFX, axis=1)
        XPSD = np.mean(np.abs(FX)**2, axis=1)

        BX = np.append(BX, 0)
        XPSD = np.append(XPSD, 0)
        BXpart = [np.append(np.mean(FFXp, axis=1), 0) for FFXp in FFXpart]

        if initialize:
            lradj = 1
            fflr = 1
            lrbias = 1
            lr = 1
            self._sumlr = 1
            self._sumlr2 = 1 / FX.shape[1]
        else: #?look, not enter
            lradj, lr = self.learningfunction(self.hos_learning_rate, m)
            fflr, _ = self.learningfunction(self.filter_adaptation_rate, m, 1 / self.filter_adaptation_rate)
            asympedf = 2 / lr - 1  # Asymptotic EDF
            lrbias = 1 / asympedf * (1 - (1 - lr)**(2 * m))
            self._sumlr = self._sumlr * (1 - lradj) + lradj
            self._sumlr2 = self._sumlr2 * (1 - lr)**(2 * m) + lrbias

        self.B = (1 - lradj) * self.B + lradj * BX
        for kk in range(self.order):
            self.Bpart[kk] = (1 - fflr) * self.Bpart[kk] + fflr * BXpart[kk]
        self.PSD = (1 - lradj) * self.PSD + lradj * XPSD

        if self.normalization == 'awplv':
            NX = np.mean(np.abs(FFX), axis=1)
            NX = np.append(NX, 0)
            XbiasNum = np.sum(np.abs(FFX)**2, axis=1) / m**2
            XbiasNum = np.append(XbiasNum, 0)
            self.__BIASnum = self.__BIASnum * (1 - lr)**(2 * m) + lrbias * XbiasNum
            self.D = (1 - lradj) * self.D + lradj * NX + np.finfo(float).eps
        elif self.normalization in ('bicoh', 'bicoherence'): #?look, not enter
            XBCpart = np.mean(np.abs(FFXpart[0])**2, axis=1)
            XBCpart = np.append(XBCpart, 0)
            self.__BCpart = (1 - lradj) * self.__BCpart + lradj * XBCpart
            self.D = np.sqrt(self.__BCpart * self.PSD[self._freqindx['Is'][:, 0]]) + np.finfo(float).eps

    def update_filter(self):
        """
        Updates the filter based on the current bispectrum estimates.
        """
        Bpart = np.zeros_like(self._freqindx['remap'], dtype=complex)
        for k in range(len(self.Bpart)):
            Bpart += self.Bpart[k][self._freqindx['remap']] * (self._freqindx['partialSymmetryRegions'] == k + 1)
        Bpart[self._freqindx['PDconj']] = np.conj(Bpart[self._freqindx['PDconj']])
        GG = Bpart * self.H

        # Time windowing preservation is commented out in MATLAB, so it's omitted here

        G = np.sum(GG, axis=1)

        # Linear phase trend removal is also commented out in MATLAB

        self.__G = G[self.keepfreqs[0][0][(np.abs(self.freqs[0]) <= self.lowpass)[0]]]

        if self.adjust_lag:
            ffun = np.real(np.fft.ifft(self.filterftlag))
            mph = np.sum(np.exp(-1j * 2 * np.pi * self._sampt / self.__bufferN) * np.abs(ffun)**2) / np.sum(np.abs(ffun)**2)
            mph = mph / (np.abs(mph) + np.finfo(float).eps)
            self.lag = mph  # Circular shift to keep filter energy centered

    def update_waveform(self, FXsh, initialize=False):
        """
        Updates the waveform estimate.
        """
        m = FXsh.shape[1]
        if initialize:
            lradj = 1
        else:
            lradj, _ = self.learningfunction(self.filter_adaptation_rate, m, 1 / self.filter_adaptation_rate)
        self.wavefft = self.wavefft * (1 - lradj) + np.mean(FXsh, axis=1) * lradj

    def write_buffer(self, snip):
        """
        Writes data to the input buffer.
        """
        if snip.size > self.__bufferN:
            self.get_input(snip)
            return
        if self._bufferPos == self.__bufferN:
            self.get_input(self._inputbuffer)
            self._bufferPos = 0
            return
        self._inputbuffer[snip.size : ] = self._inputbuffer[: self.__bufferN - snip.size]
        self._inputbuffer[: snip.size] = snip
        self._bufferPos += snip.size

    def get_input(self, xin, apply_window=True, use_shifted=True, initialize=False):
        """
        Processes input data and updates the HOS object.
        """
        nxin = xin.size
        if nxin >= self.__bufferN:
            self._bufferPos = 0  # Discard the buffer
            if xin.shape[0] != self.__bufferN:
                stepn = round(self.poverlap * self.__bufferN)
                nget = nxin - self.__bufferN + 1
                tindx = np.arange(self.__bufferN)[None, :].T
                wint = np.arange(0, nget, stepn)[None, :]
                T = np.tile(tindx,(1, wint.size)) + np.tile(wint, (tindx.size, 1))
                Xchop = xin[T]
                snip = xin[T[-1, -1] + 1 :].squeeze(1)
                if snip.size:
                    self.write_buffer(snip)
            else:
                Xchop = xin
            self.do_updates(Xchop, apply_window, use_shifted, initialize)
        else:
            self.write_buffer(xin)

    def get_block(self, xin, maxiter=25):
        """
        Fits a block of data all at once and handles multiple components.
        """
        nxin = xin.size
        if nxin >= self.__bufferN:
            self._bufferPos = 0  # Discard the buffer
            if xin.shape[0] != self.__bufferN:
                stepn = round(self.poverlap * self.__bufferN)
                nget = nxin - self.__bufferN + 1
                tindx = np.arange(self.__bufferN)[None, :].T
                wint = np.arange(0, nget, stepn)[None, :]
                T = np.tile(tindx,(1, wint.size)) + np.tile(wint, (tindx.size, 1))
                #Xchop = xin[T].squeeze(2)
                #snip = xin[T[-1, -1] + 1 :].squeeze(1)
                Xchop = xin[T]
                snip = xin[T[-1, -1] + 1 :]
                if snip.size:
                    self.write_buffer(snip)
            else:
                Xchop = xin
            del_val = np.inf
            tol = self.sampling_rate / self.lowpass
            k = 0
            olddt2 = 0
            olddt = 0
            Xwin = Xchop * np.tile(self.__win[:,None], (1, Xchop.shape[1]))
            Xsh = Xwin.copy()
            Xfilt = Xsh.copy()
            # ... (plotting code omitted) ... 
            while del_val > tol and k < maxiter:
                # ... (plotting code omitted) ... 
                k += 1
                if k > 2:
                    olddt = self._delay
                apply_window = False
                use_shifted = False
                initialize = True
                self.get_input(Xsh, apply_window, use_shifted, initialize)
                Xfilt, FXsh, _ = self.apply_filter(Xsh, False, True)
                Xsh = np.real(np.fft.ifftshift(np.fft.ifft(FXsh, axis=0), axes=0))
                newdt = self._delay
                del_val = np.std(olddt2 - newdt) 
                olddt2 = olddt
            # ... (plotting code omitted) ... 
            _, _, _ = self.apply_filter(Xwin, apply_window)
        else:
            self.write_buffer(xin)


    def filter_threshold(self, Xfilt, thresh=None, use_adaptive_threshold=True):
        """
        Applies a moment-based threshold to the filtered data.
        """
        if thresh is None:
            thresh = self.thresh

        Xcent = (Xfilt - np.nanmean(Xfilt, axis=0)) / np.nanstd(Xfilt, axis=0)
        Xmom = Xcent**self.order

        if Xfilt.shape[0] == self.__bufferN and Xfilt.shape[1] == 1 and use_adaptive_threshold:
            trialthresh = self.current_threshold
            Xcs = []
        elif self.order == 3:
            srt = np.sort(Xcent, axis=0)
            if self.outlier_threshold != 0:
                keepsamples = ~np.isnan(iterz(srt, self.outlier_threshold, -1))
            else:
                keepsamples = np.ones_like(srt, dtype=bool)
            m1 = np.nancumsum(srt * keepsamples, axis=0) / np.nancumsum(keepsamples, axis=0)
            m2 = np.nancumsum(srt**2 * keepsamples, axis=0) / np.nancumsum(keepsamples, axis=0)
            m3 = np.nancumsum(srt**3 * keepsamples, axis=0) / np.nancumsum(keepsamples, axis=0)
            c3 = m3 - 3 * m2 * m1 + 2 * m1**3
            keepsrt = (srt > 0) & (c3 > thresh)
            detect = np.any(keepsrt, axis=0)
            trialthresh = np.nansum((np.diff(keepsrt, axis=0) > 0) * srt[1:, :]**self.order, axis=0)
            trialthresh[~detect] = np.inf
            Xcs = []
        else:
            Xsrt = np.sort(Xmom, axis=0)
            srti = np.argsort(Xmom, axis=0)
            Xpow = Xcent[srti]**2
            keepsamples = np.ones_like(Xsrt, dtype=bool)
            Mcs = np.nancumsum(Xsrt * keepsamples, axis=0) / np.nancumsum(keepsamples, axis=0)
            Powcs = np.nancumsum(Xpow * keepsamples, axis=0) / np.nancumsum(keepsamples, axis=0)
            if self.order % 2 == 0:
                Xbaseline = (self.order - 1) * np.nanmean(Xcent**2, axis=0)**(self.order / 2) + thresh
            else:
                Xbaseline = thresh
            Mstd = Mcs / Powcs**(self.order / 2) - Xbaseline
            Xthr = Mstd > thresh
            Xthr = np.cumsum(np.diff(np.concatenate((np.zeros((1, Xthr.shape[1])), Xthr)), axis=0) > 0, axis=0)[::-1] == 0
            threshold_crossing = np.diff(Xthr, axis=0) > 0
            detect = np.any(threshold_crossing, axis=0)
            trialthresh = Xsrt[threshold_crossing]
            trialthresh[~detect] = np.inf

        if self.threshold_type == 'hard':
            THR = Xmom >= trialthresh
        elif self.threshold_type == 'soft':
            sigm = lambda x: 1 / (1 + np.exp(-x))
            THR = sigm(self.threshtemp * (Xmom - trialthresh))
        else:
            raise ValueError("Unrecognized threshold type")

        Xthresh = Xfilt * THR
        return Xthresh, Xcs, trialthresh
    
    def do_updates(self, X, apply_window=True, use_shifted=False, initialize=False):
        """
        Performs updates to the bispectrum, filter, and waveform based on input data.
        """
        if not self.do_update:
            print("Warning: Updating is currently disabled. Set do_update = True to enable.")
            return

        Xfilt, FXsh, _ = self.apply_filter(X, apply_window, use_shifted)
        getwin = self.update_criteria(Xfilt)

        if not getwin.any():
            return

        if self.do_bsp_update:
            self.update_bispectrum(FXsh[:, getwin], initialize)

        if self.do_filter_update:
            self.update_filter()
            Xsrt = np.mean(np.sort(zscore(Xfilt[:, getwin])**self.order, axis=0), axis=1)
            lradj, _ = self.learningfunction(self.filter_adaptation_rate, np.sum(getwin))
            self._thresholdbuffer = self._thresholdbuffer * (1 - lradj) + lradj * np.cumsum(Xsrt, axis=0)

        if self.do_wave_update:
            self.update_waveform(FXsh[:, getwin], initialize)

        self.window_number += np.sum(getwin)
        self._outputbuffer = np.mean(Xfilt, axis=1)
        self._shiftbuffer = np.real(np.fft.ifft(np.mean(FXsh, axis=1)))

    def update_criteria(self, Xfilt):
        """
        Determines which windows to use for updates (placeholder for now).
        """
        return np.ones(Xfilt.shape[1], dtype=bool)

    @property
    def current_threshold(self):
        """
        Gets the current adaptive threshold level.
        """
        xsrt = np.diff(self._thresholdbuffer)
        threshi = np.where(np.diff(self._thresholdbuffer > self.thresh))[0]
        if not threshi.size:
            threshi = len(xsrt)
        return xsrt[threshi]