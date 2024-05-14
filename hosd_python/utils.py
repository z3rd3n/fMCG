import numpy as np
from ismember import ismember
from scipy.signal.windows import (
    bartlett, barthann, blackman, blackmanharris, bohman,
    chebwin, flattop, gaussian, hamming, hann, kaiser,
    nuttall, parzen, boxcar, taylor, tukey, triang
)
from scipy.stats import zscore

def sasaki(N):
    """
    Minimum bias window for bispectral estimation following Sasaki, Sato, 
    and Yamashita (1975).

    Parameters:
    N (int): Number of points in the window.

    Returns:
    numpy.ndarray: Sasaki window.
    """
    t = np.linspace(-1, 1, N)
    saswin = lambda x: (1/np.pi) * np.abs(np.sin(np.pi * x)) + (1 - np.abs(x)) * np.cos(np.pi * x)
    out = saswin(t)
    tol = 1e-9
    out = np.round(out / tol) * tol
    return out

def fftfreq(N):
    return np.fft.ifftshift((np.arange(N) - np.floor(N / 2))) / N

def adjust_indices(ism, ismi):
    newIdx = np.zeros((ism.shape))
    counter = 0
    for idx in range(ism.shape[0]):
        if ism[idx]:
            newIdx[idx] = ismi[counter]
            counter += 1
    return newIdx
            

def freq2index(freqsin, order=None, lowpass=0.5, highpass=0, keepfreqs=None, condense=False, frequency_spacing='linear'):
    """
    Generates indexing for polyspectra of a given order.

    Args:
        freqsin (array_like or list): Vector or list of vectors containing frequencies.
        order (int, optional): Order of the higher-order spectrum. Defaults to 3 if not provided or None.
        lowpass (float or array_like, optional): Lowpass cutoff frequency(ies). Defaults to 0.5.
        highpass (float or array_like, optional): Highpass cutoff frequency(ies). Defaults to 0.
        keepfreqs (array_like or list, optional): Indices of frequencies to keep. Defaults to all frequencies.
        condense (bool, optional): Whether to condense the output to the principal domain. Defaults to False.
        frequency_spacing (str, optional): Spacing of frequencies ('linear' or 'log'). Defaults to 'linear'.

    Returns:
        dict: A dictionary containing the following keys:
            - 'Is': Indices of frequencies for each dimension.
            - 'freqs': Frequencies used in the calculation.
            - 'keep': Logical array indicating which frequencies were kept.
            - 'principal_domain': Logical array indicating the principal domain.
            - 'remap': Mapping of frequencies to the principal domain.
            - 'reduce': Indices of frequencies in the principal domain.
            - 'PDconj': Logical array indicating the complex conjugate of the principal domain.
            - 'Bfreqs': Frequencies used for the bispectrum calculation.
            - 'partialSymmetryRegions': Symmetry types for each region.
    """

    if order is None:
        if isinstance(freqsin, list):
            order = len(freqsin)
        else:
            order = 3
    if not isinstance(lowpass, (list, np.ndarray)):
        lowpass = [lowpass] * order
    if not isinstance(highpass, (list, np.ndarray)):
        highpass = [highpass] * order

    if isinstance(freqsin, (list, np.ndarray)): #?look -- in matlab isNumeric?
        freqsin = [np.array(f) for f in freqsin]
    else:
        freqsin = [freqsin] * order
        condense = True

    if keepfreqs is None:
        keepfreqs = [np.arange(len(f)) for f in freqsin]
    elif isinstance(keepfreqs, (list, np.ndarray)): #?look -- in matlab isNumeric?
        keepfreqs = [np.array(k) for k in keepfreqs]
    else:
        keepfreqs = [keepfreqs] * order

    if len(freqsin) < order:
        freqsin.extend([freqsin[-1]] * (order - len(freqsin)))

    freqs = [f[k] for f, k in zip(freqsin, keepfreqs)]

    nneg = np.sum(freqs[-1] < 0)
    npos = np.sum(freqs[-1] > 0)
    two_sided = nneg > npos / 2

    PD, Ws, Is, keep = find_principal_domain(freqs, order, lowpass, highpass)
    Fsum = Ws[-1]

    n = len(Fsum)
    frsrti = np.argsort(freqs[-1])
    frsrt = freqs[-1][frsrti]
    if frequency_spacing == 'linear':
        dfr = np.diff(frsrt) / 2
        frcent = np.concatenate(([frsrt[0] - dfr[0]], frsrt[:-1] + dfr, [frsrt[-1] + dfr[-1]]))
    elif frequency_spacing in ('log', 'logarithmic'): #?look not checked but upper if correct
        lfrsrt = np.log(frsrt)
        dlfr = np.diff(lfrsrt) / 2
        lfrcent = np.concatenate(([lfrsrt[0] - dlfr[0]], lfrsrt[:-1] + dlfr, [lfrsrt[-1] + dlfr[-1]]))
        frcent = np.exp(lfrcent)
        frcent[np.isnan(frcent)] = 0
    else:
        raise ValueError("Invalid frequency_spacing. Choose 'linear' or 'log'.")

    to_be_sorted = np.concatenate((((-1) ** two_sided) * Fsum, frcent))
    srti = np.argsort(to_be_sorted)
    srt = to_be_sorted[srti]
    E = np.concatenate((np.zeros(n), np.ones(len(frcent))))
    E = E[srti]
    IND_temp = np.zeros_like(E)
    IND_temp[srti] = np.cumsum(E)
    IND = IND_temp[:n]
    IND[IND == 0] = 1
    IND[IND > len(frsrti)] = len(frsrti)
    IND = frsrti[IND.astype(int).reshape(Fsum.shape[0])]
    Is.append(IND) #?look, Is{order} normally

    freqindex = [np.append(np.where(kf)[1], 0) for kf in keepfreqs]
    Is= [fri[x[:]].T for x, fri in zip(Is, freqindex)]    
    Is = np.column_stack(Is)
    W = np.column_stack([w for w in Ws]).T

    int_types = ['uint8', 'uint16', 'uint32', 'uint64']
    max_ints = {i_type: np.iinfo(i_type).max for i_type in int_types}
    num_rows = Is.shape[0]
    use_int_list = [num_rows <= max_ints[i_type] for i_type in int_types]

    use_int = int_types[use_int_list.index(1)]

    subremap = np.zeros_like(keep, dtype=use_int)
    subremap[keep] = np.flatnonzero(keep)
    subremap = subremap.T

    tol = np.min(np.abs(np.diff(np.concatenate(freqs)))) / 2

    if condense:
        sort_me = np.round(np.abs(W) / tol) * tol
        wsrti = np.argsort(sort_me, axis=0)
        Wsrt = np.take_along_axis(sort_me, wsrti, axis=0)
        signedW = np.take_along_axis(W, wsrti, axis=0)
        Wsrt = Wsrt * np.sign(signedW)

        wsrt_transposed = Wsrt.T
        wsrt_pd = Wsrt[:, PD].T

        ism, ismi = ismember(wsrt_transposed, wsrt_pd, 'rows')
        ismi = adjust_indices(ism, ismi)
        
        ismconj, ismiconj = ismember(-wsrt_transposed, wsrt_pd, 'rows')
        ismiconj = adjust_indices(ismconj, ismiconj)
        # Adjust indices based on your provided MATLAB code
        ismiconj[0] = 0
        ismconj[0] = False
        # ismiconj[ism & ismconj] = 0 #?look notes
        ismconj[ism & ismconj] = False

        IsPD = Is[PD.ravel(), :]  # Assuming 'Is' is some array that can be indexed like this
        PDremap = np.zeros_like(subremap) - 1  # Assuming 'subremap' is defined somewhere
        PDremap[keep] = ismi + ismiconj  # Assuming 'keep' is a boolean or index array for subsetting

        PDconjugate = np.zeros_like(subremap, dtype=bool)
        PDconjugate[keep] = ismconj  # If 'keep' is an index or boolean array for keeping valid entries
        Is = IsPD
        subremap = PDremap

        W23 = W[1:order, :]
        W23srt = np.sort(np.round(np.abs(W23) / tol) * tol, axis=0)
        w23srti = np.argsort(np.round(np.abs(W23) / tol) * tol, axis=0)
        signedW23 = np.take_along_axis(W23, w23srti, axis=0)
        W23srt *= np.sign(signedW23)
        SR = np.zeros(np.sum(keep), dtype=np.uint8)

        for k in range(order, 0, -1):
            W23pd = W[np.setdiff1d(np.arange(order), np.array([k - 1])), :][:, PD]
            W23pdsrt = np.sort(np.round(np.abs(W23pd) / tol) * tol, axis=0)
            w23pdsrti = np.argsort(np.round(np.abs(W23pd) / tol) * tol, axis=0)
            signedW23pd = np.take_along_axis(W23pd, w23pdsrti, axis=0)
            W23pdsrt = (W23pdsrt * np.sign(signedW23pd)).T
            sri, _ = ismember(W23srt.T, np.vstack([W23pdsrt, -W23pdsrt]), 'rows')
            SR[sri] = k
    else:
        PDconjugate = False

    # Translation of `arrayfun` and `cellfun`
    keeplp = [abs(fr) <= lp for fr, lp in zip(freqsin[:-1], lowpass[:-1])]
    keeplp2 = [kpfr[kplp] for kpfr, kplp in zip(keepfreqs[:-1], keeplp)]

    # Convert dims calculation to Python
    dims = np.array([np.sum(x) for x in keeplp])

    # Translate conditional array assignments and keep logics
    keepregion = np.zeros(dims, dtype=bool)
    keepregion[np.ix_(*keeplp2)] = True
    keepall = np.zeros(dims, dtype=bool)
    keepall[keepregion] = keep.flatten('F')
    Z = np.zeros(dims, dtype=use_int)
    PDall = Z.copy()
    PDall[keepall] = PD
    PDall = PDall.T
    remap = Z.copy() - 1
    remap[keepregion] = subremap.flatten('F')

    remap[remap == np.iinfo(use_int).max] = Is.shape[0] 

    Is[:,order - 1] = Is[:,order - 1] - 1
    output = {}
    output['Is'] = Is
    output['freqs'] = freqs[:-1]
    output['keep'] = keepall
    output['principal_domain'] = PDall
    output['remap'] = remap
    output['reduce'] = np.where(PDall.flatten('F'))[0]

    PDconj = np.zeros_like(Z, dtype=bool)
    PDconj[keepregion] = PDconjugate.flatten('F')
    output['PDconj'] = PDconj

    # Translating the cellfun equivalent
    output['Bfreqs'] = [fr[kpfr] for fr, kpfr in zip(freqsin[:-1], keeplp)]

    SymReg = np.zeros_like(keepall, dtype=np.uint8)
    SymReg[keepall] = SR
    output['partialSymmetryRegions'] = SymReg.T

    # 'output' is a dictionary equivalent to a MATLAB struct array.
    # Access elements with output['<field_name>'] e.g., output['Is']
    return output

def find_principal_domain(freqs, order=None, lowpass=np.inf, highpass=0):
    """
    Finds the principal domain in a higher-order spectrum.

    Args:
        freqs (array_like or list): Array of frequencies or list of frequency arrays.
        order (int, optional): Order of the higher-order spectrum. Defaults to 3 if not provided or None.
        lowpass (float or array_like, optional): Lowpass cutoff frequency(ies). Defaults to infinity.
        highpass (float or array_like, optional): Highpass cutoff frequency(ies). Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - PD: Logical array indicating the principal domain.
            - Ws: List of frequency arrays for each dimension.
            - Is: List of index arrays for each dimension.
            - keep: Logical array indicating which frequencies were kept.
    """

    if order is None:
        if isinstance(freqs, list):
            order = len(freqs)
        else:
            order = 3
    if not isinstance(lowpass, (list, np.ndarray)): #?look
        lowpass = [lowpass] * order
    if not isinstance(highpass, (list, np.ndarray)): #?look
        highpass = [highpass] * order

    if not isinstance(freqs, list):
        freqs = [freqs] * order

    if len(freqs) < order:
        freqs.extend([freqs[-1]] * (order - len(freqs)))

    nsig = order // 2
    signatures = (-1) ** (np.tile(np.arange(1, order + 1), (nsig, 1)) - 1 >= 
                         order - np.tile(np.arange(1, nsig + 1)[:, np.newaxis], (1, order))) #?look, running but complex

    freqsi = [np.arange(len(f)) for f in freqs[:-1]]
    Is = [np.array([])] * (order - 1)
    Is[:] = np.meshgrid(*freqsi, indexing='ij')  # 'ij' for row-major order
    Ws = []
    Wsum = 0
    for k in range(len(Is)):
        Ws.append(freqs[k][Is[k]])
        Wsum = Wsum + Ws[k]
    Ws.append(-Wsum)

    keep = np.ones(Ws[0].shape, dtype=bool)
    for k in range(len(Ws)):
        if not np.isinf(lowpass[k]):
            keep = keep & (np.abs(Ws[k]) < lowpass[k])
        if highpass[k] > 0:
            keep = keep & (np.abs(Ws[k]) > highpass[k])

    for k in range(len(Ws)):
        WskT = Ws[k].T
        Ws[k] = WskT[keep]
        if k < len(Is):
            IskT = Is[k].T
            Is[k] = IskT[keep]

    PD = np.zeros(Ws[0].shape, dtype=bool)
    for k in range(nsig):
        PD0 = (Ws[0] >= 0) & (Ws[-1] <= 0)
        for kk in range(1, order):
            if signatures[k, kk] == signatures[k, kk - 1]:
                PD0 = PD0 & (signatures[k, kk] * Ws[kk] >= signatures[k, kk - 1] * Ws[kk - 1])
            else:
                PD0 = PD0 & (signatures[k, kk] * Ws[kk] >= 0)
        PD = PD | PD0

    if np.max(highpass) > 0 and order > 3: #?look, it does not go inside here, but is converstion correct?
        discard = np.zeros(PD.shape, dtype=bool)
        WPD = np.column_stack(Ws)
        WPD = WPD[PD]
        for k in range(order - 1):
            discard[PD] = discard[PD] | np.any(np.abs(np.tile(WPD[:, k:k + 1], (1, order - k)) +
                                                     WPD[:, k + 1:]) <= np.max(highpass), axis=1)

        PD = PD[~discard]
        for k in range(len(Ws)):
            Ws[k] = Ws[k][~discard]
            if k < len(Is):
                Is[k] = Is[k][~discard]
        keep[keep] = keep[keep] & ~discard

    return PD, Ws, Is, keep

# def zscore(array, axis=0, side=0):
#         mean_val = np.nanmean(array, axis=axis)
#         std_val = np.nanstd(array, axis=axis)
#         z_vals = (array - mean_val) / std_val
#         if side == 0:
#             return np.abs(z_vals)
#         elif abs(side) == 1:
#             return side * z_vals
#         else:
#             raise ValueError('Side parameter must be 0 (two-sided), 1 (high threshold), or -1 (low threshold)')


def iterz(x, thresh=10, side=0):
    """
    Iterative z-score thresholding. At each iteration a z-score is computed
    on the columns of x, ignoring NaNs, and all values in x above the given z
    threshold are replaced with NaNs until no values in x are above the threshold. 
    If side = 0, thresholding is based on the magnitude of the z-score. Thresholding 
    is applied to positive values if side = 1 and negative values if side =
    -1.
    """

    # Example usage:
    # x = np.array([[1, 2, 3], [4, 5, np.nan], [np.nan, 7, 8]], dtype=float)
    # result = iterz(x)

    z = zscore(x)

    while np.any(z > thresh):
        x[z > thresh] = np.nan
        z = zscore(x)

    return x

def windowfunc(wname, N, *args):
    """
    Generate an N-point window of a specified type.
    
    Parameters:
    - wname: Name of the window function (string).
    - N: Number of points in the window.
    - args: Additional arguments for certain window types.
    
    Returns:
    - An N-point window of the specified type.
    """

    window_map = {
    'bartlett': bartlett,
    'barthannwin': barthann,
    'blackman': blackman,
    'blackmanharris': blackmanharris,
    'bohmanwin': bohman,
    'chebwin': chebwin,
    'flattopwin': flattop,
    'gausswin': gaussian,
    'hamming': hamming,
    'hann': hann,
    'kaiser': kaiser,
    'nuttallwin': nuttall,
    'parzenwin': parzen,
    'rectwin': boxcar,
    'parzenwin': taylor,
    'tukeywin': tukey,
    'triang': triang,
    'sasaki': sasaki
    }

    if wname in window_map:
        win_function = window_map[wname]
        return win_function(N, *args)
    else:
        raise ValueError(f"Unknown window name '{wname}'")
    
