import numpy as np
import numpy.matlib
import datetime
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fft as fft
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import tdscdf
import cdflib
import jutils

# Waveform plot
def plot_waveform(ww, t0, sr, rec):
    """
        Plotting the TDS-TSWF waveform snapshots
    """
    nsamp = len(ww[0, :])
    ncomp = len(ww[:, 0])
    timestr = t0.item().strftime('%Y/%m/%d, %H:%M:%S.%f')
    tt = np.arange(0, nsamp / sr, 1 / sr) * 1e3

    fig, ax = plt.subplots(nrows=ncomp, ncols=1, sharex=True, sharey=True, figsize=(8, 6), dpi=80)
    if ncomp == 1:
        ax.plot(tt, ww[0, :])
        ax.set(ylabel='$EY_{SRF}$ (mV/m)')
    else:
        for c in np.arange(0, ncomp):
            ax[c].plot(tt, ww[1, :])
            ax[c].set(ylabel='$E{c+1}_{SRF}$ (mV/m)')
    plt.xlabel('Time since trigger (ms)')
    plt.suptitle(('TDS-TSWF waveforms in SRF: %s SWF#%d' % (timestr, rec)))
    plt.xlim(0, nsamp / sr * 1e3)
    plt.show()


# Spectrum
def plot_spectrum(ww, t0, sr, rec):
    """
        Plotting the TDS-TSWF spectra computed from Ey and Ez SRF
    """
    figure(figsize=(8, 6), dpi=80)
    nsamp = int(ww.size / 2)
    tt = np.arange(0, nsamp / sr, 1 / sr)
    fourier_transform = np.fft.rfft(ww)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sr / 2, len(power_spectrum[0, :]))
    xmin = (np.abs(frequency - 200)).argmin()
    if sr > 300000:
        fmax = 200000
    else:
        fmax = 100000
    xmax = (np.abs(frequency - fmax)).argmin()

    plt.plot(frequency[xmin:xmax] * 1e-3, power_spectrum[0, xmin:xmax])
    plt.plot(frequency[xmin:xmax] * 1e-3, power_spectrum[1, xmin:xmax])
    plt.legend(['$EY_{SRF}$', '$EZ_{SRF}$'])

    plt.yscale("log")
    plt.xlim(2, fmax * 1e-3)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power spectral density')
    timestr = t0.item().strftime('%Y/%m/%d, %H:%M:%S.%f')
    plt.title(('SolO TDS TSWF spectrum  %s SWF#%d' % (timestr, rec)))


# Hodogram
def plot_hodogram(ww, t0, rec, size=200, samp=-1):
    """
        Plotting a hodogram from Ey-Ez component
    """
    figure(figsize=(8, 6), dpi=80)
    nsamp = int(ww.size / 2)
    if samp == -1:
        amp = np.abs(ww[0, :]) + np.abs(ww[1, :])
        samp = np.argmax(amp)
        if samp < size / 2:
            samp = 251
        elif samp > nsamp - (size / 2):
            samp = nsamp - 251

    if samp < size / 2:
        samp = size + 1
    elif samp > nsamp - (size / 2):
        samp = nsamp - size - 1

    y = ww[0, int(samp - size):int(samp + size / 2)]
    z = ww[1, int(samp - size):int(samp + size / 2)]

    plt.plot(y, z)
    m = ww.max() * 1.1
    plt.gca().set_aspect('equal')
    plt.xlim(-m, m)
    plt.ylim(-m, m)
    plt.xlabel('$EY_{SRF}$ (mV/m)')
    plt.ylabel('$EZ_{SRF}$ (mV/m)')
    timestr = t0.item().strftime('%Y/%m/%d, %H:%M:%S.%f')
    plt.title(('SolO TDS TSWF hohogram %s SWF#%d' % (timestr, rec)))
    plt.show()


def tds_calculate_swf_spectrogram(swf, stime=None, ftime=None, frqmin=None, frqmax=None, channel=None, return_spectra=False):
    ep = swf['Epoch']
    if stime is None:
        t0 = 0
  
    if ftime is None:
        t1 = len(ep)
        
    if channel is None:
        ch = 0
    else:
        ch = channel
    if channel > 1:
        ch = 1
    ep = ep[t0:t1]    
    wf = swf['WAVEFORM_DATA'][t0:t1, :, :]
    samp_per_ch = swf['SAMPS_PER_CH'][t0:t1]
    wlen = max(samp_per_ch)
    nrec = t1 - t0
    fs = max(swf['SAMPLING_RATE'])
      
    ncomp = len(wf[0, :, 0])
    if ncomp > 1:
        ww = np.zeros((nrec, 2, wlen))
        for r in np.arange(0, nrec):
            ww[r, :, 0:samp_per_ch[r]] = tdscdf.convert_to_SRF(swf, r)
        ncomp = 2    
    else:
        ww = wf
    """
        SETTINGS
    """
    fftlen = 4096 # FFT Length 
    if frqmin is None:
        freqmin = 0. # Minimal frequency; if <=0 then the first set
    else:
        freqmin = frqmin
    if frqmax is None:
        freqmax = fs / 2.
    else:
        freqmax = frqmax # Maximal frequency; if <0 then fs / 2 set
    
    rc_min = 0 # First snapshot to plot; if <0 then the first set
    rc_max = -1 # Last snapshot to plot; if <0 then the last set
    if freqmax < 0:
        freqmax = fs / 2

    if rc_min < 0:
        rc_min = 0
    if rc_max < 0:
        rc_max = nrec
    if rc_max > nrec:
        rc_max = nrec
    if rc_min > rc_max:
        temp = rc_min
        rc_min = rc_max
        rc_max = temp
    
    rc_range = np.arange(rc_min, rc_max)
    nrec = rc_max - rc_min

    nfreq = int(fftlen / 2)

    Pwx = np.zeros((nrec, nfreq))
    Freq = np.full((nfreq + 1), float('nan'))
   

    for rc in rc_range:
        freq, sp = signal.welch(ww[rc, :, :], swf['SAMPLING_RATE'][rc], window='hann', nperseg=fftlen, 
                                       noverlap=fftlen//2 )    
        Pwx[rc-rc_min, :] = sp[ch, 1:]
    find = (np.asarray((freq > freqmin) & (freq < freqmax)).nonzero())[0]
    Freq = freq[find]
    Pwx = Pwx[:, find]
    #df = freq[1] - freq[0]
    #Freq[:, nfreq +1]  = Freq[:, nfreq +1] + df
    
    Epoch = ep[rc_range]
    Pwx[np.asarray(Pwx <= 0).nonzero()] = float('nan')
    
    return Pwx, Freq, Epoch

def tds_plot_swf_spectrogram(x, y, data, ax=None, freqmin=None, freqmax=None,vmin=None,vmax=None):
    
    if not ax:
         ax = plt.gca()

    if freqmin:
        find = np.asarray(y > freqmin).nonzero()
        y = y[find]
        data = data[:,find]

    if freqmax:
        find = np.asarray(y <= freqmax).nonzero()
        y = y[find]
        data = data[:,find]

    time_threshold = 5
    verbose = False
    # Calculate median time step
    median_dt = np.median(np.diff(x))

    # If the length of y matches the number of rows in data, assume y needs fixing
    if y.size == data.shape[1]:
        if verbose:
            print('Attempting to fix frequency or similar (same size as the spectrum). Assuming linear yscale')
        y = np.append(y, y[-1] + (np.median(np.diff(y))))-np.median(np.diff(y))/2
    
    # If the length of x matches the number of rows in data, assume x needs fixing
    if x.size == data.shape[0]:
        if verbose:
            print('Attempting to fix timetags (same size as the spectrum)')
        x = np.append(x, x[-1] + int(median_dt))

    # Repeat frequency axis values to match the shape of data
    y = np.matlib.repmat(y, data.shape[0] + 1, 1)

    # Find where there are gaps in the time axis
    gap_indices = np.where(np.diff(x) > time_threshold * median_dt)[0]

    # Convert TT2000 time to datetime objects
    x_datetime = cdflib.cdfepoch.to_datetime(x)
    x_datetime = np.matlib.repmat(x_datetime, data.shape[1] + 1, 1).T

    minv, maxv = jutils.percentile_2D(data, 10, 90)
        
    
    #Freq = np.arange(1, fftlen/2+1) * fs / fftlen
    #y /= 1e3
    
    minv, maxv = jutils.percentile_2D(np.log10(data), 5, 98)
    minv = 10**np.floor(minv)
    maxv = 10**np.ceil(maxv)
    if vmin:
        minv = vmin
    if vmax:
        maxv = vmax

    im1 = ax.pcolormesh(x_datetime, y, data, shading='auto',cmap=   'rainbow', norm=LogNorm(vmin=minv,vmax=maxv))
    # Fill the gaps with white patches
    for idx in gap_indices:
        ax.fill_betweenx(y[idx,:], x_datetime[idx,0], x_datetime[idx + 1,0], color='white')
    ax.set_ylabel('Frequency [Hz]', fontsize=20)
    #ax.set_xlabel('UTC Time', fontsize=16)
    #ax.set_title(complabel[ch], fontsize=16, loc='right')
    ax.tick_params(labelsize=12)
    # ax.plot([ep[0], ep[-1]], [5e3, 5e3], '--', color='magenta')
    c1 = plt.colorbar(im1, ax=ax, pad=0.01)
    c1.set_label('$\mathrm{V^2m^{-2}Hz^{-1}}$', fontsize=12)
    return ax

def tds_plot_mamp(ax, mamp, channel=0):
    mamp['WAVEFORM_DATA'][np.asarray(mamp['WAVEFORM_DATA'] < 0).nonzero()] = float('nan')
    ep = cdflib.cdfepoch.to_datetime(mamp['Epoch'])    
    sdate = (ep[0].astype(datetime.datetime)).strftime('%Y/%m/%d')
    complabel = f"V{int(mamp['CHANNEL_REF'][0][channel]/10)}-V{np.mod(mamp['CHANNEL_REF'][0][channel],10)}"
    mamp['WAVEFORM_DATA'][np.asarray(mamp['WAVEFORM_DATA'] < 0).nonzero()] = float('nan')
    ax.semilogy(ep, mamp['WAVEFORM_DATA'][:,channel], label='full rate')
    nav = int(mamp['SAMPLING_RATE'][0])
    ax.semilogy(ep[nav//2-1:-nav//2], jutils.moving_average(mamp['WAVEFORM_DATA'][:,channel], nav), 'orange', label='1s mov avg')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(f"SolO RPW MAMP & SWF - {sdate}", fontsize=16)
    ax.set_ylabel('Amplitude [V/m]', fontsize=16)
    ax.set_title(complabel, fontsize=16, loc='right')
    ax.legend()

def create_spectrogram(x, y, data, time_threshold=5, verbose=True, labelfontsize=16, tickfontsize=14, argtitle='', argxlabel='UTC Time', argylabel='Frequency (kHz)', argcblabel='log($V^2/m^2/Hz$)',yscale='linear',nocrop=False):
    """
    Create a spectrogram with gaps in the time axis.

    Parameters:
        x (array-like): Time axis values in TT2000 format.
        y (array-like): Frequency axis values.
        data (2D array-like): Spectrogram data.
        time_threshold (float): Threshold for detecting gaps in time axis (in TT2000 units).
        verbose (bool): Whether to print additional information for debugging. Default is True.
        labelfontsize (int): Font size for axis labels. Default is 16.
        tickfontsize (int): Font size for axis ticks. Default is 14.
        argtitle (str): Title for the spectrogram plot. Default is empty string.
        argxlabel (str): Label for the x-axis. Default is 'UTC Time'.
        argylabel (str): Label for the y-axis. Default is 'Frequency (kHz)'.
        argcblabel (str): Label for the colorbar. Default is 'log($V^2/m^2/Hz$)'.

    Returns:
        None. Plot the spectrogram.
    """

    # Calculate median time step
    median_dt = np.median(np.diff(x))

    # If the length of y matches the number of rows in data, assume y needs fixing
    if y.size == data.shape[1]:
        if verbose:
            print('Attempting to fix frequency or similar (same size as the spectrum). Assuming linear yscale')
        y = np.append(y, y[-1] + (np.median(np.diff(y))))-np.median(np.diff(y))/2
    
    # If the length of x matches the number of rows in data, assume x needs fixing
    if x.size == data.shape[0]:
        if verbose:
            print('Attempting to fix timetags (same size as the spectrum)')
        x = np.append(x, x[-1] + int(median_dt))

    # Repeat frequency axis values to match the shape of data
    y = np.matlib.repmat(y, data.shape[0] + 1, 1)

    # Find where there are gaps in the time axis
    gap_indices = np.where(np.diff(x) > time_threshold * median_dt)[0]

    # Convert TT2000 time to datetime objects
    x_datetime = cdflib.cdfepoch.to_datetime(x)
    x_datetime = np.matlib.repmat(x_datetime, data.shape[1] + 1, 1).T

    minv, maxv = jutils.percentile_2D(data, 10, 90)
    if nocrop:
        plt.pcolormesh(x_datetime, y, data, shading='auto',cmap='jet')
    else:
        plt.pcolormesh(x_datetime, y, data, shading='auto',vmin=minv,vmax=maxv,cmap='jet')
    
    # Fill the gaps with white patches
    for idx in gap_indices:
        plt.fill_betweenx(y[idx,:], x_datetime[idx,0], x_datetime[idx + 1,0], color='white')
        
    
    
    plt.yscale(yscale)
    plt.ylabel(argylabel, fontsize=labelfontsize)
    plt.xlabel(argxlabel, fontsize=labelfontsize)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title(argtitle, fontsize=labelfontsize)
    cbar = plt.colorbar()
    cbar.set_label(label=argcblabel,fontsize=labelfontsize) 
    plt.title(argtitle, fontsize=labelfontsize)
    plt.tick_params(axis='both', which='major', labelsize=tickfontsize)


# Example usage:
# Assuming you have your TT2000 time axis, frequency axis, and spectrogram data ready
# x, y, data = ...
# create_spectrogram(x, y, data)

# Function to do the fourier transform

def calculate_spectrogram(cdfdata, fftlen=256, freqmin=0., freqmax=8e4, nfreq=None):
    if nfreq is None:
        nfreq = int(fftlen / 2 + 1)

    for d in [cdfdata]:
        ep = d['Epoch']
        wf = d['WAVEFORM_DATA']
        fs = d['SAMPLING_RATE'][0]
        ncomp = sum(d['CHANNEL_ON'][0][:])
        wlen = min(d['SAMPS_PER_CH'][:])
        if fftlen > wlen:
            fftlen = wlen
        nrec = len(ep)
        if ncomp > 1:
            ncomp = 2
        ww = np.zeros((nrec, ncomp, wlen))
        Pwx = np.zeros((nrec, ncomp, nfreq))
        for rc in np.arange(0, nrec-1):
            if ncomp > 1:
                ww[rc, :] = tdscdf.convert_to_SRF(d, rc)  # Converts the first snap
            else:
                ww[rc, 0] = wf[rc, :]
            freq, sp = signal.welch(ww[rc, :, :], fs, window='hann', nperseg=fftlen,
                                     noverlap=fftlen//2)
            Pwx[rc, :, :] = sp[:, :]

        if ncomp > 1:
            complabel = ('Ey', 'Ez')
        else:
            complabel = (f"V{int(d['CHANNEL_REF'][0][0]/10)}-V{np.mod(d['CHANNEL_REF'][0][0], 10)}",)

        if (freqmax < 0) or (freqmax > fs / 2):
            freqmax = fs / 2
        find = (np.asarray((freq > freqmin) & (freq < freqmax)).nonzero())[0]
        freq = freq[find] / 1e3

        if ncomp == 1:
            Pwx[np.asarray(Pwx <= 0).nonzero()] = float('nan')
            Pxx = np.log10(np.transpose(Pwx[:, 0, :]))
            Pxx = Pxx[find, :]
            return np.array(Pxx)

        else:
            Pxx_list = []
            for c in np.arange(0, ncomp):
                Pwx[np.asarray(Pwx <= 0).nonzero()] = float('nan')
                Pxx = np.log10(np.transpose(Pwx[:, c, :]))
                Pxx = Pxx[find, :]
                Pxx_list.append(Pxx)
            return np.array(Pxx_list), freq

def movmean(arr, window_size):
    """
    Calculates the moving average of a 2D array.

    Parameters:
        arr (numpy.ndarray): Input 2D array.
        window_size (tuple): Tuple specifying the number of records to be averaged
                             in each dimension.

    Returns:
        numpy.ndarray: Array with the same dimension as the input array containing
                       the moving averages.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    if not isinstance(window_size, tuple) or len(window_size) != 2:
        raise ValueError("Window size must be a tuple of length 2")

    # Convert array to float type to allow NaN values
    arr = arr.astype(float)

    # Extract window size for each dimension
    window_size_row, window_size_col = window_size

    # Pad the array to handle boundaries
    padded_arr = np.pad(arr, ((window_size_row // 2, window_size_row // 2),
                               (window_size_col // 2, window_size_col // 2)),
                        mode='constant', constant_values=np.nan)

    # Define a function to calculate moving average for a window
    def moving_avg_window(window):
        return np.nanmean(window)

    # Apply moving average using a convolution
    moving_avg = np.array([[moving_avg_window(padded_arr[i:i+window_size_row, j:j+window_size_col])
                            for j in range(arr.shape[1])]
                           for i in range(arr.shape[0])])

    return moving_avg

# Example usage:
# Pxx = calculate_spectrogram(cdfdata, fftlen=256, freqmin=0., freqmax=8e4.)
def _percentile_2D(arr, low=0.0, up=100.):
    flatened_arr = arr.flatten()
    return np.nanpercentile(flatened_arr, low), np.nanpercentile(flatened_arr, up)
