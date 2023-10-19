import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.pyplot import figure
from matplotlib.colors import LogNorm
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


def tds_plot_swf_spectrogram(ax, swf, stime=None, ftime=None, frqmin=None, frqmax=None):

    ep = swf['Epoch']
    if stime is None:
        t0 = 0
    
    if ftime is None:
        t1 = len(ep)
        
    ep = ep[t0:t1]    
    wf = swf['WAVEFORM_DATA'][t0:t1, :, :]
    samp_per_ch = swf['SAMPS_PER_CH'][t0:t1]
    wlen = max(samp_per_ch)
    nrec = t1 - t0
    fs = max(swf['SAMPLING_RATE'])
    print(fs)
      
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
    fftlen = 256 # FFT Length 
    freqmin = 0. # Minimal frequency; if <=0 then the first set
    freqmax = 1e5 # Maximal frequency; if <0 then fs / 2 set
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

    Pwx = np.zeros((nrec, 2, nfreq))
    for rc in rc_range:
        freq, sp = signal.welch(ww[rc, :, :], swf['SAMPLING_RATE'][rc], window='hann', nperseg=fftlen, 
                                       noverlap=fftlen//2 )    
        if fs==swf['SAMPLING_RATE'][rc]:
            Pwx[rc-rc_min, :, :] = sp[:, 1:]
        else:    
            Pwx[rc-rc_min, :, :] = sp[:, 1:]
            
    if ncomp > 1:
        complabel = ('Ey', 'Ez')
    else:
        complabel = list()
        for c in np.arange(0, ncomp):
            complabel.append(f"V{int(swf['CHANNEL_REF'][t0][c]/10)}-V{np.mod(swf['CHANNEL_REF'][t0][c], 10)}",)
        # Plot the data as a color map
    
    times = cdflib.cdfepoch.to_datetime(ep[rc_range])
    #fig, ax = plt.subplots()
    #fig.suptitle(f"SolO RPW TDS-SURV-TSWF \n {times[0]} - {times[-1]}", fontsize=14)
    find = (np.asarray((freq > freqmin) & (freq < freqmax)).nonzero())[0]
    Pwx[np.asarray(Pwx <= 0).nonzero()] = float('nan')
    Pxx = np.transpose(Pwx[:, 0, :])
    Pxx = Pxx[find, :]
    #Freq = np.arange(1, fftlen/2+1) * fs / fftlen
    freq /= 1e3
    # rotate IMAGE and cut upper frequencies
    minv, maxv = jutils.percentile_2D(np.log10(Pxx), 5, 98)
    minv = np.floor(minv)
    maxv = np.ceil(maxv)
    im1 = ax.imshow(Pxx, cmap='rainbow', norm=LogNorm(vmin=10**minv, vmax=10**maxv), aspect='auto', extent=[times[0], times[-1], freq[-1], freq[0]])
    ax.invert_yaxis()
    ax.set_ylabel('Freqency [kHz]', fontsize=16)
    ax.set_xlabel('UTC Time', fontsize=16)
    ax.set_title(complabel[0], fontsize=16, loc='right')
    ax.tick_params(labelsize=8)
    # ax.plot([ep[0], ep[-1]], [5e3, 5e3], '--', color='magenta')
    c1 = plt.colorbar(im1)
    c1.set_label('V^2/m^2/Hz')
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