# Imports
import urllib.request

import numpy as np
import pandas
import datetime
import os
import cdflib
import cdflib.epochs as epoch
import datetime
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# Loading downloaded cdf files
def load_data(time, tlen=None, product='surv-tswf'):
    """
        Loading TDS CDF file
        date - desired date as datetime.datetime(year, month, day) where
            year - given year (>=2020)
            month - given month (1-12)
            day - give day (1-31)
    """
    # assert isinstance(stime, 'datetime.datetime')
    stime = cdflib.epochs.CDFepoch.compute_tt2000(time)
    date = cdflib.epochs.CDFepoch.compute_tt2000(time[0:3])
    if tlen is None:
        ftime = date + np.int64(86400e9)
    else:
        ftime = stime + np.int64(tlen * 1e9)
    
    delta_days = np.ceil((ftime - date) / 86400e9)
    if delta_days == 0:
        delta_days = 1
    data = dict()
    for d in np.arange(0, delta_days):
        if d == 0:
            dat = load_date(stime, ftime, product=product)
        else:
            dat = load_date(np.int64(stime+(d * 86400e9)), ftime, product=product)
        for dd in dat:
            if dd is None:
                continue
            if not data:
                for v in dd.keys():
                    data[v] = dd[v]
            else:
                for v in dd.keys():
                    if v == 'varinq':
                        continue
                    if data['varinq'][v].Rec_Vary:
                        data[v] = np.concatenate((data[v], dd[v]))
    return data                             
            
def load_date(stime, ftime=None, product='surv-tswf'):
    """
        Loading TDS CDF file
        date - desired date as datetime.datetime(year, month, day) where
            year - given year (>=2020)
            month - given month (1-12)
            day - give day (1-31)
    """
    solo_dir = 'data/rpw/L2'
    if product == 'surv-tswf':
        product = 'surv-tswf-e'
        solo_dir += '/tds_wf_e'
    elif product == 'surv-rswf':
        product = 'surv-rswf-e'
        solo_dir += '/tds_wf_e'
    elif product == 'sbm1-rswf':    
        product = 'sbm1-rswf-e'
        solo_dir += '/tds_sbm1_wf_e'
    elif product == 'sbm2-tswf':
        product = 'sbm2-tswf-e'
        solo_dir += '/tds_sbm2_wf_e'
    elif product == 'surv-mamp':
        solo_dir += '/tds_mamp'
    elif product == 'surv-stat':
        solo_dir += '/tds_stat'
    else:
        print(f'Unknown or Unsupported TDS product - {product}! Exiting...')
        return None
    
    fname = list()
    t0 = cdflib.epochs.CDFepoch.breakdown_tt2000(stime)    
    year = t0[0]
    month = t0[1]
    day = t0[2]
    if ftime is None:
        ftime = cdflib.epochs.CDFepoch.compute_tt2000((year, month, day, 23, 59, 59))
    for names in os.listdir(os.path.join(solo_dir, f"{year:4d}/{month:02d}")):
        if (names.find(f"solo_L2_rpw-tds-{product}-cdag_{year:4d}{month:02d}{day:02d}") != -1):
            fname.append(names)
            print('loading ' + names)
    datas = list()
    if not fname:
        print(f"No local file(s) found for {year:4d}/{month:02d}/{day:02d}")
        print(f"Downloading...")
        fname = download_data(t0[0:3], product)

        
    for f in fname:
        cdf = cdflib.CDF(os.path.join(solo_dir, f"{year:4d}/{month:02d}", f))
        data = {}
        info = cdf.cdf_info()
        varnames = info.zVariables
        if len(varnames)>0:
            data['varinq'] = dict()
        for varname in varnames:
            data[varname] = cdf.varget(varname, epoch='Epoch', starttime=np.int64(stime), endtime=np.int64(ftime))
            data['varinq'][varname] = cdf.varinq(varname)
        datas.append(data)
    return datas


# Download TDS-SURV-TSWF cdf file for a given date.
# !! the file might have a size of several hundreds of MB
def download_data(date, product, output_dir='./Download'):
    # Input args
    # date          Date of CDF CDF files to download as datetime.datetime() object
    # output_dir    Directory where resulting CDFs will be saved [{OUTPUT_DIR}].

    y = date[0]
    m = date[1]
    d = date[2]
    descriptor = f'RPW-TDS-{product.upper()}'
    if product == 'surv-tswf':
        descriptor += '-E'
    elif product == 'surv-rswf':
        descriptor += '-E'
    elif product == 'sbm1-rswf':    
        descriptor += '-E'
    elif product == 'sbm2-tswf':
        descriptor += '-E'
        
    output_dir = os.path.join(output_dir, ('%04d' % y), ('%02d' % m))
    descriptor = descriptor.upper()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Looking for metadata
    instru = descriptor[0:3]
    date0 = datetime.datetime(y, m, d, 0, 0, 0, 0)
    date1 = date0 + datetime.timedelta(days=1)
    sdate = date0.strftime('%Y-%m-%d')
    sdate1 = date1.strftime('%Y-%m-%d')
    link = f"http://soar.esac.esa.int/soar-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=SELECT+data_item_id,descriptor,level+FROM+v_sc_data_item+WHERE+begin_time%3E=%27{sdate}%27+AND+begin_time%3C%27{sdate1}%27+AND+file_format=%27CDF%27+AND+level=%27L2%27+AND+instrument=%27RPW%27+AND+descriptor=%27{descriptor.lower()}%27+ORDER+BY+begin_time+ASC"
    f = urllib.request.urlopen(link)
    myfile = pandas.read_csv(f)
    # Saving data_item_id to list
    fnames = myfile.data_item_id.tolist()
    # Dowloading with wget
    progress = 1
    fpath = list()
    for fn in enumerate(fnames):
        ind, data_item_id = fn
        print(f"{data_item_id} item {ind+1} /  {len(fnames)} \n")
        url = f'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=LAST_PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE'
        cmd = f'wget -q -nc -P {output_dir} --content-disposition "{url}"'
        oFile = os.path.join(output_dir, data_item_id + '.cdf')
        if not os.path.isfile(oFile):
            os.system(cmd)
            print('download complete')
        fpath.append(oFile)
    return fpath

# Convering to SRF
def convert_to_SRF(data, index=0):
    """
        Convert TDS SWF from the antenna to the spacecraft
        reference frame (SRF).
        Using the effective antenna directions
        two components of the E-field in the Y-Z SRF plane
        is calculated for the given TDS configuration.
        data(ncomp, nsamples) - input array 2-D vectors of the electric field expressed in the
                    ANT coordinate system in V/m
        index - a snapshot number to be transformed, the first is default
        E(2,nsamples) - 2D E-field E[0, *] = Ey, E[1, *] = Ez
    """
    nsamp = data['SAMPS_PER_CH'][index]
    tds_mode = data['TDS_CONFIG_LABEL'][index]
    if 'SE1' in tds_mode:
        # Pachenko's antenna angle
        pacang = np.deg2rad(125)
        V1 = [0, 1]
        # Pachenko monopole antennas
        V2 = [np.sin(pacang), np.cos(pacang)]
        V3 = [-np.sin(pacang), np.cos(pacang)]
        # SE1 TDS mode
        M = np.array([V1, V2])  # [CH1, CH2]
    else:
        pacang = np.deg2rad(158.1)
        ant21 = [np.sin(pacang), np.cos(pacang)]  # E - field in the same sense.
        pacang = np.deg2rad(-158.2)
        ant13 = [-np.sin(pacang), -np.cos(pacang)]  # ant31 then - 1. to ant13
        M = np.array([ant13, ant21])  # [CH1, CH2]

    ww = data['WAVEFORM_DATA'][index, :, 0:nsamp]
    # projection: E = MAT(ANT->SRF) * V; where MAT(2,2) and V is observed field
    M = np.linalg.inv(M)
    E = np.dot(M, ww[0:2, :]) * 1e3  # transformation into SRF (Y-Z) in (mV/m)
    return E


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


# Example:
#
# dt = datetime.datetime(2021,11,15)
# cdf=download_tswf(dt)
# cdf=load_tswf(dt)
# plot_spectrum(cdf,0)
# plot_waveform(cdf,0)
# plot_hodogram(cdf,0)