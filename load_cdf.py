# Imports
import urllib.request

import numpy as np
import pandas
import datetime
# import wget
import os
import cdflib
import cdflib.epochs as epoch
from pathlib import Path
import os
import matplotlib.pyplot as plt


def load_tswf(year, month, day, opt=''):
    filetype = 'tswf'

    data_folder = os.path.join(os.getcwd(), 'Downloads')
    for names in os.listdir(os.path.join('Download', ('%04d' % year), ('%02d' % month))):
        if ('solo_L2_rpw-tds-surv-%s-e_%04d%02d%02d_V' % (filetype, year, month, day)) in names:
            fname = names
    print('loading ' + fname)
    cdf = cdflib.CDF(os.path.join('Download', ('%04d' % year), ('%02d' % month), fname))

    data = {}
    info = cdf.cdf_info()
    varnames = info['zVariables']
    for varname in varnames:
        data[varname] = cdf.varget(varname)
    return data


def download_tswf(date='2021-10-09', output_dir='Download'):
    # Input args
    # date          Date of CDF CDF files to download in format YYYY-MM-DD')
    # output_dir    Directory where resulting CDFs will be saved [{OUTPUT_DIR}].

    descriptor = 'RPW-TDS-SURV-TSWF-E'

    try:
        y = int(date[0:4])
        m = int(date[5:7])
        d = int(date[8:10])
    except:
        print('ERROR: Incorrect date input, use the YYYY-MM-DD format')
        return

    output_dir = os.path.join(output_dir, ('%04d' % y), ('%02d' % m))
    descriptor = descriptor.upper()
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
    # Looking for metadata
    instru = descriptor[0:3]
    date1 = datetime.datetime(y, m, d, 0, 0, 0, 0)
    date0 = date1 - datetime.timedelta(days=1)
    date0 = date0.strftime('%Y-%m-%' + 'd')
    link = 'http://soar.esac.esa.int/soar-sl-tap/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=SELECT+data_item_id,descriptor,level+FROM+v_sc_data_item+WHERE+begin_time%' + '3E' + '%27' + date0 + '%27+AND+begin_time%' + '3C' + '=%27' + date + '%27+AND+file_format=%27CDF%27+AND+level=%27L2%27+ORDER+BY+begin_time+ASC'
    #    print(link)
    f = urllib.request.urlopen(link)
    myfile = pandas.read_csv(f)
    # Saving data_item_id to list
    fullnamelist = myfile.data_item_id.tolist()
    namelist = []
    index = 0
    for i in myfile.descriptor.tolist():
        if descriptor in i:
            namelist.append(fullnamelist[index])
        index += 1

    # Dowloading with wget
    progress = 1
    for data_item_id in namelist:
        print(data_item_id + 'item ' + str(progress) + '/' + str(len(namelist)) + '\n')
        progress += 1
        # outFilename = os.path.join(output_dir, data_item_id + '.cdf')
        # print(outFilename)
        url = 'http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=LAST_PRODUCT&data_item_id=' + data_item_id + '&product_type=SCIENCE'
        cmd = 'wget -q -nc -P ' + output_dir + ' --content-disposition "' + url + '"'
        print('downloading cdf file from ' + url)
        os.system(cmd)
        print('download complete')
    #       wget.download('http://soar.esac.esa.int/soar-sl-tap/data?retrieval_type=PRODUCT&data_item_id=' + data_item_id + '&product_type=SCIENCE', out = outFilename)


def convert_to_SRF(data, index=0):
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

def plot_waveform(cdf, rec=0):
    ww = convert_to_SRF(cdf, rec)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    sr = cdf['SAMPLING_RATE'][rec]
    nsamp = cdf['SAMPS_PER_CH'][rec]
    t0 = epoch.CDFepoch.to_datetime(cdf['Epoch'][rec])
    t0 = t0[0].strftime('%Y/%m/%d, %H:%M:%S.%f')
    tt = np.arange(0, nsamp/sr, 1/sr)*1e3
    ax1.plot(tt,ww[0, :])
    ax1.set(ylabel='$EY_{SRF}$ (mV/m)')
    plt.xlabel('Time since trigger (ms)')
    ax2.plot(tt,ww[1, :])
    ax2.set(ylabel='$EZ_{SRF}$ (mV/m)')
    plt.xlabel('Time since trigger (ms)')
    plt.suptitle(('TDS-TSWF waveforms in SRF: %s SWF#%d' % (t0, rec)))
    plt.xlim(0, nsamp/sr*1e3)
    plt.show()

def plot_spectrum(cdf, rec=0):
    nsamp = cdf['SAMPS_PER_CH'][rec]
    ww = convert_to_SRF(cdf, rec)
    sr = cdf['SAMPLING_RATE'][rec]
    tt = np.arange(0, nsamp/sr, 1/sr)
    fourier_transform = np.fft.rfft(ww)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sr / 2, len(power_spectrum[0,:]))
    xmin = (np.abs(frequency - 200)).argmin()
    xmax = (np.abs(frequency - min(sr/2, 200000))).argmin()
    plt.plot(frequency[xmin:xmax]*1e-3, power_spectrum[0, xmin:xmax])
    plt.plot(frequency[xmin:xmax]*1e-3, power_spectrum[1, xmin:xmax])
    plt.yscale("log")
    plt.xlim(2, min(sr / 2, 200))
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power spectral density')
    t0 = epoch.CDFepoch.to_datetime(cdf['Epoch'][rec])
    t0 = t0[0].strftime('%Y/%m/%d, %H:%M:%S.%f')
    plt.title(('SolO TDS TSWF spectrum  %s SWF#%d' % (t0, rec)))
    plt.show()


def plot_hodogram(cdf, rec=0, size=200, samp=-1):
    ww = convert_to_SRF(cdf, rec)
    nsamp = cdf['SAMPS_PER_CH'][rec]
    if samp == -1:
        amp = np.abs(ww[0, :]) + np.abs(ww[1, :])
        samp = np.argmax(amp)
        if samp<size/2:
            samp = 251
        elif samp>nsamp-(size/2):
            samp = nsamp-251

    if samp < size / 2:
        samp = size + 1
    elif samp > nsamp - (size / 2):
        samp = nsamp - size - 1

    y = ww[0, samp-size:int(samp+size/2)]
    z = ww[1, samp-size:int(samp+size/2)]
    plt.plot(y,z)
    m = ww.max()*1.1
    plt.xlim(-m,m)
    plt.ylim(-m,m)
    plt.xlabel('$EY_{SRF}$ (mV/m)')
    plt.ylabel('$EZ_{SRF}$ (mV/m)')
    t0 = epoch.CDFepoch.to_datetime(cdf['Epoch'][rec])
    t0 = t0[0].strftime('%Y/%m/%d, %H:%M:%S.%f')
    plt.title(('SolO TDS TSWF hodogram %s SWF#%d' % (t0, rec)))
    plt.show()

# Example:
#
# cdf=download_tswf('2021-11-15')
# cdf=load_tswf(2021, 11, 15)
# plot_spectrum(cdf,0)
# plot_waveform(cdf,0)
# plot_hodogram(cdf,0)