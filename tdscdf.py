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
    if ftime > date + np.int64(86400e9):
        print('Loading multiple files is not supported yet.')                       
        ftime = date + np.int64(86400e9)
    delta_days = np.ceil((ftime - date) / 86400e9)
    if delta_days == 0:
        delta_days = 1
    data = dict()
    for d in np.arange(0, delta_days):
        if d == 0:
            dat = load_date(stime, ftime, product=product)
        else:
            dat = load_date(np.int64(date+(d * 86400e9)), ftime, product=product)
        for dd in dat:
            if dd is None:
                continue
            if not data:
                for v in dd.keys():
                    data[v] = dd[v]
            else:
                if 'WAVEFORM_DATA' in dd.keys():
                    data = _def_concatenate_swf(data, dd)
                else:
                    for v in dd.keys():
                        if v == 'varinq':
                            continue
                        if data['varinq'][v].Rec_Vary:
                            data[v] = np.concatenate((data[v], dd[v]))

    if 'varinq' in data:
        del data['varinq']                            
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
    idir = os.path.join(solo_dir, f"{year:4d}/{month:02d}")
    if os.path.isdir(idir):
       # print(f"No folder ({idir}) found! Exiting")
        #return dict()
        for names in os.listdir(idir):
            if (names.find(f"solo_L2_rpw-tds-{product}-cdag_{year:4d}{month:02d}{day:02d}") != -1):
                fname.append(f"{year:4d}/{month:02d}/{names}")
                print('loading ' + names)
    if not fname:
        print(f"No local file(s) found for {year:4d}/{month:02d}")
        print(f"Downloading...")
        fname = download_data(t0[0:3], f'rpw-tds-{product}')
        solo_dir = './'
    
    datas = list()    
    for f in fname:
        cdf = cdflib.CDF(os.path.join(solo_dir, f))
        data = {}
        info = cdf.cdf_info()
        varnames = info.zVariables
        if len(varnames)>0:
            data['varinq'] = dict()
        for varname in varnames:
            try:
                data[varname] = cdf.varget(varname, epoch='Epoch', starttime=np.int64(stime), endtime=np.int64(ftime))
            except:
                #data[varname] = -1
                print(f'Empty var: {varname} for {f}')
                continue
            data['varinq'][varname] = cdf.varinq(varname)
        datas.append(data)
    return datas


# Download TDS-SURV-TSWF cdf file for a given date.
# !! the file might have a size of several hundreds of MB
import os
import datetime
import urllib.request
import urllib.parse
import pandas as pd
import requests

def download_data(date, descriptor, output_dir='./Download'):
    """
    Download ESA SOAR CDF data files for a given date and descriptor.

    Args:
        date (datetime.datetime): Date of data to download.
        descriptor (str): Instrument data product descriptor (e.g. 'RPW_L2_XYZ').
        output_dir (str): Directory where resulting CDFs will be saved.

    Returns:
        list of str: Paths to downloaded files.
    """
    y = date[0]
    m = date[1]
    d = date[2]

    output_dir = os.path.join(output_dir, f"{y:04d}", f"{m:02d}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    descriptor = descriptor.upper()
    instru = descriptor[0:3]  # Assuming first 3 chars are instrument code
    
    # Construct date range for query (one full day)
    date0 = datetime.datetime(y, m, d)
    date1 = date0 + datetime.timedelta(days=1)
    sdate = date0.strftime('%Y-%m-%d')
    sdate1 = date1.strftime('%Y-%m-%d')

    # Construct query URL with descriptor and instrument dynamically included
    url = (
        f"http://soar.esac.esa.int/soar-sl-tap/tap/sync?"
        f"REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY="
    )
    query = (
        f"SELECT data_item_id,descriptor,level,filename FROM v_sc_data_item "
        f"WHERE begin_time>='{sdate}' AND begin_time<'{sdate1}' "
        f"AND file_format='CDF' AND level='L2' AND instrument='{instru}' "
        f"AND descriptor='{descriptor.lower()}' ORDER BY begin_time ASC"
    )
    url = url + urllib.parse.quote(query)
    # Retrieve metadata CSV from SOAR TAP service
    try:
        with urllib.request.urlopen(url) as f:
            myfile = pd.read_csv(f)

        fnames = myfile.filename.tolist()
        data_items = myfile.data_item_id.tolist()
    except:
        print("Download troubles...")
        return []
    
    fpath = []
    for ind, data_item_id in enumerate(data_items):
        print(f"Downloading item {ind+1} of {len(fnames)}: {fnames[ind]}")
        download_url = (
            f"http://soar.esac.esa.int/soar-sl-tap/data?"
            f"retrieval_type=LAST_PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE"
        )
        oFile = os.path.join(output_dir, fnames[ind])
    
        if not os.path.isfile(oFile):
            try:
                with requests.get(download_url, stream=True) as r:
                    r.raise_for_status()
                    with open(oFile, 'wb') as f_out:
                        for chunk in r.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                print('Download complete')
            except Exception as e:
                print(f"Failed to download {fnames[ind]}: {e}")
                continue
        else:
            print('File already exists, skipping download')

        fpath.append(oFile)
    '''
    for ind, data_item_id in enumerate(data_items):
        print(f"Downloading item {ind+1} of {len(fnames)}: {fnames[ind]}")
        url = (
            f"http://soar.esac.esa.int/soar-sl-tap/data?"
            f"retrieval_type=LAST_PRODUCT&data_item_id={data_item_id}&product_type=SCIENCE"
        )
        oFile = os.path.join(output_dir, fnames[ind])

        if not os.path.isfile(oFile):
            # Use wget command to download the file quietly, without overwriting existing file
            cmd = f'wget -nc -P {output_dir}/ --content-disposition -O {fnames[ind]} {url}'
            print(os.getcwd(),cmd)
            os.system(cmd)
            print('Download complete')
        else:
            print('File already exists, skipping download')
        '''
    
        #fpath.append(oFile)

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

def _def_concatenate_swf(dat, dd):
    ich = dd['CHANNEL_REF'][0,:]
    #ich = ich[ich <= 40]
    och = dat['CHANNEL_REF'][0,:]
    #och = och[och <= 40]
    # create list of merge channels
    ch = np.concatenate((och, ich))
    # create list of unique channels ignoring 255 (! check in TDS_CALBA! )
    ch = np.unique(ch[ch <= 40])
    if np.array_equal(ich, ch):
        for v in dd.keys():
            if v == 'varinq':
                continue
            if dat['varinq'][v].Rec_Vary:
                dat[v] = np.concatenate((dat[v], dd[v]))
    else:
        for v in dd.keys():
            if v == 'varinq':
                continue
            if dat['varinq'][v].Rec_Vary:
                if (dat[v].ndim == 1) | (v == 'RPW_ANTENNA_RTN'):
                    dat[v] = np.concatenate((dat[v], dd[v]))        
                elif dat[v].ndim == 2:
                    oshape = dat[v].shape
                    ishape = dd[v].shape
                    en_ch = np.where(ch <= 40)[0]
                    temp = np.zeros((oshape[0] + ishape[0], len(ch)), dtype=dat[v].dtype)
                    temp[0:oshape[0], 0:oshape[1]] = dat[v][:, en_ch]
                    ch_ptr = len(en_ch)
                    for c in ich:
                        ind = np.where(och == c)[0]
                        idx = np.where(ich == c)[0]
                        if len(ind) == 1:
                            temp[oshape[0]:oshape[0]+ishape[0], ind] = dd[v][:, idx]
                        else:
                            temp[oshape[0]:oshape[0]+ishape[0], ch_ptr] = dd[v][:, idx]
                            ch_ptr += 1
                    dat[v] = temp 
                elif dat[v].ndim == 3:
                    oshape = dat[v].shape
                    ishape = dd[v].shape
                    en_ch = np.where(ch <= 40)[0]
                    max_samps = np.max((dat[v].shape[2], dd[v].shape[2]))
                    temp = np.full((dat[v].shape[0] + dd[v].shape[0], len(ch), max_samps), -1e31)
                    temp[0:oshape[0], 0:len(en_ch), 0:oshape[2]] = dat[v][:, en_ch, :]
                    ch_ptr = oshape[1]
                    for c in ich:
                        ind = np.where(och == c)[0]
                        idx = np.where(ich == c)[0]
                        if len(ind) > 0:
                            temp[oshape[0]:oshape[0]+ishape[0], ind, 0:ishape[2]] = dd[v][:, idx, :]
                        else:
                            temp[oshape[0]:oshape[0]+ishape[0], ch_ptr, 0:ishape[2]] = dd[v][:, idx, :]
                            ch_ptr += 1
                    dat[v] = temp                   
    return dat
