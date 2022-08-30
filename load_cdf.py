# Imports
import urllib.request

import numpy as np
import pandas
import datetime
# import wget
import os
import cdflib
from pathlib import Path
import os


def load_tswf(year, month, day, opt=''):
    filetype = 'tswf'

    data_folder = os.path.join(os.getcwd(), 'Downloads')
    for names in os.listdir(os.path.join('Download', ('%04d' % year), ('%02d' % month))):
        if ('solo_L2_rpw-tds-surv-%s-e_%04d%02d%02d_V' % (filetype, year, month, day)) in names:
            fname = names
    print('loading ' + fname)
    cdf = cdflib.CDF(os.path.join('Download', ('%04d' % year), ('%02d' % month), fname))

    data = {}

    if opt == 'nnet':
        for varname in ['Epoch', 'SAMPLING_RATE', 'CHANNEL_ON', 'SAMPS_PER_CH', 'WAVEFORM_DATA',
                        'TDS_CONFIG_LABEL']:
            data[varname] = cdf.varget(varname)
        return data

    (why, varnames) = cdf._get_varnames()
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
    E = np.dot(M, ww[0:2, :])  # transformation into SRF (Y-Z) in (V/m)
    return E
