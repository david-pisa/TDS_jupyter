import tdscdf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cdflib
import pickle

def percentile_2D(arr, low=0.0, up=100.):
    flatened_arr = arr.flatten()
    return np.nanpercentile(flatened_arr, low), np.nanpercentile(flatened_arr, up)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def tds_collect_stat():
    t0 = datetime.datetime(2020,7,1)
    t1 = datetime.datetime.now() # excluding

    print('Collecting TDS-STAT data...')
    print(f"Startime {t0.strftime('%Y-%m-%d')}")
    print(f"Stoptime {t1.strftime('%Y-%m-%d')}")
    dt = t1 - t0
    nday = dt.total_seconds() // 86400
    if nday==0:
        nday = int(1)
    stat = dict()
    stat_total = dict()
    stat_total['Epoch'] = list()
    stat_total['records'] = list()
    s_var = ['Epoch', 'DU_MED_AMP', 'DU_NR_IMPACT', 
             'SN_THRESHOLD', 'SN_RMS_E', 'SN_MED_MAX_E', 
             'SN_MAX_E', 'SN_NR_EVENTS', 'CHANNEL_REF', 
             'SNAPSHOT_LEN', 'TDS_CONFIG_LABEL', 'INPUT_CONFIG', 
             'SURVEY_MODE', 'SAMPLING_RATE']
    for d in np.arange(0, nday):
        tday = datetime.timedelta(days=int(d))
        dd = t0 + tday
        date = [dd.year, dd.month, dd.day]
        s = tdscdf.load_data(date, product='surv-stat')
        if not s:
            continue
        stat_total['Epoch'].append(s['Epoch'][0])
        stat_total['records'].append(len(s['Epoch']))  
        ind = np.asarray((s['DU_NR_IMPACT'][:] > 0).nonzero())[0]
        if stat:
            for z in s_var:                
                stat[z] = np.concatenate((stat[z], s[z][ind]))
        else:
            for z in s_var:
                stat[z] = s[z][ind]
    file = './dust/tds_stat_dust.pkl'
    with open(file, 'wb') as f:
       # np.save(f, stat_total)
        pickle.dump(stat, f)
    print(f'Done and stored at {file}')