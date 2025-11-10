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
    t0 = datetime.datetime(2020,6,29)
    t1 = datetime.datetime(2025,1,1)
    #t1 = datetime.datetime.now() # excluding
    bad_days = [ 
    datetime.datetime(2020,5,14),
    datetime.datetime(2020,5,15),
    datetime.datetime(2020,11,17),
    datetime.datetime(2020,12,27),
    datetime.datetime(2021,1,24),
    datetime.datetime(2021,2,4),
    datetime.datetime(2021,2,5),
    datetime.datetime(2021,2,13),
    datetime.datetime(2021,2,14),
    datetime.datetime(2021,2,15),
    datetime.datetime(2021,2,16),
    datetime.datetime(2021,2,17),
    datetime.datetime(2021,2,18),
    datetime.datetime(2021,2,19),
    datetime.datetime(2021,2,20),
    datetime.datetime(2021,2,21),
    datetime.datetime(2021,2,22),
    datetime.datetime(2021,2,23),
    datetime.datetime(2021,2,24),
    datetime.datetime(2021,2,25),
    datetime.datetime(2021,3,4),
    datetime.datetime(2021,3,8),
    datetime.datetime(2021,3,10),
    datetime.datetime(2021,3,11),
    datetime.datetime(2021,3,12),
    datetime.datetime(2021,3,13),
    datetime.datetime(2021,3,14),
    datetime.datetime(2021,8,9),
    datetime.datetime(2021,11,27),
    datetime.datetime(2022,9,3),
    datetime.datetime(2022,9,4)    
]
    
    
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
    stat_total['dust'] = list()
    s_var = ['Epoch', 'DU_MED_AMP', 'DU_NR_IMPACT', 
             'SN_THRESHOLD', 'SN_RMS_E', 'SN_MED_MAX_E', 
             'SN_MAX_E', 'SN_NR_EVENTS', 'CHANNEL_REF', 
             'SNAPSHOT_LEN', 'TDS_CONFIG_LABEL', 'INPUT_CONFIG', 
             'SURVEY_MODE', 'SAMPLING_RATE']
    for d in np.arange(0, nday):
        tday = datetime.timedelta(days=int(d))
        dd = t0 + tday
        if dd in bad_days:
            stat_total['Epoch'].append(dd)
            stat_total['records'].append(0)
            stat_total['dust'].append(-1)
            continue
        
        date = [dd.year, dd.month, dd.day]
        s = tdscdf.load_data(date, product='surv-stat')
        if not s:
            continue
        stat_total['Epoch'].append(dd)
        stat_total['records'].append(len(s['Epoch']))  
        ind = np.asarray((s['DU_NR_IMPACT'][:] > 0).nonzero())[0]
        stat_total['dust'].append(len(ind))
        if stat:
            for z in s_var:                
                stat[z] = np.concatenate((stat[z], s[z][ind]))
        else:
            for z in s_var:
                stat[z] = s[z][ind]
    file = './dust/tds_stat_dust_gacr.pkl'
    with open(file, 'wb') as f:
        pickle.dump(stat, f)
        pickle.dump(stat_total, f)
    print(f'Done and stored at {file}')

def load_cdf(file, stime=None, ftime=None):
    
    cdf = cdflib.CDF(file)
    data = {}
    info = cdf.cdf_info()
    varnames = info.zVariables
    if stime:
         epoch = 'Epoch'
          # if stime.dtype == np.float64
    else:
        epoch = None
            
    for varname in varnames:
        data[varname] = cdf.varget(varname, epoch=epoch, starttime=stime, endtime=ftime)
        
    return data

    
