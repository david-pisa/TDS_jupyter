import cdflib
from pathlib import Path
import os


def load_rpw(filetype, year, month, day, opt):
    filetype = filetype.lower()

    if filetype == 'tswf' or filetype == 'rswf':
        data_folder = Path('Downloads')
        for names in os.listdir(data_folder / ('%04d' % year) / ('%02d' % month)):
            if ('solo_L2_rpw-tds-surv-%s-e-cdag_%04d%02d%02d_V' % (filetype, year, month, day)) in names:
                fname = names
        cdf = cdflib.CDF(data_folder / ('%04d' % year) / ('%02d' % month) / fname)

        data = {}

        if opt=='nnet':
            for varname in ['Epoch', 'SAMPLING_RATE', 'CHANNEL_ON', 'SAMPS_PER_CH', 'WAVEFORM_DATA', 'TDS_CONFIG_LABEL']:
                data[varname] = cdf.varget(varname)
            return data

        (why, varnames) = cdf._get_varnames()
        for varname in varnames:
            data[varname] = cdf.varget(varname)
        return data

    if filetype == 'stat' and opt == 'L1':
        data_folder = Path('Z:/rpw/L1')
        try:
            for names in os.listdir(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day)):
                if ('solo_L1_rpw-tds-surv-%s-cdag_%04d%02d%02d_V' % (filetype, year, month, day)) in names:
                    fname = names
            cdf = cdflib.CDF(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day) / fname)
        except:
            print('Dir not found')
            return -1

        data = {}

        (why, varnames) = cdf._get_varnames()
        for varname in varnames:
            data[varname] = cdf.varget(varname)
        return data

    if filetype == 'stat' and opt == 'L2':
        data_folder = Path('Z:/rpw/L2/tds_stat')
        try:
            for names in os.listdir(data_folder / ('%04d' % year) / ('%02d' % month)):
                if ('solo_L2_rpw-tds-surv-%s-cdag_%04d%02d%02d_V' % (filetype, year, month, day)) in names:
                    fname = names
            cdf = cdflib.CDF(data_folder / ('%04d' % year) / ('%02d' % month) / fname)
        except:
            print('Dir not found')
            return -1

        data = {}

        (why, varnames) = cdf._get_varnames()
        for varname in varnames:
            data[varname] = cdf.varget(varname)
        return data

    if filetype == 'hk' and opt == 'das':
        data_folder = Path('Z:/rpw/HK')
        try:
            for names in os.listdir(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day)):
                if ('solo_HK_rpw-das_%04d%02d%02d_V' % (year, month, day)) in names:
                    fname = names
            cdf = cdflib.CDF(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day) / fname)
        except:
            print('Dir not found')
            return -1

        data = {}

        (why, varnames) = cdf._get_varnames()
        for varname in varnames:
            data[varname] = cdf.varget(varname)
        return data

    if filetype == 'hk' and opt == 'das-stat':
        data_folder = Path('Z:/rpw/HK')
        try:
            for names in os.listdir(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day)):
                if ('solo_HK_rpw-das-statistics_%04d%02d%02d_V' % (year, month, day)) in names:
                    fname = names
            cdf = cdflib.CDF(data_folder / ('%04d' % year) / ('%02d' % month) / ('%02d' % day) / fname)
        except:
            print('Dir not found')
            return -1

        data = {}

        (why, varnames) = cdf._get_varnames()
        for varname in varnames:
            data[varname] = cdf.varget(varname)
        return data