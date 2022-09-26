#!/usr/bin/env python
# coding: utf-8

import time
from tqdm import tqdm
from multiprocessing import Pool

import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, MinuteLocator, SecondLocator, DateFormatter
from matplotlib.patches import Ellipse
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle

import obspy as op
from obspy import read,read_inventory, UTCDateTime, Stream, Trace
from obspy.io.xseed import Parser
from obspy.signal.cross_correlation import correlate
from obspy.signal.filter import bandpass,lowpass
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import prev_pow_2
from obspy.signal.tf_misfit import cwt
from scipy.stats import moment,kurtosis,skew

import json
import glob
import os
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from itertools import combinations
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from scipy.signal import spectrogram, detrend, resample,savgol_filter
from scipy.linalg import norm
from sklearn.preprocessing import normalize as normalize_matrix

import random
import collections
from copy import copy
import datetime
import matplotlib.dates as mdates
from itertools import compress


import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature

from pyasdf import ASDFDataSet
from obspy.signal.trigger import classic_sta_lta, trigger_onset, coincidence_trigger,recursive_sta_lta

# ====================================================================================================
# Configuration file
# ====================================================================================================

MSEED_DIR = '/home/diogoloc/dados_posdoc/ON_MAR/obs_data_MSEED/'

STATIONXML_DIR = '/home/diogoloc/dados_posdoc/ON_MAR/XML_OBS/'

EARTHQUAKE_FINDER_OUTPUT = '/home/diogoloc/dados_posdoc/ON_MAR/EARTHQUAKE_FINDER_OUTPUT/DAYPLOT_FIGURAS/'

FIRSTDAY = '2019-12-01'
LASTDAY = '2019-12-15'

FILTER_DATA = [3,5]

NETWORK = 'ON'

STATION = 'OBS22'

CHANNEL = 'HHX'

# ========================
# Constants and parameters
# ========================

DTINY = np.finfo(0.0).tiny

ONESEC = datetime.timedelta(seconds=1)
ONEDAY = datetime.timedelta(days=1)


# ================
# MULTIPROCESSING
# ================
num_processes = 4

# =================
# Filtering by date
# =================

fday = UTCDateTime(FIRSTDAY)
lday = UTCDateTime(LASTDAY)
INTERVAL_PERIOD = [UTCDateTime(x.astype(str)) for x in np.arange(fday.datetime,lday.datetime+ONEDAY,ONEDAY)]
INTERVAL_PERIOD_DATE = [str(x.year)+'.'+"%03d" % x.julday for x in INTERVAL_PERIOD]

# =========
# Functions
# =========


def filelist(basedir,interval_period_date):
    """
    Returns the list of files in *basedir* whose are in the specified period
    """
    files = []
    files_list = glob.glob(basedir+'/*')
    for s in files_list:
        if any(day_s in s for day_s in interval_period_date):
            files.append(s)
            
    files = [i for i in files if CHANNEL in i]

    return sorted(files)

#-------------------------------------------------------------------------------

def get_stations_data(f):
    """
    Gets stations daily data from miniseed file and convert in ASDF

    @type f: paht of the minissed file (str)
    @rtype: list of L{StationDayData}
    """

    # splitting subdir/basename
    subdir, filename = os.path.split(f)

    # network, station name and station channel in basename,
    # e.g., ON.TIJ01..HHZ.D.2020.002

    network, name = filename.split('.')[0:2]
    sta_channel_id = filename.split('.D.')[0]
    channel = sta_channel_id.split('..')[-1]
    time_day = filename.split('.D.')[-1]
    year_day = time_day.split('.')[0]
    julday_day = time_day.split('.')[1]
    st = read(f)

    st.filter("bandpass", freqmin=3, freqmax=5)
    
    #-----------------------------------------------------
    daily_wind_output = EARTHQUAKE_FINDER_OUTPUT+NETWORK+'.'+STATION+'/'
    os.makedirs(daily_wind_output,exist_ok=True)
    outfile_name = daily_wind_output+NETWORK+'_'+STATION+'_'+CHANNEL+'_'+year_day+'_'+julday_day+'.png'
	#------------------------------------------------------------------
    st.plot(outfile=outfile_name,type="dayplot",title=sta_channel_id+'('+st[0].stats.starttime.strftime("%d/%m/%Y")+')', interval=60, right_vertical_labels=True,vertical_scaling_range=1e4, one_tick_per_line=True,color='k', show_y_UTC_label=True)

# ============
# Main program
# ============

print('===============================')
print('Scanning name of miniseed files')
print('===============================')
print('\n')

# initializing list of stations by scanning name of miniseed files

files = filelist(basedir=MSEED_DIR+'*/'+NETWORK+'/'+STATION+'/'+CHANNEL+'.D/',interval_period_date=INTERVAL_PERIOD_DATE)

print('Total of miniseed files = '+str(len(files)))
print('\n')

print('============================================================')
print('Opening miniseed files, preprocessing and converting to ASDF')
print('============================================================')
print('\n')

start_time = time.time()

with Pool(processes=num_processes) as p:
	max_ = len(files)
	with tqdm(total=max_) as pbar:
		for i, _ in enumerate(p.imap_unordered(get_stations_data, files)):
			pbar.update()

print("--- %.2f execution time (min) ---" % ((time.time() - start_time)/60))
print('\n')
