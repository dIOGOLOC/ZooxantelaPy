#!/usr/bin/env python
# coding: utf-8

import time
from tqdm import tqdm
from multiprocessing import Pool

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, MinuteLocator, SecondLocator, DateFormatter
from matplotlib.patches import Ellipse
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, CheckButtons

import obspy as op
from obspy import read,read_inventory, UTCDateTime, Stream, Trace
from obspy.signal.array_analysis import array_processing
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.io.xseed import Parser
from obspy.signal.cross_correlation import correlate
from obspy.signal.filter import bandpass,lowpass
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import prev_pow_2
from obspy.signal.tf_misfit import cwt
import pywt
import scipy.stats as stats
from pyrocko import obspy_compat
obspy_compat.plant()

import json
import glob
import os
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from itertools import combinations
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from scipy.signal import spectrogram, detrend, resample,savgol_filter
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.linalg import norm
from sklearn.metrics import jaccard_score
from obspy.signal.util import next_pow_2

import random
import collections
from copy import copy
import datetime
import matplotlib.dates as mdates
from itertools import compress
from PIL import Image
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature


from obspy.signal.trigger import classic_sta_lta, trigger_onset, coincidence_trigger,recursive_sta_lta,plot_trigger


# ========================
# Input and output folders
# ========================

MSEED_FOLDER = '/home/diogoloc/dados_posdoc/ON_MAR/obs_data_MSEED/'

EARTHQUAKE_FINDER_OUTPUT = '/home/diogoloc/dados_posdoc/ON_MAR/EARTHQUAKE_FINDER_NETWORK_OUTPUT/FIGURAS/'

ASDF_FILES = '/home/diogoloc/dados_posdoc/ON_MAR/EARTHQUAKE_FINDER_NETWORK_OUTPUT/ASDF_FILES/'

STATIONXML_DIR = '/home/diogoloc/dados_posdoc/ON_MAR/XML_ON_OBS_CC/'

BINARY_FILES = '/home/diogoloc/dados_posdoc/ON_MAR/EARTHQUAKE_FINDER_NETWORK_OUTPUT/BINARY_FILES/'

# ==========
# Parameters
# ==========

EVENT_TIME = '2019,11,19,23,16,00'

FIRSTDAY = '2019,11,19,22,45,00'
LASTDAY = '2019,11,19,23,45,00'

NETWORK = 'ON'

STATION = 'OBS22'

CHANNEL = 'HHZ'

# Maximum frequency for plot
fmax = 49.

# Minimum value for spectrograms (in dB)
dBmin = -180

# Maximum value for spectrograms (in dB)
dBmax = -60

# Start time for spectrogram (in any format that datetime understands)
tstart = None
            
# End time for spectrogram
tend = None

# Window length for long-period spectrogram (in seconds)
plot_ratio = 0.3

# Window length for high-frequency spectrogram (in seconds)
winlen_HF = 4

# Window length for long-period spectrogram (in seconds)
winlen_LF = 60.

# Tradeoff between time and frequency resolution in CWT (Lower numbers: better time resolution/Higher numbers: better freq resolution)
w0 = 10

# Percentage of overlap (between 0 and 1) for spectrogram computation, if kind=='spec'
overlap=0.5

# Size of the produced figure in Inches. Default is 16x9, which is good for high resolution screen display.
figsize = (16, 9)

dpi = 300

# Unit of input data. Options: ACC, VEL, DIS. Plot is in acceleration.
unit = 'ACC'

# Calculate spectrogram (spec) or continuous wavelet transfort (cwt, much slower)? Default: 'spec')
kind = 'spec' 

# =========
# Constants
# =========

ONESEC = datetime.timedelta(seconds=1)
HOUR12 = datetime.timedelta(hours=12)
ONEDAY = datetime.timedelta(days=1)

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

# ------------------------------------------------------------------------------
def format_axes(ax_cb: plt.Axes,
                ax_psd_HF: plt.Axes, ax_psd_LF: plt.Axes, ax_seis_HF: plt.Axes, ax_seis_LF: plt.Axes,
                ax_spec_HF: plt.Axes, ax_spec_LF: plt.Axes, fmax_HF: plt.Axes,
                fmax_LF: float, fmin_HF: float, fmin_LF: float, st_HF: Stream, st_LF: Stream,
                tstart: datetime, tend: datetime, dBmax: float
                ):
    ax_seis_LF.set_ylabel('%s \n[µm/s²] < 1 Hz' % st_LF[0].get_id(), color='grey')
    ax_seis_HF.set_ylabel('%s \n[µm/s²] > 1 Hz' % st_HF[0].get_id(),color='k')

    ax_seis_LF.tick_params('y', colors='grey')
    ax_seis_HF.tick_params('y', colors='k')
    ax_spec_HF.set_ylim(fmin_HF, fmax_HF)
    ax_spec_HF.set_ylabel('Frequência [Hz]', fontsize=12)
    ax_spec_LF.set_ylabel('Frequência [Hz]', fontsize=12)

    for ax_psd in [ax_psd_HF, ax_psd_LF]:
        ax_psd.yaxis.set_label_position("right")
        ax_psd.yaxis.set_ticks_position("right")
    ax_psd_LF.set_ylim(fmin_LF, fmax_LF)

    ax_psd_LF.set_xlabel('PSD [dB]')
    ax_psd_LF.set_ylabel('Frequência [Hz]', fontsize=12, rotation=270., va='bottom')
    ax_psd_HF.set_ylabel('Frequência [Hz]', fontsize=12, rotation=270., va='bottom')
    ax_psd_HF.yaxis.set_major_locator(MultipleLocator(10))
    ax_psd_HF.yaxis.set_minor_locator(MultipleLocator(1))
    ax_psd_LF.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_psd_LF.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_psd_LF.xaxis.set_major_locator(MultipleLocator(50))
    ax_psd_LF.xaxis.set_minor_locator(MultipleLocator(10))

    ax_psd_HF.set_xticks([])
    # make unnecessary labels disappear
    for ax in [ax_spec_HF, ax_seis_LF]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel('')

    for ax in [ax_spec_HF, ax_spec_LF, ax_seis_LF]:
        ax.set_xlim(tstart, tend)

    # Axis with colorbar
    mappable = ax_spec_HF.collections[0]
    plt.colorbar(mappable=mappable, cax=ax_cb)
    ax_cb.set_ylabel('PSD [dB]', rotation=270., va='bottom')

    locator = mdates.AutoDateLocator(minticks=9, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_spec_LF.xaxis.set_major_locator(locator)
    ax_spec_LF.xaxis.set_major_formatter(formatter)
    ax_spec_LF.set_xticks(ax_spec_LF.get_xticks()[:-1])
    ax_spec_HF.xaxis.offsetText.get_text()
    ax_spec_HF.xaxis.offsetText.set_text('')
    ax_spec_HF.xaxis.offsetText.set_visible(False)
    mins5 = mdates.MinuteLocator(interval=5)
    mins1 = mdates.MinuteLocator(interval=1)
    # format the ticks
    for ax in (ax_spec_HF, ax_spec_LF):
        ax.xaxis.set_major_locator(mins5)
        ax.xaxis.set_minor_locator(mins1)


# =================
# Filtering by date
# =================

fday = UTCDateTime(FIRSTDAY)
lday = UTCDateTime(LASTDAY)
eday = UTCDateTime(EVENT_TIME)

tstart = fday.datetime
tend = lday.datetime

INTERVAL_PERIOD = [UTCDateTime(x.astype(str)) for x in np.arange(fday.datetime,lday.datetime+HOUR12,HOUR12)]
INTERVAL_PERIOD_DATE = [str(x.year)+'.'+"%03d" % x.julday for x in INTERVAL_PERIOD]
# ============
# Main program
# ============

print('=======================')
print('Loading inventory files')
print('=======================')
print('\n')

xml_files = sorted(glob.glob(STATIONXML_DIR+'ON.OBS*'))
inv = read_inventory(xml_files[0])
for xml_file in xml_files[1:]:
	inv.extend(read_inventory(xml_file))
        
#----------------------------------------------------------------------------

print('===============================')
print('Scanning name of miniseed files')
print('===============================')
print('\n')

# initializing list of stations by scanning name of miniseed files

files = filelist(basedir=MSEED_FOLDER+'*/'+NETWORK+'/'+STATION+'/'+CHANNEL+'.D/',interval_period_date=INTERVAL_PERIOD_DATE)

st = Stream()
if len(files) > 1: 
    for file in files:
        st += read(file)
    st.merge(method=1, fill_value='interpolate')
else: 
    st += read(files[0])

samp_rate_original = st[0].stats.sampling_rate

if samp_rate_original > fmax * 10 and samp_rate_original % 5 == 0.:
    st.decimate(5)

while st[0].stats.sampling_rate > 4. * fmax:
    st.decimate(2)

# Retrieving information from INVENTORY:
for tr in st:
    coords = inv.get_coordinates(tr.get_id())
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.elevation = coords['elevation']

# Trimming the data
st.trim(starttime=fday- 120., endtime=lday+ 120.)

# Removing response
st.remove_response(inventory=inv, output=unit)

# The computation of the LF spectrograms with long time windows or even CWT
# can be REALLY slow, thus, decimate it to anything larger 2.5 Hz
st_LF = st.copy()
st_LF.filter('lowpass', freq=0.95, corners=16)

# For LF, 2 Hz is enough
st_LF.interpolate(sampling_rate=2.,method='nearest')

st_HF = st.copy()

winlen_LF = int(winlen_LF * st_LF[0].stats.sampling_rate)
winlen_HF = int(winlen_HF * st_HF[0].stats.sampling_rate)

if st_LF[0].stats.sampling_rate > 5.0:
    dec_fac = int(st_LF[0].stats.sampling_rate / 5.0)
    st_LF.decimate(dec_fac)

for st in [st_HF, st_LF]:
    st.detrend()
    st.filter('highpass', freq=1. / winlen_LF)
    st.trim(starttime=fday,endtime=lday)

while st_LF[0].stats.sampling_rate > 4.:
    st_LF.decimate(2)

fmin_LF = 2. / winlen_LF
fmax_LF = 1.0

fmin_HF = 1.0
fmax_HF = fmax

st_LF.filter('highpass', freq=fmin_LF * 0.9,zerophase=True, corners=6)
st_HF.filter('bandpass', freqmin=fmin_HF * 0.5, freqmax=fmax_HF,zerophase=True, corners=6)

fig = plt.figure(figsize=figsize)

# [left bottom width height]
h_spec_total = 0.7
h_base = 0.13

h_spec_LF = h_spec_total * plot_ratio
h_spec_HF = h_spec_total - h_spec_LF
h_seis = 0.15  # 0.2
w_base = 0.08
w_spec = 0.77
w_psd = 0.1
ax_seis_LF = fig.add_axes([w_base, h_base + h_spec_total, w_spec, h_seis], label='seismogram LF')
ax_seis_HF = ax_seis_LF.twinx()
ax_spec_LF = fig.add_axes([w_base, h_base, w_spec, h_spec_LF],sharex=ax_seis_LF, label='spectrogram LF')
ax_spec_HF = fig.add_axes([w_base, h_base + h_spec_LF, w_spec, h_spec_HF],sharex=ax_seis_LF,label='spectrogram HF')
ax_psd_LF = fig.add_axes([w_base + w_spec, h_base, w_psd, h_spec_LF], sharey=ax_spec_LF,label='PSD LF')
ax_psd_HF = fig.add_axes([w_base + w_spec, h_base + h_spec_LF,0.1, h_spec_HF],sharey=ax_spec_HF,label='PSD HF')

# Colorbar axis
ax_cb = fig.add_axes([w_base + w_spec + w_psd / 1.5,h_base + h_spec_HF + h_spec_LF + h_seis * 0.1, w_psd * 0.1, h_seis * 0.8], label='colorbar')


for tr in st_LF:
    t_LF = np.arange(UTCDateTime(tr.stats.starttime).datetime,UTCDateTime(tr.stats.endtime + tr.stats.delta).datetime,datetime.timedelta(seconds=tr.stats.delta))
    ax_seis_LF.plot(t_LF, tr.data * 1e6,'grey', lw=1,alpha=0.7, label='< 1 Hz')
    ax_seis_LF.legend(loc='lower left', fontsize=9, edgecolor='grey', labelcolor='grey')

for tr in st_HF:
    t_HF = np.arange(UTCDateTime(tr.stats.starttime).datetime,UTCDateTime(tr.stats.endtime + tr.stats.delta).datetime,datetime.timedelta(seconds=tr.stats.delta))
    ax_seis_HF.plot(t_HF, tr.data * 1e6,'k', lw=0.5,alpha=0.7, label='> 1 Hz')
    ax_seis_HF.legend(loc='lower right', fontsize=9, edgecolor='k', labelcolor='k')
    ax_seis_HF.annotate(eday.strftime('%H:%M:%S'), (eday, tr.data.max()),xytext=(0.5, 0.2), textcoords='axes fraction',
            arrowprops=dict(facecolor='r', shrink=0.005,alpha=0.75),fontsize=12,horizontalalignment='right', verticalalignment='top')

    
for st, ax_spec, ax_psd, flim, winlen in zip([st_LF, st_HF],[ax_spec_LF, ax_spec_HF],[ax_psd_LF, ax_psd_HF],[(fmin_LF, fmax_LF),(fmin_HF, fmax_HF)], [winlen_LF, winlen_HF]):
    for tr in st:
       
        p, f, t = mlab.specgram(tr.data, NFFT=winlen,Fs=tr.stats.sampling_rate,noverlap=int(winlen * overlap),pad_to=next_pow_2(winlen) * 4)

        t = np.array([UTCDateTime(i + tr.stats.starttime.timestamp) for i in t])
    
        delta = t[1] - t[0]
        t = np.arange(t[0].datetime,(t[-1] + delta * 0.9).datetime,datetime.timedelta(seconds=delta))

        bol = np.array((f >= flim[0] * 0.9, f <= flim[1])).all(axis=0)
            
        vmin = np.percentile(10 * np.log10(p[bol, :]), q=1, axis=None)
        vmax = np.percentile(10 * np.log10(p[bol, :]), q=90, axis=None)
        
        ax_spec.pcolormesh(t, f[bol], 10 * np.log10(p[bol, :]),vmin=dBmin,vmax=dBmax, shading='nearest',cmap='inferno')

        median = np.percentile(p[bol, :], axis=1, q=50)
        perc_95 = np.percentile(p[bol, :], axis=1, q=95)
        perc_01 = np.percentile(p[bol, :], axis=1, q=1)
        ax_psd.plot(10 * np.log10(median), f[bol],color='k',lw=0.3,label='Med.')
        ax_psd.plot(10 * np.log10(perc_95), f[bol],color='r',lw=0.3,label='95%')
        ax_psd.plot(10 * np.log10(perc_01), f[bol],color='darkgrey',lw=0.3,label='1%')
        nhnm = op.signal.spectral_estimation.get_nhnm()
        nlnm = op.signal.spectral_estimation.get_nlnm()
        ax_psd.fill_betweenx(y=1. / nlnm[0], x1=-300 * np.ones_like(nlnm[1]), x2=nlnm[1], color='lightgray')
        ax_psd.fill_betweenx(y=1. / nhnm[0], x2=300 * np.ones_like(nhnm[1]), x1=nhnm[1], color='lightgray')
        ax_psd.set_xlim(dBmin, dBmax)

        if ax_psd == ax_psd_HF:
            ax_psd.legend(fontsize='small',bbox_to_anchor=(0.45,0.975) , borderaxespad=0.,edgecolor='None',facecolor='None')
        
format_axes(ax_cb, ax_psd_HF, ax_psd_LF, ax_seis_HF, ax_seis_LF,ax_spec_HF, ax_spec_LF, fmax_HF,fmax_LF, fmin_HF, fmin_LF, st_HF, st_LF,tstart, tend, dBmax=vmax)
os.makedirs(EARTHQUAKE_FINDER_OUTPUT,exist_ok=True)
plt.savefig(EARTHQUAKE_FINDER_OUTPUT+st_LF[0].get_id()+'.'+fday.strftime('.%Y.%m.%d.%H.%M.%S')+'.png')
