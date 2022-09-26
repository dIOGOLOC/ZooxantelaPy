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

#Bandpass frequency (Hz) - minimum and maximum
FILTER_DATA = [4,16]

NETWORK = 'ON'

OBS_NAME = ['OBS17']

CHANNEL = 'HHX'

# =========
# Constants
# =========

DTINY = np.finfo(0.0).tiny

ONESEC = datetime.timedelta(seconds=1)
HOUR12 = datetime.timedelta(hours=12)
ONEDAY = datetime.timedelta(days=1)

# =================
# Filtering by date
# =================
EVENT_HOUR = UTCDateTime('2019,12,08,04,50,00')
period_date = str(EVENT_HOUR.year)+'.'+"%03d" % EVENT_HOUR.julday 

# =========
# Functions
# =========


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
# time
stime = EVENT_HOUR-5
etime = EVENT_HOUR+120


obs_day_files = glob.glob(MSEED_FOLDER+'**/**/**/HHZ.D/*'+str(period_date))
st = Stream()
for file in obs_day_files:
    if any(x in file for x in OBS_NAME):
        st_u = read(file)[0]
        st_u.trim(starttime=stime, endtime=etime)
        print(st_u.stats.network+'.'+st_u.stats.station+'..'+st_u.stats.channel)
        sta_coord = inv.get_coordinates(st_u.stats.network+'.'+st_u.stats.station+'..'+st_u.stats.channel)

        st_u.stats.coordinates = AttribDict({
                                            'latitude': sta_coord['latitude'],
                                            'elevation': sta_coord['elevation'],
                                            'longitude': sta_coord['longitude']})
        st.append(st_u)

st.remove_response(inventory=inv)
st.detrend('demean')
st.detrend('linear')
st.taper(max_percentage=0.05, type='cosine') 
st.filter('highpass', freq=1)

# Execute array_processing

kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=5.0, win_frac=.5,
        # frequency properties
        frqlow=3.0, frqhigh=5.0, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=stime, etime=etime
    )

out = array_processing(st, **kwargs)

# Plot
labels = ['rel.power', 'abs.power', 'baz', 'slow']

xlocator = mdates.AutoDateLocator()
fig = plt.figure()
for i, lab in enumerate(labels):
        ax = fig.add_subplot(4, 1, i + 1)
        ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                edgecolors='none', cmap=obspy_sequential)
        ax.set_ylabel(lab)
        ax.set_xlim(out[0, 0], out[-1, 0])
        ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

fig.suptitle('AGFA skyscraper blasting in Munich %s' % (
        stime.strftime('%Y-%m-%d'), ))
fig.autofmt_xdate()
fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
plt.show()

# Plot

cmap = obspy_sequential

# make output human readable, adjust backazimuth to values between 0 and 360
t, rel_power, abs_power, baz, slow = out.T
baz[baz < 0.0] += 360

# choose number of fractions in plot (desirably 360 degree/N is an integer!)
N = 36
N2 = 30
abins = np.arange(N + 1) * 360. / N
sbins = np.linspace(0, 3, N2 + 1)

# sum rel power in bins given by abins and sbins
hist, baz_edges, sl_edges = \
    np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)

# transform to radian
baz_edges = np.radians(baz_edges)

# add polar and colorbar axes
fig = plt.figure(figsize=(8, 8))
cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")

dh = abs(sl_edges[1] - sl_edges[0])
dw = abs(baz_edges[1] - baz_edges[0])

# circle through backazimuth
for i, row in enumerate(hist):
    bars = ax.bar((i * dw) * np.ones(N2),
                  height=dh * np.ones(N2),
                  width=dw, bottom=dh * np.arange(N2),
                  color=cmap(row / hist.max()))

ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
ax.set_xticklabels(['N', 'E', 'S', 'W'])

# set slowness limits
ax.set_ylim(0, 3)
[i.set_color('grey') for i in ax.get_yticklabels()]
ColorbarBase(cax, cmap=cmap,
             norm=Normalize(vmin=hist.min(), vmax=hist.max()))

plt.show()