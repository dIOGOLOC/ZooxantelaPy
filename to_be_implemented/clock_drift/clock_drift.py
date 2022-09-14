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
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.transforms import offset_copy

import obspy as op
from obspy import read,read_inventory, UTCDateTime, Stream, Trace
from obspy.io.xseed import Parser
from obspy.signal.filter import bandpass
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import prev_pow_2
from obspy.signal.cross_correlation import correlate as obscorr
from obspy.signal.cross_correlation import xcorr_max
from obspy.core.util import AttribDict

import glob
import os
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from itertools import combinations,product,compress
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import pyarrow.feather as feather
from scipy.signal import spectrogram, detrend, resample,savgol_filter,decimate
from scipy.linalg import norm
import random
import collections
from copy import copy
import datetime

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,HuberRegressor,TheilSenRegressor
from sklearn.metrics import mean_squared_error

from pyasdf import ASDFDataSet

# ==================
# Configuration file
# ==================

# Folders input

MSEED_DIR_OBS = '/home/diogoloc/dados_posdoc/ON_MAR/obs_data_MSEED/'

MSEED_DIR_STA = '/home/diogoloc/dados_posdoc/ON_MAR/data/'

# -------------------------------

# Shapefile  boundary states input

BOUNDARY_STATES_SHP = '/home/diogoloc/dados_posdoc/SIG_Dados/Brasil_RSBR/Shapefile/estados_brasil/UFEBRASIL.shp'

# -------------------------------

# Stations and OBSs information

OBS_LST = ['OBS17','OBS18','OBS20','OBS22']

STATIONS_LST = ['ABR01','DUB01','MAN01','OBS20','OBS22','TER01','ALF01','GDU01','NAN01','TIJ01','CAJ01','GUA01','OBS17','PET01','TRI01','CAM01','JAC01','OBS18','RIB01','VAS01','CMC01','MAJ01','SLP01','PARB','CNLB','BSFB']
STATIONS_LST = sorted(STATIONS_LST)

STATIONXML_DIR = '/home/diogoloc/dados_posdoc/ON_MAR/XML_ON_OBS_CC/'

CHANNEL_LST = ['HHZ.D','HHN.D','HHE.D','HH1.D','HH2.D']

# -------------------------------

# Folders output

CLOCK_DRIFT_OUTPUT = '/home/diogoloc/dados_posdoc/ON_MAR/CLOCK_DRIFT_OUTPUT/FIGURAS/'

ASDF_FILES = '/home/diogoloc/dados_posdoc/ON_MAR/ORIENTATION_OUTPUT/ASDF_FILES/'

FEATHER_FILES = '/home/diogoloc/dados_posdoc/ON_MAR/CLOCK_DRIFT_OUTPUT/FEATHER_FILES/'

# -------------------------------
#create figures?
VERBOSE = False

# Input parameters

FIRSTDAY = '2019-08-01'
LASTDAY = '2020-06-01'

#Each hour-long seismogram is amplitude clipped at twice its standard deviation of that hour-long time window.
CLIP_FACTOR = 6

MIN_WINDOWS = 6

WINDOW_LENGTH = 7200

#max time window (s) for cross-correlation
SHIFT_LEN = 900

PERIOD_BANDS = [[10,20], [20, 35],[30,50]]
# (these bands focus on periods 15,30 and 75 seconds)
PERIOD_BANDS2 = [[5,10],[10,20], [20, 35],[30,50]]

# default parameters to define the signal and noise windows used to
# estimate the SNR:
# - the signal window is defined according to a min and a max velocity as:
#   dist/vmax < t < dist/vmin
# - the noise window has a fixed size and starts after a fixed trailing
#   time from the end of the signal window
SIGNAL_WINDOW_VMIN = 2.0
SIGNAL_WINDOW_VMAX = 4.0
SIGNAL2NOISE_TRAIL = 500.0
NOISE_WINDOW_SIZE = 500.0

#Returns pairs and spectral SNR array whose spectral SNRs are all >= minspectSNR
minspectSNR = 1

#RESAMPLING
NEW_SAMPLING_RATE = 1

# -------------------------------

# Constants and parameters

ONESEC = datetime.timedelta(seconds=1)
ONEDAY = datetime.timedelta(days=1)
TENDAY = datetime.timedelta(days=10)

# -------------------------------

# MULTIPROCESSING

num_processes = 8

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
#-------------------------------------------------------------------------------

# Calculating signal-to-noise ratio
def obscorr_window(data1,time_data1,data2,time_data2,dist,vmin,vmax):
    """
    Calculate the CrossCorrelation according to the distance between two stations.

    The signal window is defined by *vmin* and *vmax*:
        dist/*vmax* < t < dist/*vmin*

    @type data1: numpy array
    @type time_data1: numpy array
    @type data2: numpy array
    @type time_data2: numpy array
    @type dist: float
    @type vmin: float
    @type vmax: float
    """

    # signal window
    tmin_signal = dist/vmax
    tmax_signal = dist/vmin

    signal_window1 = (time_data1 >= tmin_signal) & (time_data1 <= tmax_signal)
    signal_window2 = (time_data2 >= tmin_signal) & (time_data2 <= tmax_signal)

    trace1 = data1[signal_window1]
    trace2 = data2[signal_window2]

    cc = obscorr(trace1,trace2,np.max([len(trace1),len(trace2)]))
    shift, coefficient = xcorr_max(cc)

    return shift/NEW_SAMPLING_RATE, coefficient

#-------------------------------------------------------------------------------

def Normalize(data):
	"""
	z(i)=2*(x(i)−min(x)/max(x)-min(x))−1

	where x=(x1,...,xn) and z(i) is now your ith normalized data between -1 and 1.

	@type data: list
	"""

	normalized_data = [2*(i-data.min()/(data.max()-data.min()))-1 for i in data]

	return normalized_data


# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def Calculating_clock_drift_func(ipair):
        '''
        Calculating clock drift from cross-correlation data
        @type input: name of the stations pair (str)
        '''

        pair_sta_1 = ipair.split('_')[0].split('..')[0]
        pair_sta_2 = ipair.split('_')[1].split('..')[0]

        #Check if file exists
        file_feather_name = FEATHER_FILES+pair_sta_1+'.'+pair_sta_2+'_clock_drift_data.feather'
        if os.path.isfile(file_feather_name):
            pass

        else:

            # ---------------------
            # Separating by channel
            # ---------------------

            HHE_HHE_lst = []
            HHN_HHN_lst = []
            HHZ_HHZ_lst = []

            HHE_HHN_lst = []
            HHN_HHE_lst = []

            HHE_HHZ_lst = []
            HHZ_HHE_lst = []

            HHN_HHZ_lst = []
            HHZ_HHN_lst = []

            for i in crosscorr_pairs_obs:

                if pair_sta_1 and pair_sta_2 in i:

                    # splitting subdir/basename
                    subdir, filename = os.path.split(i)
                    nameslst = filename.split("_20")[0]

                    name_pair1 = nameslst.split('_')[-2]
                    name_pair2 = nameslst.split('_')[-1]

                    name1 = nameslst.split('_')[-2].split('..')[0]
                    name2 = nameslst.split('_')[-1].split('..')[0]

                    channel_sta1 = nameslst.split('_')[-2].split('..')[1]
                    channel_sta2 = nameslst.split('_')[-1].split('..')[1]

                    if pair_sta_1 == name1 and pair_sta_2 == name2:

                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHE' and channel_sta2 == 'HHE' or channel_sta1 == 'HH2' and channel_sta2 == 'HH2' or channel_sta1 == 'HH2' and channel_sta2 == 'HHE' or  channel_sta1 == 'HHE' and channel_sta2 == 'HH2':
                            HHE_HHE_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHN' and channel_sta2 == 'HHN' or channel_sta1 == 'HH1' and channel_sta2 == 'HH1' or channel_sta1 == 'HHN' and channel_sta2 == 'HH1' or channel_sta1 == 'HH1' and channel_sta2 == 'HHN':
                            HHN_HHN_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHZ' and channel_sta2 == 'HHZ':
                            HHZ_HHZ_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHE' and channel_sta2 == 'HHN' or channel_sta1 == 'HH2' and channel_sta2 == 'HHN' or channel_sta1 == 'HHE' and channel_sta2 == 'HH1' or channel_sta1 == 'HH2' and channel_sta2 == 'HH1':
                            HHE_HHN_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHN' and channel_sta2 == 'HHE' or channel_sta1 == 'HH1' and channel_sta2 == 'HHE' or channel_sta1 == 'HHN' and channel_sta2 == 'HH2' or channel_sta1 == 'HH1' and channel_sta2 == 'HH2':
                            HHN_HHE_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHE' and channel_sta2 == 'HHZ' or channel_sta1 == 'HH2' and channel_sta2 == 'HHZ':
                            HHE_HHZ_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHZ' and channel_sta2 == 'HHE' or channel_sta1 == 'HHZ' and channel_sta2 == 'HH2':
                            HHZ_HHE_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHN' and channel_sta2 == 'HHZ' or channel_sta1 == 'HH1' and channel_sta2 == 'HHZ':
                            HHN_HHZ_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------
                        if channel_sta1 == 'HHZ' and channel_sta2 == 'HHN' or channel_sta1 == 'HHZ' and channel_sta2 == 'HH1':
                            HHZ_HHN_lst.append(i)
                        # ------------------------------------------------------------------------------------------------------

            CHANNEL_fig_lst = [HHE_HHE_lst,HHE_HHN_lst,HHE_HHZ_lst,HHN_HHN_lst,HHN_HHE_lst,HHN_HHZ_lst,HHZ_HHE_lst,HHZ_HHN_lst,HHZ_HHZ_lst]
            chan_lst = ['HHE-HHE','HHE-HHN','HHE-HHZ','HHN-HHN','HHN-HHE','HHN-HHZ','HHZ-HHE','HHZ-HHN','HHZ-HHZ']

            # ------------------------------------------------------------------------------------------------------
            # Starting the value list and the columns_headers:
            columns_headers = ['sta_1','sta_2']

            data_drift_lst = [pair_sta_1,pair_sta_2]

            # ---------------------
            # Calculating the drift
            # ---------------------

            for idch, i in enumerate(tqdm(CHANNEL_fig_lst,desc='Drift:'+pair_sta_1+'-'+pair_sta_2)):
                if len(i) > 1:

                    # ------------
                    # Reading data
                    # ------------

                    crosscorr_pair_date_filename = [filename.split('/')[-1] for filename in i]
                    crosscorr_pair_date = [datetime.datetime.strptime(filename.split('/')[-1].split('_')[-2]+'.'+filename.split('/')[-1].split('_')[-1].split('.')[0], '%Y.%j') for filename in crosscorr_pair_date_filename]

                    sta1_sta2_asdf_file = [ASDFDataSet(k, mode='r') for k in i]

                    name_sta1 = [file.auxiliary_data.CrossCorrelation.list()[0] for file in sta1_sta2_asdf_file]
                    name_sta2 = [file.auxiliary_data.CrossCorrelation.list()[1] for file in sta1_sta2_asdf_file]

                    dist_pair = [sta1_sta2_asdf_file[id].auxiliary_data.CrossCorrelation[name_sta1[id]][name_sta2[id]].parameters['dist'] for id,jd in enumerate(name_sta1)]
                    loc_sta1 = [sta1_sta2_asdf_file[id].auxiliary_data.CrossCorrelation[name_sta1[id]][name_sta2[id]].parameters['sta1_loc'] for id,jd in enumerate(name_sta1)]
                    loc_sta2 = [sta1_sta2_asdf_file[id].auxiliary_data.CrossCorrelation[name_sta1[id]][name_sta2[id]].parameters['sta2_loc'] for id,jd in enumerate(name_sta1)]

                    causal_time = [sta1_sta2_asdf_file[id].auxiliary_data.CrossCorrelation[name_sta1[id]][name_sta2[id]].parameters['crosscorr_daily_causal_time'][::]  for id,jd in enumerate(name_sta1)]
                    acausal_time = [sta1_sta2_asdf_file[id].auxiliary_data.CrossCorrelation[name_sta2[id]][name_sta1[id]].parameters['crosscorr_daily_acausal_time'] for id,jd in enumerate(name_sta1)]

                    # ------------
                    # Stacked data
                    # ------------

                    causal_lst = [jd.auxiliary_data.CrossCorrelation[name_sta1[id]][name_sta2[id]].data[::] for id,jd in enumerate(sta1_sta2_asdf_file)]
                    acausal_lst = [jd.auxiliary_data.CrossCorrelation[name_sta2[id]][name_sta1[id]].data[::] for id,jd in enumerate(sta1_sta2_asdf_file)]

                    # ----------------------------------------------------------------------------------------------------
                    for iband, per_bands in enumerate(PERIOD_BANDS):

                        column_date_n = chan_lst[idch]+' date ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'
                        column_coefficient_n = chan_lst[idch]+' coefficient ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'
                        column_shift_n = chan_lst[idch]+' shift ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'

                        columns_headers.extend([column_date_n,column_coefficient_n,column_shift_n])

                        # ----------------------------------------------------------------------------------------------------

                        date_to_plot_clock = []
                        data_to_plot_coefficient_clock_drift = []
                        data_to_plot_shift_clock_drift = []

                        for k in range(len(causal_lst)):
                            if len(glob.glob(ASDF_FILES+'CROSS_CORR_10_DAYS_STACKED_FILES/'+name_sta1[k]+'.'+name_sta2[k]+'/*')) > 0:


                                data_acausal_causal = np.array(acausal_lst[k] + causal_lst[k])
                                data_normalized = np.array(Normalize(data_acausal_causal))
                                time_acausal_causal = np.array(acausal_time[k] + causal_time[k])

                                dist_pair_norm = dist_pair[k]

                                # --------------------------------------------------------
                                # Collecting daily list of 10-day stack cross-correlations
                                # --------------------------------------------------------

                                sta1_sta2_asdf_file_10_day = ASDFDataSet(glob.glob(ASDF_FILES+'CROSS_CORR_10_DAYS_STACKED_FILES/'+name_sta1[k]+'.'+name_sta2[k]+'/*')[0], mode='r')
                                    
                                stacked_10_day_data = sta1_sta2_asdf_file_10_day.auxiliary_data.CrossCorrelationStacked[name_sta2[k]][name_sta1[k]].data[::]+sta1_sta2_asdf_file_10_day.auxiliary_data.CrossCorrelationStacked[name_sta1[k]][name_sta2[k]].data[::]
                                stacked_10_day_time = sta1_sta2_asdf_file_10_day.auxiliary_data.CrossCorrelationStacked[name_sta2[k]][name_sta1[k]].parameters['crosscorr_stack_time'] +sta1_sta2_asdf_file_10_day.auxiliary_data.CrossCorrelationStacked[name_sta1[k]][name_sta2[k]].parameters['crosscorr_stack_time']
                                    
                                stacked_10_day_data_normalized = np.array(Normalize(stacked_10_day_data))

                                # --------------------------------------------------------

                                stacked_10_day_data_normalized_band = bandpass(stacked_10_day_data_normalized, 1.0/per_bands[1], 1.0/per_bands[0], NEW_SAMPLING_RATE, zerophase=True)
                                data_normalized_band = bandpass(data_normalized, 1.0/per_bands[1], 1.0/per_bands[0], NEW_SAMPLING_RATE, zerophase=True)

                                shift_clock_drift, coefficient_clock_drift = obscorr_window(stacked_10_day_data_normalized_band,stacked_10_day_time,data_normalized_band,time_acausal_causal,dist_pair_norm,SIGNAL_WINDOW_VMIN,SIGNAL_WINDOW_VMAX)

                                date_to_plot_clock.append(crosscorr_pair_date[k])
                                data_to_plot_coefficient_clock_drift.append(coefficient_clock_drift)
                                data_to_plot_shift_clock_drift.append(shift_clock_drift)
                            
                            else:
                                date_to_plot_clock.append([])
                                data_to_plot_coefficient_clock_drift.append([])
                                data_to_plot_shift_clock_drift.append([])
                        # --------------------------------------------------------------------------------------

                        data_drift_lst.append(date_to_plot_clock)
                        data_drift_lst.append(data_to_plot_coefficient_clock_drift)
                        data_drift_lst.append(data_to_plot_shift_clock_drift)
                            
                else:

                    for iband, per_bands in enumerate(PERIOD_BANDS):

                        column_date_n = chan_lst[idch]+' date ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'
                        column_coefficient_n = chan_lst[idch]+' coefficient ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'
                        column_shift_n = chan_lst[idch]+' shift ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'

                        columns_headers.extend([column_date_n,column_coefficient_n,column_shift_n])

                        data_drift_lst.append([])
                        data_drift_lst.append([])
                        data_drift_lst.append([])

            # ----------------------------------------------------------------------------------------------------
            # Creating a Pandas DataFrame:
            column_info = [dist_pair[0],loc_sta1[0],loc_sta2[0]]
            for i in column_info:
                data_drift_lst.append(i)

            columns_headers1 = ['distance','loc_sta1','loc_sta2']
            for i in columns_headers1:
                columns_headers.append(i)


            clock_drift_df = pd.DataFrame(data_drift_lst, index=columns_headers).T

            # ----------------------------------------------------------------------------------------------------
            # Convert from pandas to Arrow and saving in feather formart file
            os.makedirs(FEATHER_FILES,exist_ok=True)
            file_feather_name = FEATHER_FILES+pair_sta_1+'.'+pair_sta_2+'_clock_drift_data.feather'
            feather.write_feather(clock_drift_df, file_feather_name)
            # ----------------------------------------------------------------------------------------------------

            if VERBOSE == True:

                # --------------------------------------------
                # Creating the figure and plotting Clock-drift
                # --------------------------------------------

                fig = plt.figure(figsize=(20, 15))
                fig.suptitle('Clock-drift: '+pair_sta_1+'-'+pair_sta_2+'('+str(round(dist_pair))+' km)',fontsize=20)
                fig.autofmt_xdate()
                # ----------------------------------------------------------------------------------------------------

                gs = gridspec.GridSpec(9, 2,wspace=0.5, hspace=0.8)
                map_loc = fig.add_subplot(gs[:,0],projection=ccrs.PlateCarree())

                LLCRNRLON_LARGE = -52
                URCRNRLON_LARGE = -38
                LLCRNRLAT_LARGE = -30
                URCRNRLAT_LARGE = -12

                map_loc.set_extent([LLCRNRLON_LARGE,URCRNRLON_LARGE,LLCRNRLAT_LARGE,URCRNRLAT_LARGE])
                map_loc.yaxis.set_ticks_position('both')
                map_loc.xaxis.set_ticks_position('both')

                map_loc.set_xticks(np.arange(LLCRNRLON_LARGE,URCRNRLON_LARGE+3,3), crs=ccrs.PlateCarree())
                map_loc.set_yticks(np.arange(LLCRNRLAT_LARGE,URCRNRLAT_LARGE+3,3), crs=ccrs.PlateCarree())
                map_loc.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, labelsize=12)
                map_loc.grid(True,which='major',color='gray',linewidth=0.5,linestyle='--')

                reader_1_SHP = Reader(BOUNDARY_STATES_SHP)
                shape_1_SHP = list(reader_1_SHP.geometries())
                plot_shape_1_SHP = cfeature.ShapelyFeature(shape_1_SHP, ccrs.PlateCarree())
                map_loc.add_feature(plot_shape_1_SHP, facecolor='none', edgecolor='k',linewidth=0.5,zorder=-1)
                # Use the cartopy interface to create a matplotlib transform object
                # for the Geodetic coordinate system. We will use this along with
                # matplotlib's offset_copy function to define a coordinate system which
                # translates the text by 25 pixels to the left.
                geodetic_transform = ccrs.Geodetic()._as_mpl_transform(map_loc)
                text_transform = offset_copy(geodetic_transform, units='dots', y=50,x=100)

                map_loc.plot([loc_sta1[1],loc_sta2[1]],[loc_sta1[0],loc_sta2[0]],c='k',alpha=0.5,transform=ccrs.PlateCarree())
                map_loc.scatter(loc_sta1[1],loc_sta1[0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())
                map_loc.scatter(loc_sta2[1],loc_sta2[0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())

                map_loc.text(loc_sta1[1],loc_sta1[0], pair_sta_1,fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)
                map_loc.text(loc_sta2[1],loc_sta2[0], pair_sta_2,fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)

                # ----------------------------------------------------------------------------------------------------
                days_major = DayLocator(interval=5)   # every 5 day
                days_minor = DayLocator(interval=1)   # every day
                months = MonthLocator(interval=3)  # every month
                yearsFmt = DateFormatter('%b-%Y')

                for z,x in enumerate(data_to_plot_coefficient_clock_drift_chan):
                    if len(x) > 1:
                        # ----------------------------------------------------------------------------------------------------
                        ax0 = fig.add_subplot(gs[z,1])
                        ax0.xaxis.set_major_locator(months)
                        ax0.xaxis.set_major_formatter(yearsFmt)
                        ax0.xaxis.set_minor_locator(days_minor)
                        ax0.yaxis.set_major_locator(MultipleLocator(100))
                        ax0.yaxis.set_minor_locator(MultipleLocator(25))
                        ax0.set_ylabel('Erro do Relógios (s)')
                        ax0.set_title(chan_lst[z])
                        ax0.set_ylim(-100,100)

                        # -------------------------------------------------------------------------------------------------------------
                        slope, intercept, r, p, std_err = stats.linregress(x, y)

                        def myfunc(x):
                          return slope * x + intercept

                        mymodel = list(map(myfunc, x))
                        pol_reg.fit(X_poly, data_to_plot_shift_clock_drift_chan[z])
                        # -------------------------------------------------------------------------------------------------------------
                        for y,u in enumerate(data_to_plot_shift_clock_drift_chan[z]):
                            if data_to_plot_coefficient_clock_drift_chan[z][y] > 0.3:
                                im = ax0.scatter(date_to_plot_clock_chan[z][y],data_to_plot_shift_clock_drift_chan[z][y],c=data_to_plot_coefficient_clock_drift_chan[z][y],marker='o',edgecolors=None,cmap='magma',s=10,vmin=0,vmax=1,alpha=0.9)
                            else:
                                im = ax0.scatter(date_to_plot_clock_chan[z][y],data_to_plot_shift_clock_drift_chan[z][y],c=data_to_plot_coefficient_clock_drift_chan[z][y],marker='o',edgecolors=None,cmap='magma',s=5,vmin=0,vmax=1,alpha=0.2)

                        ax0.plot(date_to_plot_clock_chan[z], pol_reg.predict(poly_reg.fit_transform(np.array(range(len(data_to_plot_shift_clock_drift_chan[z]))).reshape(-1, 1))),'--b')

                        if z == 0:
                            axins = inset_axes(ax0,
                                                   width="30%",  # width = 10% of parent_bbox width
                                                   height="10%",  # height : 5%
                                                   loc='upper left',
                                                   bbox_to_anchor=(0.65,0.1, 1, 1),
                                                   bbox_transform=ax0.transAxes,
                                                   borderpad=0,
                                                   )
                            plt.colorbar(im, cax=axins, orientation="horizontal", ticklocation='top')

                    else:

                        ax0 = fig.add_subplot(gs[z,1])
                        ax0.xaxis.set_major_locator(months)
                        ax0.xaxis.set_major_formatter(yearsFmt)
                        ax0.xaxis.set_minor_locator(days_minor)
                        ax0.yaxis.set_major_locator(MultipleLocator(100))
                        ax0.yaxis.set_minor_locator(MultipleLocator(25))
                        ax0.set_ylabel('Erro do Relógio (s)')
                        ax0.set_title(chan_lst[z])
                        ax0.set_ylim(-200,200)

                # -------------------------------------------------------------------------------------------------------------
                output_figure_CLOCK_DRIFT = CLOCK_DRIFT_OUTPUT+'CLOCK_DRIFT_FIGURES/'
                os.makedirs(output_figure_CLOCK_DRIFT,exist_ok=True)
                fig.savefig(output_figure_CLOCK_DRIFT+'CLOCK_DRIFT_BETWEEN_'+pair_sta_1+'_'+pair_sta_2+'.png',dpi=300)
                plt.close()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# ============
# Main program
# ============

print('\n')
print('========================')
print('Clock Drift Calculating:')
print('========================')
print('\n')

for iOBS in OBS_LST:

    # -------------------------------------------
    # Collecting daily list of cross-correlations
    # -------------------------------------------

    crosscorr_pairs = sorted(glob.glob(ASDF_FILES+'CROSS_CORR_10_DAYS_FILES/**/*.h5', recursive=True))

    # --------------------------------
    # Separating according to OBS name
    # --------------------------------

    crosscorr_pairs_obs = [i for i in crosscorr_pairs if iOBS in i]

    # ------------------
    # Separating by pair
    # ------------------

    crosscorr_pairs_name_lst = []
    for i in crosscorr_pairs_obs:

        # splitting subdir/basename
        subdir, filename = os.path.split(i)
        nameslst = filename.split("_20")[0]

        name_sta1 = nameslst.split('_')[-2].split('..')[0]
        name_sta2 = nameslst.split('_')[-1].split('..')[0]

        if name_sta1 != name_sta2:
            crosscorr_pairs_name_lst.append(name_sta1+'_'+name_sta2)

    crosscorr_pairs_names = sorted(list(set(crosscorr_pairs_name_lst)))

    start_time = time.time()
    with Pool(processes=num_processes) as p:
        max_ = len(crosscorr_pairs_names)
        with tqdm(total=max_,desc='Processing') as pbar:
            for i, _ in enumerate(p.imap_unordered(Calculating_clock_drift_func, crosscorr_pairs_names)):
                pbar.update()
    print("--- %.2f execution time (min) ---" % ((time.time() - start_time)/60))


print('\n')
print('===============================')
print('Total Clock Drift for each OBS:')
print('===============================')
print('\n')

clock_drift_files_lst = sorted(glob.glob(FEATHER_FILES+'/*'))

clock_drift_files = []

for i,j in enumerate(clock_drift_files_lst):
    # splitting subdir/basename
    subdir, filename = os.path.split(j)
    pair_sta_1 = filename.split('_')[0].split('.')[1]
    pair_sta_2 = filename.split('_')[0].split('.')[3]

    if 'OBS' in pair_sta_1 and 'OBS' in pair_sta_2:
        pass
    else:
        clock_drift_files.append(j)

for iOBS in OBS_LST:
    for iband, per_bands in enumerate(PERIOD_BANDS):

        clock_drift_df_lst = [pd.read_feather(j) for i,j in enumerate(clock_drift_files) if iOBS in j]
        # ----------------------------------------------------------------------------------------------------

        df = pd.concat(clock_drift_df_lst, ignore_index=True)

        # --------------------------------------------
        # Creating the figure and plotting Clock-drift
        # --------------------------------------------

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Deriva do Relógio: '+iOBS,fontsize=20)
        fig.autofmt_xdate()

        # ----------------------------------------------------------------------------------------------------
        gs = gridspec.GridSpec(9, 2,wspace=0.5, hspace=0.8)
        map_loc = fig.add_subplot(gs[:,0],projection=ccrs.PlateCarree())

        LLCRNRLON_LARGE = -52
        URCRNRLON_LARGE = -38
        LLCRNRLAT_LARGE = -30
        URCRNRLAT_LARGE = -12
        # ----------------------------------------------------------------------------------------------------

        map_loc.set_extent([LLCRNRLON_LARGE,URCRNRLON_LARGE,LLCRNRLAT_LARGE,URCRNRLAT_LARGE])
        map_loc.yaxis.set_ticks_position('both')
        map_loc.xaxis.set_ticks_position('both')

        map_loc.set_xticks(np.arange(LLCRNRLON_LARGE,URCRNRLON_LARGE+3,3), crs=ccrs.PlateCarree())
        map_loc.set_yticks(np.arange(LLCRNRLAT_LARGE,URCRNRLAT_LARGE+3,3), crs=ccrs.PlateCarree())
        map_loc.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, labelsize=12)
        map_loc.grid(True,which='major',color='gray',linewidth=0.5,linestyle='--')

        reader_1_SHP = Reader(BOUNDARY_STATES_SHP)
        shape_1_SHP = list(reader_1_SHP.geometries())
        plot_shape_1_SHP = cfeature.ShapelyFeature(shape_1_SHP, ccrs.PlateCarree())
        map_loc.add_feature(plot_shape_1_SHP, facecolor='none', edgecolor='k',linewidth=0.5,zorder=-1)
        # Use the cartopy interface to create a matplotlib transform object
        # for the Geodetic coordinate system. We will use this along with
        # matplotlib's offset_copy function to define a coordinate system which
        # translates the text by 25 pixels to the left.
        geodetic_transform = ccrs.Geodetic()._as_mpl_transform(map_loc)
        text_transform = offset_copy(geodetic_transform, units='dots', y=10,x=20)

        # ----------------------------------------------------------------------------------------------------
        for ista,staname in enumerate(df['sta_1'].tolist()):
            map_loc.plot([df['loc_sta1'][ista][1],df['loc_sta2'][ista][1]],[df['loc_sta1'][ista][0],df['loc_sta2'][ista][0]],c='k',alpha=0.5,transform=ccrs.PlateCarree())
            map_loc.scatter(df['loc_sta1'][ista][1],df['loc_sta1'][ista][0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())
            map_loc.scatter(df['loc_sta2'][ista][1],df['loc_sta2'][ista][0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())

            if iOBS in df['sta_1'][ista]:
                map_loc.text(df['loc_sta1'][ista][1],df['loc_sta1'][ista][0], iOBS,color='r',fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)

            if iOBS in df['sta_2'][ista]:
                map_loc.text(df['loc_sta2'][ista][1],df['loc_sta2'][ista][0], iOBS,color='r',fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)

        # ----------------------------------------------------------------------------------------------------

        days_major = DayLocator(interval=15)   # every 5 day
        days_minor = DayLocator(interval=1)   # every day
        months = MonthLocator(interval=1)  # every month
        yearsFmt = DateFormatter('%b-%Y')

        # ----------------------------------------------------------------------------------------------------
        chan_lst = ['HHE-HHE','HHE-HHN','HHE-HHZ','HHN-HHN','HHN-HHE','HHN-HHZ','HHZ-HHE','HHZ-HHN','HHZ-HHZ']

        clock_drift_date_to_plot_total = []
        clock_drift_data_to_plot_coefficient_total = []
        clock_drift_data_to_plot_shift_total = []
        for z,i in enumerate(chan_lst):
            clock_drift_date_to_plot = np.array([item for sublist in df[i+' date ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'].tolist() for item in sublist])
            clock_drift_data_to_plot_coefficient = np.array([abs(num) for num in [item for sublist in df[i+' coefficient ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'].tolist() for item in sublist]])
            clock_drift_data_to_plot_shift = np.array([item for sublist in df[i+' shift ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]'].tolist() for item in sublist])
            # ----------------------------------------------------------------------------------------------------
            ax0 = fig.add_subplot(gs[z,1])
            ax0.xaxis.set_major_locator(months)
            ax0.xaxis.set_major_formatter(yearsFmt)
            ax0.xaxis.set_minor_locator(days_major)
            ax0.yaxis.set_major_locator(MultipleLocator(50))
            ax0.yaxis.set_minor_locator(MultipleLocator(10))
            ax0.set_ylabel('Erro do '+'\n'+'Relógio (s)')
            ax0.set_title(chan_lst[z]+' ['+str(per_bands[0])+'-'+str(per_bands[1])+' s]')
            ax0.set_ylim(-100,100)
            ax0.set_xlim(clock_drift_date_to_plot[0],clock_drift_date_to_plot[-1])

            # -------------------------------------------------------------------------------------------------------------
            data_coefficient_6 = 0.6
            mask = clock_drift_data_to_plot_coefficient >= data_coefficient_6
            mask_ = clock_drift_data_to_plot_coefficient < data_coefficient_6
            # -------------------------------------------------------------------------------------------------------------
            data_coefficient_80 = 0.8
            mask80 = clock_drift_data_to_plot_coefficient >= data_coefficient_80
            # -------------------------------------------------------------------------------------------------------------

            if len(clock_drift_date_to_plot[mask]) > 2:

                clock_drift_date_to_plot_total.append(clock_drift_date_to_plot[mask80])
                clock_drift_data_to_plot_coefficient_total.append(clock_drift_data_to_plot_coefficient[mask80])
                clock_drift_data_to_plot_shift_total.append(clock_drift_data_to_plot_shift[mask80])

                #calculating the mean of cc coefficient and the std of the clock drift
                data_coefficient_mean = np.mean(clock_drift_data_to_plot_coefficient[mask])
                clock_drift_data_to_plot_shift_mean = np.std(clock_drift_data_to_plot_shift[mask])

                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax0.text(0.9, 0.7, '$CC_{av}:$'+str(round(data_coefficient_mean,2))+'\n'+'$\sigma:$'+str(round(clock_drift_data_to_plot_shift_mean,2))+' s', horizontalalignment='center',verticalalignment='center', transform=ax0.transAxes,bbox=props)

            else:
                pass

            # -------------------------------------------------------------------------------------------------------------

            im = ax0.scatter(clock_drift_date_to_plot[mask],clock_drift_data_to_plot_shift[mask],c=clock_drift_data_to_plot_coefficient[mask],marker='o',edgecolors=None,cmap='viridis_r',s=10,vmin=0.5,vmax=1,alpha=0.9)

            ax0.scatter(clock_drift_date_to_plot[mask_],clock_drift_data_to_plot_shift[mask_],c=clock_drift_data_to_plot_coefficient[mask_],marker='o',edgecolors=None,cmap='viridis_r',s=2,vmin=0.5,vmax=1,alpha=0.3)

            if z == 0:

                axins = inset_axes(ax0,
                                   width="30%",  # width = 10% of parent_bbox width
                                   height="10%",  # height : 5%
                                   loc='upper left',
                                   bbox_to_anchor=(0.7,0.2, 1, 1),
                                   bbox_transform=ax0.transAxes,
                                   borderpad=0,
                                   )
                plt.colorbar(im, cax=axins, orientation="horizontal", ticklocation='top',label='CC Pearson')

        # -------------------------------------------------------------------------------------------------------------
        output_figure_CLOCK_DRIFT = CLOCK_DRIFT_OUTPUT+'CLOCK_DRIFT_TOTAL_FIGURES/'
        os.makedirs(output_figure_CLOCK_DRIFT,exist_ok=True)
        fig.savefig(output_figure_CLOCK_DRIFT+'CLOCK_DRIFT_TOTAL_'+iOBS+'_'+str(per_bands[0])+'_'+str(per_bands[1])+'s.png',dpi=300)
        plt.close()

        # --------------------------------------------------
        # Creating the figure and plotting Clock-drift total
        # --------------------------------------------------

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Deriva do Relógio: '+iOBS,fontsize=20)
        fig.autofmt_xdate()

        # ----------------------------------------------------------------------------------------------------

        gs = gridspec.GridSpec(4, 4,wspace=0.5, hspace=0.8)
        map_loc = fig.add_subplot(gs[0:3,:],projection=ccrs.PlateCarree())
        LLCRNRLON_LARGE = -52
        URCRNRLON_LARGE = -38
        LLCRNRLAT_LARGE = -30
        URCRNRLAT_LARGE = -12
        # ----------------------------------------------------------------------------------------------------

        map_loc.set_extent([LLCRNRLON_LARGE,URCRNRLON_LARGE,LLCRNRLAT_LARGE,URCRNRLAT_LARGE])
        map_loc.yaxis.set_ticks_position('both')
        map_loc.xaxis.set_ticks_position('both')

        map_loc.set_xticks(np.arange(LLCRNRLON_LARGE,URCRNRLON_LARGE+3,3), crs=ccrs.PlateCarree())
        map_loc.set_yticks(np.arange(LLCRNRLAT_LARGE,URCRNRLAT_LARGE+3,3), crs=ccrs.PlateCarree())
        map_loc.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, labelsize=12)
        map_loc.grid(True,which='major',color='gray',linewidth=0.5,linestyle='--')

        reader_1_SHP = Reader(BOUNDARY_STATES_SHP)
        shape_1_SHP = list(reader_1_SHP.geometries())
        plot_shape_1_SHP = cfeature.ShapelyFeature(shape_1_SHP, ccrs.PlateCarree())
        map_loc.add_feature(plot_shape_1_SHP, facecolor='none', edgecolor='k',linewidth=0.5,zorder=-1)
        # Use the cartopy interface to create a matplotlib transform object
        # for the Geodetic coordinate system. We will use this along with
        # matplotlib's offset_copy function to define a coordinate system which
        # translates the text by 25 pixels to the left.
        geodetic_transform = ccrs.Geodetic()._as_mpl_transform(map_loc)
        text_transform = offset_copy(geodetic_transform, units='dots', y=10,x=20)

        # ----------------------------------------------------------------------------------------------------
        for ista,staname in enumerate(df['sta_1'].tolist()):
            map_loc.plot([df['loc_sta1'][ista][1],df['loc_sta2'][ista][1]],[df['loc_sta1'][ista][0],df['loc_sta2'][ista][0]],c='k',alpha=0.5,transform=ccrs.PlateCarree())
            map_loc.scatter(df['loc_sta1'][ista][1],df['loc_sta1'][ista][0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())
            map_loc.scatter(df['loc_sta2'][ista][1],df['loc_sta2'][ista][0], marker='^',s=200,c='k',edgecolors='w', transform=ccrs.PlateCarree())

            if iOBS in df['sta_1'][ista]:
                map_loc.text(df['loc_sta1'][ista][1],df['loc_sta1'][ista][0], iOBS,color='r',fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)

            if iOBS in df['sta_2'][ista]:
                map_loc.text(df['loc_sta2'][ista][1],df['loc_sta2'][ista][0], iOBS,color='r',fontsize=12,verticalalignment='center', horizontalalignment='right',transform=text_transform)

        # ----------------------------------------------------------------------------------------------------

        days_major = DayLocator(interval=15)   # every 5 day
        days_minor = DayLocator(interval=1)   # every day
        months = MonthLocator(interval=1)  # every month
        yearsFmt = DateFormatter('%b-%Y')

        # ----------------------------------------------------------------------------------------------------
        clock_drift_date_to_plot_total1 = np.array([item for sublist in clock_drift_date_to_plot_total for item in sublist])
        clock_drift_data_to_plot_coefficient_total1 = np.array([item for sublist in clock_drift_data_to_plot_coefficient_total for item in sublist])
        clock_drift_data_to_plot_shift_total1 = np.array([item for sublist in clock_drift_data_to_plot_shift_total for item in sublist])
        # ----------------------------------------------------------------------------------------------------
        clock_drift_data_to_plot_shift_total1_std = np.std(clock_drift_data_to_plot_shift_total1)
        clock_drift_data_to_plot_shift_total1_mean = np.mean(clock_drift_data_to_plot_shift_total1)
        mask_std = (clock_drift_data_to_plot_shift_total1 >= clock_drift_data_to_plot_shift_total1_mean-clock_drift_data_to_plot_shift_total1_std) & (clock_drift_data_to_plot_shift_total1 <= clock_drift_data_to_plot_shift_total1_mean+clock_drift_data_to_plot_shift_total1_std)

        clock_drift_date_to_plot_total = clock_drift_date_to_plot_total1[mask_std]
        clock_drift_data_to_plot_coefficient_total = clock_drift_data_to_plot_coefficient_total1[mask_std]
        clock_drift_data_to_plot_shift_total = clock_drift_data_to_plot_shift_total1[mask_std]

        # ----------------------------------------------------------------------------------------------------

        ax0 = fig.add_subplot(gs[3,:])
        ax0.xaxis.set_major_locator(months)
        ax0.xaxis.set_major_formatter(yearsFmt)
        ax0.xaxis.set_minor_locator(days_major)
        ax0.yaxis.set_major_locator(MultipleLocator(5))
        ax0.yaxis.set_minor_locator(MultipleLocator(1))
        ax0.set_ylabel('Erro do '+'\n'+'Relógio (s)')
        ax0.set_ylim(-10,10)
        ax0.set_xlim(clock_drift_date_to_plot_total.min(),clock_drift_date_to_plot_total.max())

        #Simple Linear Regression With scikit-learn
        #Provide data:
        x = np.array(range(len(clock_drift_date_to_plot_total))).reshape((-1, 1))
        y = clock_drift_data_to_plot_shift_total

        start_time = np.array(clock_drift_date_to_plot[mask]).min()
        stop_time = np.array(clock_drift_date_to_plot[mask]).max()

        x_pred = np.arange(start=clock_drift_date_to_plot_total.min(),stop=clock_drift_date_to_plot_total.max(),step=TENDAY)

        #model = make_pipeline(PolynomialFeatures(1), HuberRegressor(alpha=.0001,epsilon=3))
        #model = make_pipeline(PolynomialFeatures(1), TheilSenRegressor(n_jobs=12,max_iter=500))

        #model = HuberRegressor(alpha=.0001,epsilon=1.3)
        model = TheilSenRegressor(n_jobs=12,max_iter=500)
        model.fit(x, y)
        y_pred = model.predict(np.array(range(len(x_pred))).reshape((-1, 1)))
        #slope:
        slope = model.coef_[0]*1000


        ax0.plot(x_pred, y_pred,'--k',alpha=0.75)

        #calculating the mean of cc coefficient and the std of the clock drift
        data_coefficient_total_mean = np.mean(clock_drift_data_to_plot_coefficient_total)
        clock_drift_data_to_plot_shift_total_mean = np.std(clock_drift_data_to_plot_shift_total)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax0.text(0.9, 0.8, '$CC_{av}:$'+str(round(data_coefficient_total_mean,2))+'\n'+'$\sigma:$'+str(round(clock_drift_data_to_plot_shift_total_mean,2))+' s'+'\n'+'$r:$'+str(round(slope,2))+' ms/dia', horizontalalignment='center',verticalalignment='center', transform=ax0.transAxes,bbox=props)

        # -------------------------------------------------------------------------------------------------------------

        im = ax0.scatter(clock_drift_date_to_plot_total,clock_drift_data_to_plot_shift_total,c=clock_drift_data_to_plot_coefficient_total,marker='o',edgecolors=None,cmap='viridis_r',s=15,vmin=0.5,vmax=1,alpha=0.9)

        axins = inset_axes(ax0,
                               width="30%",  # width = 10% of parent_bbox width
                               height="10%",  # height : 5%
                               loc='upper left',
                               bbox_to_anchor=(0.7,0.2, 1, 1),
                               bbox_transform=ax0.transAxes,
                               borderpad=0,
                               )
        plt.colorbar(im, cax=axins, orientation="horizontal", ticklocation='top', label='CC Pearson')

        # -------------------------------------------------------------------------------------------------------------
        output_figure_CLOCK_DRIFT = CLOCK_DRIFT_OUTPUT+'CLOCK_DRIFT_TOTAL_FIGURES/'
        os.makedirs(output_figure_CLOCK_DRIFT,exist_ok=True)
        fig.savefig(output_figure_CLOCK_DRIFT+'CLOCK_DRIFT_TOTAL_'+iOBS+'_all.png',dpi=300)
        plt.close()
