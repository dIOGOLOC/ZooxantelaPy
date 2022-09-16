#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
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
from obspy.signal.rotate import rotate_ne_rt
from obspy.io.xseed import Parser
from obspy.signal.filter import bandpass,lowpass
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import prev_pow_2
from obspy.signal.cross_correlation import correlate as obscorr
from obspy.signal.cross_correlation import xcorr_max

import json
import glob
import os
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from itertools import combinations
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from scipy.signal import spectrogram, detrend, resample,savgol_filter,decimate,hilbert
from obspy.signal.invsim import cosine_taper
from scipy.linalg import norm

import pyarrow.feather as feather

import random
import collections
from copy import copy
import datetime
from itertools import compress

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

EVENT_DIR = '/home/diogoloc/dados_posdoc/ON_MAR/EVENTS/Regional/'

# -------------------------------

# Shapefile  boundary states input

BOUNDARY_STATES_SHP = '/home/diogoloc/dados_posdoc/SIG_Dados/Brasil_RSBR/Shapefile/estados_brasil/UFEBRASIL.shp'

# -------------------------------

# Stations and OBSs information

OBS_LST = ['OBS17','OBS18','OBS20','OBS22']

CHANNEL_LST =  ['.E','.N','.Z']

PERIOD_BANDS2 = [1/50,1/25]

# -------------------------------

# Folders output

ORIENTATION_OUTPUT = '/home/diogoloc/dados_posdoc/ON_MAR/ORIENTATION_OUTPUT/'

# -------------------------------
#create figures?
VERBOSE = False

# Input parameters

FIRSTDAY = '2019-07-01'
LASTDAY = '2020-06-01'

# default parameters to define the signal and noise windows used to estimate the SNR:
# - the signal window is defined according to time after P-wave arrival:

# Rayleigh-wave time windows start
TIME_START_P_REGIONAL = 40

# Rayleigh-wave time windows final
TIME_FINAL_P_REGIONAL = 800

# Returns pairs and spectral SNR array whose spectral SNRs are all >= minspectSNR
minspectSNR = 2

#RESAMPLING
NEW_SAMPLING_RATE = 1

# -------------------------------
# Mappoing parameters

LLCRNRLON_LARGE = -52
URCRNRLON_LARGE = -38
LLCRNRLAT_LARGE = -30
URCRNRLAT_LARGE = -12

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

def filelist(basedir):
    """
    Returns the list of files in *basedir* whose are in the specified period
    """
    files_lst = []
    for root, dirs, files in os.walk(basedir):
        for file in files:
            files_path = os.path.join(root, file)
            if '.X' not in files_path:
                files_lst.append(files_path)

    file_lsts = sorted(files_lst)

    return file_lsts

#-------------------------------------------------------------------------------

# Calculating Cross-Correlation SRN between two traces
def SNR(data,time_data):
	"""
    @type data: numpy array
    @type time_data: numpy array
    """

    # signal window
	tmin_signal = TIME_START_P_REGIONAL
	tmax_signal = TIME_FINAL_P_REGIONAL

	signal_window = (time_data >= tmin_signal) & (time_data <= tmax_signal)
	noise_window = (time_data < tmin_signal) & (time_data > tmax_signal)

	peak = np.abs(data[signal_window]).max()
	noise = data[noise_window].std()

    # appending SNR
	SNR = peak / noise

    # returning SNR
	return SNR

#-------------------------------------------------------------------------------

def Normalize(data):
    """
    z(i)=2*(x(i)−min(x)/max(x)-min(x))−1

    where x=(x1,...,xn) and z(i) is now your ith normalized data between -1 and 1.

    @type data: list
	
    try: normalized_data = [2*(i-data.min()/(data.max()-data.min()))-1 for i in data]

    """
    normalized_data = data
    
    normalized_data /= np.nanmax(normalized_data)
    
    return normalized_data

#-------------------------------------------------------------------------------

# ============
# Main program
# ============

print('=====================')
print('Scanning events files')
print('=====================')
print('\n')
start_time = time.time()

files1 = filelist(basedir=EVENT_DIR)

files_final_1 = []
for i in files1:
        files_final_1.append([i for sta in OBS_LST if sta in i])

files_final = [item for sublist in files_final_1 for item in sublist]

print('\n')
print("--- %.2f execution time (min) ---" % ((time.time() - start_time)/60))
print('\n')

print('\n')
print('============================')
print('Calculating the orientation:')
print('============================')
print('\n')

SNR_MIN = 4
CC_MIN = 0.4

for iOBS in OBS_LST:

    # -----------------------------
    #  Collecting OBS events files
    # -----------------------------

    files_at_OBS = list(filter(lambda x: iOBS in x, files_final))
    # -----------------------------

    HHZ_lst = []
    HHE_lst = []
    HHN_lst = []

    for i in files_at_OBS:
        # ---------------------
        # Separating by channel
        # ---------------------

        # splitting subdir/basename
        subdir, filename = os.path.split(i)
        nameslst = filename.split(iOBS+'.')[1]

        event_name = nameslst[:-2]
        channel_obs = nameslst.split('.')[-1]
  
        # ------------------------------------------------------------------------------------------------------
        if channel_obs == 'Z':
            HHZ_lst.append(i)
        # ------------------------------------------------------------------------------------------------------
        if channel_obs == 'E':
            HHE_lst.append(i)
        # ------------------------------------------------------------------------------------------------------
        if channel_obs == 'N':
            HHN_lst.append(i)
        # ------------------------------------------------------------------------------------------------------
    
    HHE_lst = sorted(HHE_lst)
    HHN_lst = sorted(HHN_lst)
    HHZ_lst = sorted(HHZ_lst)

    OBS_orientation = []
    OBS_cc1_max = []
    OBS_SRN = []
    for k in tqdm(range(len(HHE_lst)),desc=iOBS+' orientation'):
        
        # splitting subdir/basename
        subdir, filename = os.path.split(HHE_lst[k])
        nameslst = filename.split(iOBS+'.')[1]
        event_name = nameslst[:-2]

        #Check if file exists
        output_FEATHER_FILES_ORIENTATION = ORIENTATION_OUTPUT+'FEATHER_FILES/'
        
        file_feather_name = output_FEATHER_FILES_ORIENTATION+event_name+'_ORIENTATION_data.feather'
        if os.path.isfile(file_feather_name):
            pass

        else:
            try:            
                #Data HHE
                tr2_data_file = op.read(HHE_lst[k])
                tr2_data_file.decimate(factor=10, strict_length=False)
                tr2_data_file.taper(type='cosine',max_percentage=0.1)
                tr2_data_file.filter('bandpass',freqmin=PERIOD_BANDS2[0],freqmax=PERIOD_BANDS2[1],zerophase=True)
                tr2_data_filtered = tr2_data_file[0].data

                #Data HHN
                tr1_data_file = op.read(HHN_lst[k])
                tr1_data_file.decimate(factor=10, strict_length=False)
                tr1_data_file.taper(type='cosine',max_percentage=0.1)
                tr1_data_file.filter('bandpass',freqmin=PERIOD_BANDS2[0],freqmax=PERIOD_BANDS2[1],zerophase=True)
                tr1_data_filtered = tr1_data_file[0].data

                #Data HHZ
                trZ_data_file = op.read(HHZ_lst[k])
                trZ_data_file.decimate(factor=10, strict_length=False)
                trZ_data_file.taper(type='cosine',max_percentage=0.1)
                trZ_data_file.filter('bandpass',freqmin=PERIOD_BANDS2[0],freqmax=PERIOD_BANDS2[1],zerophase=True)
                trZ_data_filtered = trZ_data_file[0].data
                trZ_time = trZ_data_file[0].times()

                #epicentral distance:
                dist_pair = trZ_data_file[0].stats.sac.dist
                gcarc_pair = trZ_data_file[0].stats.sac.gcarc
                gcarc_pair_round = round(gcarc_pair)
                dist_pair_round = round(dist_pair)
                baz_pair = trZ_data_file[0].stats.sac.baz
                #-------------------------------------------------------------------------------------------------------------------------------
                # Calculating Hilbert transform of vertical trace data
                trZ_H_data_filtered = np.imag(hilbert(trZ_data_filtered))
                    
                #-------------------------------------------------------------------------------------------------------------------------------
                signal_window = (trZ_time >= TIME_START_P_REGIONAL) & (trZ_time <= TIME_FINAL_P_REGIONAL)
                noise_window = (trZ_time < TIME_START_P_REGIONAL) != (trZ_time > TIME_FINAL_P_REGIONAL)
                noise_rotated = tr1_data_filtered[noise_window].std()

                tr2 = tr2_data_filtered[signal_window]
                tr1 = tr1_data_filtered[signal_window]
                trZ = trZ_data_filtered[signal_window]

                # Calculate Hilbert transform of vertical trace data
                trZ_H = np.imag(hilbert(trZ))

                # Rotate through and find max normalized covariance
                dphi = 0.1
                ang = np.arange(0., 360., dphi)
                ang_arr = np.zeros(len(ang))

                cc1 = np.zeros(len(ang))
                cc2 = np.zeros(len(ang))
                cc3 = np.zeros(len(ang))

                SNR_rotate = np.zeros(len(ang))

                for k, a in enumerate(ang):
                    R, T = rotate_ne_rt(tr1, tr2, a)
                    covmat = np.corrcoef(R, trZ_H)
                    cc1[k] = covmat[0,1]
                    cstar = np.cov(trZ_H, R)/np.cov(trZ_H)
                    cc2[k] = cstar[0,1]
                    cstar=np.cov(trZ_H, T)/np.cov(trZ_H)
                    cc3[k] = cstar[0,1]
                    SNR_rotate[k] = np.abs(R).max()/noise_rotated

                # Get argument of maximum of cc:
                ia = cc2.argmax()
        
                # Get azimuth and correct for angles above 360
                phi = (baz_pair - (360 - ia*dphi))
                
                if phi<0: phi+=360
                if phi>=360: phi-=360
                phi = ang[ia]
                # Get argument of maximum coherence (R_zr):
                cc1_max = cc1.max()
                cc2_max = cc2.max()
                cc3_max = cc3.max()
                SNR_rotate_max = SNR_rotate.max()

                OBS_orientation.append(ang[ia])
                OBS_cc1_max.append(cc1_max)
                OBS_SRN.append(SNR_rotate_max)

                
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

                # --------------------
                # fig CrossCorrelation
                fig = plt.figure(figsize=(20, 10))
                fig.suptitle('Evento: '+event_name+' (Δ:'+str(gcarc_pair_round)+'°)',fontsize=20)

                gs = gridspec.GridSpec(3, 2,wspace=0.2, hspace=0.5)

                new_R, new_T = rotate_ne_rt(tr1_data_filtered, tr2_data_filtered, phi)

                ax1 = fig.add_subplot(gs[0,0])
                ax1.plot(trZ_time,new_T,'-k')
                ax1.set_ylabel('$C_{tz}$')
                ax1.set_xlabel('Timelag (s)')
                ax1.axvline(x=TIME_START_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)
                ax1.axvline(x=TIME_FINAL_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)

                ax2 = fig.add_subplot(gs[1,0], sharey=ax1, sharex=ax1)
                ax2.plot(trZ_time,new_R,'-k')
                ax2.plot(trZ_time,trZ_H_data_filtered,'--r')
                ax2.set_ylabel('$C_{rz}$')
                ax2.set_xlabel('Timelag (s)')
                ax2.axvline(x=TIME_START_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)
                ax2.axvline(x=TIME_FINAL_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)

                ax3 = fig.add_subplot(gs[2,0], sharey=ax1, sharex=ax1)
                ax3.plot(trZ_time,trZ_data_filtered,'-k')
                ax3.set_ylabel('$C_{zz}$')
                ax3.set_xlabel('Timelag (s)')
                ax3.axvline(x=TIME_START_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)
                ax3.axvline(x=TIME_FINAL_P_REGIONAL, ymin=0, ymax=1,color='gray',linestyle='--',lw=1)

                ax4 = fig.add_subplot(gs[0,1])
                ax4.plot(ang,cc1,'--k')
                ax4.plot(phi,cc1_max,'*r')
                ax4.set_ylabel('$R_{rz}$')
                ax4.set_xlabel('Orientation Angle (deg)')

                ax5 = fig.add_subplot(gs[1,1], sharex=ax4)
                ax5.plot(ang,cc2,'--k')
                ax5.plot(phi,cc2_max,'*r')
                ax5.set_ylabel('$S_{rz}$')
                ax5.set_xlabel('Orientation Angle (deg)')

                ax6 = fig.add_subplot(gs[2,1])
                ax6.plot(ang,SNR_rotate,'--k')
                ax6.plot(phi,SNR_rotate_max,'*r')
                ax6.set_ylabel('SNR')
                ax6.set_xlabel('Orientation Angle (deg)')

                if (cc1_max >= CC_MIN) & (SNR_rotate_max >= SNR_MIN):
                    label = 'good'
                else:
                    label = 'bad'
                output_figure_ORIENTATION = ORIENTATION_OUTPUT+'ORIENTATION_FIGURES/EARTHQUAKES/'+iOBS+'/'
                os.makedirs(output_figure_ORIENTATION,exist_ok=True)
                fig.savefig(output_figure_ORIENTATION+'ORIENTATION_'+event_name+'_'+label+'.png',dpi=300)
                plt.close()
            except:
                pass
    #Creating the figure
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(iOBS,fontsize=20)

    gs = gridspec.GridSpec(1, 1,wspace=0.2, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])

    OBS_orientation_np = np.array(OBS_orientation)
    OBS_cc1_max_np = np.array(OBS_cc1_max)
    OBS_SRN_np = np.array(OBS_SRN)

    th_mask = (OBS_cc1_max_np >= CC_MIN) & (OBS_SRN_np >= SNR_MIN)
    th_mask_ = (OBS_cc1_max_np < CC_MIN) & (OBS_SRN_np < SNR_MIN)

    OBS_orientation_np_good = OBS_orientation_np[th_mask]
    OBS_cc1_max_np_good = OBS_cc1_max_np[th_mask]

    OBS_orientation_np_bad = OBS_orientation_np[th_mask_]
    OBS_cc1_max_np_bad = OBS_cc1_max_np[th_mask_]

    #Simple Linear Regression With scikit-learn
    #Provide data:
    x = np.array(range(len(OBS_cc1_max_np_good))).reshape((-1, 1))
    y = OBS_orientation_np_good

    x_pred = np.arange(start=0,stop=1,step=0.1)

    model = TheilSenRegressor(n_jobs=12,max_iter=500)
    model.fit(x, y)
    y_pred = model.predict(np.array(range(len(x_pred))).reshape((-1, 1)))

    ax1.plot(OBS_orientation_np_bad,OBS_cc1_max_np_bad,'.k')
    ax1.plot(OBS_orientation_np_good,OBS_cc1_max_np_good,'ok')
    ax1.plot(np.mean(y_pred),np.mean(OBS_cc1_max_np_good),'*r')

    ax1.set_ylim(0,1)
    ax1.set_xlim(0,360)
    ax1.set_ylabel('$R_{rz}$')
    ax1.set_xlabel('Orientation (degrees)')

    output_figure_ORIENTATION = ORIENTATION_OUTPUT+'ORIENTATION_FIGURES/EARTHQUAKES/'+iOBS+'/'
    os.makedirs(output_figure_ORIENTATION,exist_ok=True)
    fig.savefig(output_figure_ORIENTATION+'ORIENTATION_TOTAL_'+iOBS+'.png',dpi=300)
    plt.close()