#!/usr/bin/python -u

'''
--------------------------------------------------------------------------------
       Function to plot local the dataset according to local events time
--------------------------------------------------------------------------------

Author: Diogo L.O.C. (locdiogo@gmail.com)


Last Date: 09/2022


Project: Monitoramento Sismo-Oceanográfico
P. Number: 2015/00515-6


Description:
This code will plot the local dataset according to a given list of an events.


Inputs:
Event traces (format: SAC)


Outputs:
Figures (PDF)


Examples of Usage (in command line):
   >> python plot_LOCAL_EVENT.py

'''

import time
import os
from tqdm import tqdm
import glob
from multiprocessing import Pool
from obspy import read,read_inventory, UTCDateTime, Stream,Inventory
from obspy.signal.array_analysis import array_processing
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential

from matplotlib.dates import YearLocator, MonthLocator, DayLocator, DateFormatter
import matplotlib.dates as mdates
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt


from visual_py.event_plot import plot_event_data,plot_map_event_data,plot_map_event_data_hydrophone

from parameters_py.config import (
					OUTPUT_FIGURE_DIR,DIR_DATA,XML_FILE,OUTPUT_EV_DIR,DIR_STATUS,NUM_PROCESS,OUTPUT_PSD_DIR,XML_FILE,LABEL_LANG
					)

# =====================
# Retrieving .SAC files
# =====================
start_time = time.time()


if LABEL_LANG == 'br':
    print('\n')
    print('Procurando os arquivos dos eventos locais.')
    print('\n')
else:
    print('\n')
    print('Collecting local events files.')
    print('\n')

EVENT_dir = []

for root, dirs, files in os.walk(OUTPUT_EV_DIR+'Local/'):
    for directory in dirs:
        if len(directory.split('.')) > 5:
            EVENT_dir.append(os.path.join(directory))

EVENT_dir = sorted(list(set(EVENT_dir)))

#--------------------------------------------------------------------------------------------------------------------

HHZ_files = []

for root, dirs, files in os.walk(OUTPUT_EV_DIR+'Local/'):
    for file in files:
        if '.Z' in file:
            HHZ_files.append(os.path.join(root,file))

lst_eventsZ = []

for i in EVENT_dir:
  lst_eventsZ.append([k for k in HHZ_files if i in k ])

#--------------------------------------------------------------------------------------------------------------------

HHN_files = []

for root, dirs, files in os.walk(OUTPUT_EV_DIR+'Local/'):
    for file in files:
        if '.N' in file:
            HHN_files.append(os.path.join(root,file))

lst_eventsN = []

for i in EVENT_dir:
  lst_eventsN.append([k for k in HHN_files if i in k ])

#--------------------------------------------------------------------------------------------------------------------

HHE_files = []

for root, dirs, files in os.walk(OUTPUT_EV_DIR+'Local/'):
    for file in files:
        if '.E' in file:
            HHE_files.append(os.path.join(root,file))

lst_eventsE = []

for i in EVENT_dir:
  lst_eventsE.append([k for k in HHE_files if i in k ])

#--------------------------------------------------------------------------------------------------------------------

HHX_files = []

for root, dirs, files in os.walk(OUTPUT_EV_DIR+'Local/'):
    for file in files:
        if '.X' in file:
            HHX_files.append(os.path.join(root,file))

HHX_files_RSBR = []

lst_eventsX = []

for i in EVENT_dir:
  lst_eventsX.append([k for k in HHX_files if i in k ])


# -------------------------------------
# Saving all files into a MSEED stream:
# -------------------------------------
inv_files = [read_inventory(i) for i in glob.glob(XML_FILE+'*')]

inv = Inventory()
for i in inv_files:
    inv += i 

for i,j in enumerate(lst_eventsZ):
  
    event_name_info = j[0].split('/')[-2]

    year_ev=int(event_name_info.split('.')[0])
    julday_ev=int(event_name_info.split('.')[1])
    hour_ev=int(event_name_info.split('.')[2])
    minute_ev=int(event_name_info.split('.')[3])
    second_ev=int(event_name_info.split('.')[4])


    # stream
    st = Stream()
    for k in lst_eventsZ[i]:
        st_u = read(k)[0]
        st_u.stats.coordinates = AttribDict({
        'latitude': st_u.stats.sac.stla,
        'elevation': 0,
        'longitude': st_u.stats.sac.stlo})

        st.append(st_u)
    #st.remove_response(inventory=inv)  
    st.select(station='OBS18')
    st.plot()
        
    # Execute array_processing
    # time
    stime = st[0].stats.starttime
    etime = st[0].stats.starttime+10


    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=1.0, win_frac=0.05,
        # frequency properties
        frqlow=1.0, frqhigh=8.0, prewhiten=0,
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


#output = OUTPUT_EV_DIR+'Local/NETWORK_MSEED_FILES/'+event_name_info+'/'
#os.makedirs(output,exist_ok=True)
#st_all.write(output+'event_'+event_name_info+'.mseed')

print("--- %.2f execution time (min) ---" % ((time.time() - start_time)/60))
print('\n')
