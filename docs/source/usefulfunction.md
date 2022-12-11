# Useful functions

## Quality Control 

### Completeness

```{eval-rst}
.. py:function:: assessment_py.data_availability.plot_date_file(FIG_FOLDER_OUTPUT,directory_data)

    Given a MSEED folder, this function returns a plot with the data completeness according to the header of each file. 
    Default plot includes an colorplot of a 2-D array. 
    
    :type FIG_FOLDER_OUTPUT: str
    :param FIG_FOLDER_OUTPUT: Output file path to directly save the resulting image (e.g. ``"/tmp/figures/"``). 
    :type directory_data: str
    :param directory_data: Station folder path (e.g. ``"~/NETWORK/*STATION*/CHANNEL.D/"``).    
    
    :returns: Figure with the plot.
    :rtype: Output[PDF]
```

### Analysis of Seismic Noise Levels

```{eval-rst}
.. py:function:: assessment_py.power_spectral_densities.calc_PSD(file,XML_FILE=XML_FILE)

    Calculate the probabilistic power spectral densities for one trace (specific station/channel).
    Useful for site quality control checks.
  
    Returns a .npz file format (A simple format for saving numpy arrays to disk with the full information about them). 

    :type file: str
    :param file: file path to trace [MSEED file] (e.g. ``"~/STATION/CHANNEL.D/[MSEED FILE]"``). 
    :type XML_FILE: str
    :param XML_FILE: file path to STATIONXML file (e.g. ``"~/STATIONXML/***.XML"``). 
        
    :returns: binary zipped archive 
    :rtype: binary file[npz]
```

```{note} 
    Calculations are based on https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.html.
```
```{tip} 
    See more in https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format.
```

```{eval-rst}
.. py:function:: assessment_py.power_spectral_densities.plot_PSD(directory,INITIAL_DATE=INITIAL_DATE,FINAL_DATE=FINAL_DATE,TIME_OF_WEEKDAY_DAY=TIME_OF_WEEKDAY_DAY, TIME_OF_WEEKDAY_START_HOUR=TIME_OF_WEEKDAY_START_HOUR, TIME_OF_WEEKDAY_FINAL_HOUR=TIME_OF_WEEKDAY_FINAL_HOUR)

    Calculate 2D histogram stack with restrictions: starttime,endtime and time of day. 

    Also plot the probabilistic power spectral densities.

    :type directory: str
    :param directory: file path to .npz folder (e.g. ``"~/STATION/CHANNEL.PPSD/*[npz FILES]"``). 
        
    :returns: Figure with the plot.
    :rtype: Output[PDF]
```
```{note} 
    Calculations are based on https://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.calculate_histogram.html
```
```{tip} 
    For more information: McNamara, D. E. and Buland, R. P. (2004), Ambient Noise Levels in the Continental 
    United States,Bulletin of the Seismological Society of America, 94(4), 1517-1527.
```

## Noise Reduction

###  Tilt noise

```{eval-rst}
.. py:function:: Coherence_xy(G_xy,G_xx,G_yy,wind_number)
    
    Estimate Coherence and Coherence error.

    Coherence describes the degree to which x(t) matches y(t) at the specified frequency.
    it is the square of the correlation coefficient and represents the fraction of the power of y(t) that can be predicted linearly from x(t).

    
    :type G_xy: numpy.array
    :param G_xy: cross-spectral density function between x(t) and y(t). 
    :type G_xx: numpy.array
    :param G_xx: cross-spectral density function between x(t) and x(t). 
    :type G_yy: numpy.array
    :param G_yy: cross-spectral density function between y(t) and y(t). 
    :type wind_number: float
    :param wind_number: number of subrecords.       
    :returns: Coherence,Coherence error
    :rtype: tuple of values
```

```{tip}
    See more information in BELL, S.W.; FORSYTH, D.W.; RUAN, Y. Removing noise from the vertical component 
    records of ocean-bottom seismometers: results from year one of the cascadia initiative, bulletin of the 
    seismological society of america, 105, p. 300-313, 2014.
```

```{admonition} Formula
$$ 
    \gamma^{2}_{xy}(f) = {|G_{xy}(f)|^{2} \over G_{xx}(f)G_{yy}(f)} (Coherence)
$$

$$
    \varepsilon (\gamma^{2}_{xy}(f)) = {\sqrt{2}(1 - \gamma^{2}_{xy}(f)) \over \sqrt{n_{d}}|\gamma_{xy}(f)|}  (Error)
$$

```

```{eval-rst}
.. py:function:: Admittance_xy(G_xy,G_xx,G_yy,wind_number)

    Estimate admittance and admittance error.
    
    Admittance is the amplitude or gain factor of the transfer function between x(t) and y(t). 
    
    :type G_xy: numpy.array
    :param G_xy: cross-spectral density function between x(t) and y(t). 
    :type G_xx: numpy.array
    :param G_xx: cross-spectral density function between x(t) and x(t). 
    :type G_yy: numpy.array
    :param G_yy: cross-spectral density function between y(t) and y(t). 
    :type wind_number: float
    :param wind_number: number of subrecords.       
    :returns: Admittance,Admittance error
    :rtype: tuple of values
```

```{tip}
    See more information in BELL, S.W.; FORSYTH, D.W.; RUAN, Y. Removing noise from the vertical component 
    records of ocean-bottom seismometers: results from year one of the cascadia initiative, bulletin of the 
    seismological society of america, 105, p. 300-313, 2014.
```

```{admonition} Formula
$$ 
    A_{xy}(f) = {|G_{xy}(f)| \over G_{xx}(f)} (Admittance)
$$
    
$$
    \varepsilon (A_{xy}(f)) = {\sqrt{1 - \gamma^{2}_{xy}(f)} \over \sqrt{2_{nd}}|\gamma_{xy}(f)|}  (Error)
$$
```

```{eval-rst}
.. py:function:: Phase_xy(Q_xy,C_xy,G_xy,G_xx,G_yy,wind_number)

    The phase function gives the phase shift between x(t) and y(t) as a function of frequency.

    :type Q_xy: numpy.array
    :param Q_xy: cross-spectral density function between x(t) and y(t), after breaking X and Y into real and imaginary portions. 
    :type C_xy: numpy.array
    :param C_xy: cross-spectral density function between x(t) and y(t), after breaking X and Y into real and imaginary portions.  
    :type G_xy: numpy.array
    :param G_xy: cross-spectral density function between x(t) and y(t). 
    :type G_xx: numpy.array
    :param G_xx: cross-spectral density function between x(t) and x(t). 
    :type G_yy: numpy.array
    :param G_yy: cross-spectral density function between y(t) and y(t). 
    :type wind_number: float
    :param wind_number: number of subrecords.       
    :returns: Coherence,Coherence error
    :rtype: tuple of values
```
```{tip}
    See more information in BELL, S.W.; FORSYTH, D.W.;RUAN, Y. Removing noise from the vertical component 
    records of ocean-bottom seismometers: results from year one of the cascadia initiative, bulletin 
    of the seismological society of america, 105, p. 300-313, 2014.
```

```{admonition} Formula
$$ 
    \phi_{xy}(f) = arctan \left [Q_{xy}(f) \over C_{xy}(f) \right ] (Phase)
$$
    
$$
    \varepsilon (phi_{xy}(f)) = {\sqrt{1 - \gamma^{2}_{xy}(f)} \over \sqrt{2_{nd}}|\gamma_{xy}(f)|}  (Error)
$$
```

### Clock drift

```{eval-rst}
.. py:function:: obscorr_window(data1,time_data1,data2,time_data2,dist,vmin,vmax,sampling_rate)

    Calculate the CrossCorrelation according to the distance between two stations.

    The signal window is defined by *vmin* and *vmax*:
        dist/*vmax* < t < dist/*vmin*

    :type data1: numpy array
    :param data1: amplitude data of the station 1.
    :type time_data1: numpy array
    :param time_data1: time data of the station 1.
    :type data2: numpy array
    :param data2: amplitude data of the station 2.
    :type time_data2: numpy array 
    :param time_data2: time data of the station 2.
    :type dist: float
    :param dist: distance between the stations.
    :type vmin: float
    :param vmin: minimum velocity of Rayleigh waves.
    :type vmax: float
    :param vmax: maximum velocity of Rayleigh waves.
    :type sampling_rate: float
    :param sampling_rate: sampling rate of the data.
    :returns: shift/sampling_rate, correlation coefficient
    :rtype: tuple of values.
```
```{tip}
    See more information in Sarah Hable, Karin Sigloch, Guilhem Barruol, Simon C Stähler, Céline Hadziioannou, 
    Clock errors in land and ocean bottom seismograms: high-accuracy estimates from multiple-component noise 
    cross-correlations, Geophysical Journal International, Volume 214, Issue 3, September 2018, Pages 2014–2034,
    https://doi.org/10.1093/gji/ggy236
```

```{admonition} Formula
    See more in https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
```

### Sensor Orientation

```{eval-rst}
.. py:function:: Calc_orientation(HHE_data,HHN_data,HHZ_data)

    Estimate the optimal sensor orientations by ﬁnding the orientation azimuth that maximizes the correlation between the measured response functions C_rz and C_zz between station-event pairs. We then estimate the angle (ANG) needed to rotate the H1-H2 coordinate system into radial-transverse coordinate, using the guiding principle that the true ANG will maximizes the zero-lag cross-correlation between the radial response function C_rz and phase-shifted vertical response function C_zz (S_zr minimum). The optimization is performed via a grid search with 1 degree steps, using the R_zr value as quality control estimate. 

    :type HHE_data: obspy.trace
    :param HHE_data: horizontal component (E-W). 
    :type HHN_data: obspy.trace
    :param HHN_data: horizontal component (N-S). 
    :type HHZ_data: obspy.trace
    :param HHZ_data: vertical component (U-D). 
    :returns: ANG,S_zr, Rzr,SNR
    :rtype: tuple of values ([0, 360], [−1, +1],[−1, +1],[-,+])
```
```{tip}
    See more information in Xu, W., S. Yuan, W. Wang, X. Luo, and L. Li (2020). Comparing Orientation Analysis Methods for a Shallow-Water Ocean-Bottom Seismometer Array in the Bohai Sea, China, Bull. Seismol. Soc. Am. XX, 1–11, doi: 10.1785/0120200174.
```

```{admonition} Formula
To rotate $C_{ez}$ and $C_{nz}$ to the radial and transverse components $C_{rz}$,$C_{tz}$:

$$ 
    \left ( \frac{C_{rz}}{C_{tz}} \right ) = \begin{pmatrix}
    cos \theta & sin \theta \\ 
    -sin \theta  & cos \theta  
    \end{pmatrix}
    =
    \left ( \frac{C_{nz}}{C_{ez}} \right )
$$
        
$$
    S_{rz}(\psi) = \frac{\rho(C_{rz},\tilde{C_{zz}})}{\rho(\tilde{C_{zz}},\tilde{C_{zz}})}
$$

$$
    R_{rz}(\psi) = \frac{\rho(C_{rz},\tilde{C_{zz}})}{\sqrt{\rho(C_{rz},C_{rz})\rho(\tilde{C_{zz}},\tilde{C_{zz}})}}$$
```