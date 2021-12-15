from numpy import float64, zeros, arange, exp, ones, copy, append, dot, mean, std, sum, asarray, max, sqrt, pi, correlate, meshgrid, abs, \
conjugate, fliplr, log10, floor, power, angle, conjugate, linspace, hamming, hanning, add, diff, where, power, real, random, inf, isfinite,\
log, diag, diff, histogram, percentile, maximum, minimum, histogram2d, vectorize, int32, int64, tanh
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from netCDF4 import Dataset

from scipy import interpolate

import adas_read_ne

#---------------------------------------------------------------------------------------------------------------------------------------------------------
from netCDF4 import Dataset
#n=173
n=170

file_name="Ne37062t0.000_2.000_5"


wout = Dataset( file_name, more = 'r')

time=asarray( wout.variables['time'][:])

ne_input= asarray( wout.variables['electron_density'][:] )[0,:n]*1e+6   
nz_input= asarray( wout.variables['impurity_density'][:] )[1,:,:n]*1e+6 
n0_input= asarray( wout.variables['neutral_hydrogen_density'][:] )[0,:n]*1e+6 
tn0_input= asarray( wout.variables['neutral_hydrogen_temperature'][:] )[0,:n] 

rho_pol_input = asarray( wout.variables['rho_poloidal_grid'][:] )[:n] 

te_input= asarray( wout.variables['electron_temperature'][:] )[0,:n] 
ti_input= asarray( wout.variables['proton_temperature'][:] )[0,:n] 

r=asarray( wout.variables['radius_grid'][:])[:n]*1e-2

ni=asarray( wout.variables['proton_density'][:] )[0,:n]*1e+6 
#ti=asarray( wout.variables['proton_temperature'][:] )[0,:n]

v_input= asarray( wout.variables['anomal_drift'][:] )[0,:n]*1e-2 
D_input= asarray( wout.variables['anomal_diffusion'][:] )[0,:n]*1e-4 

dr= diff(r) 
#---------------------------------------------------------------------------------------------------------------------------------------------------------
F_ne=adas_read_ne.Lookuptable_ne()

acd_ne= F_ne.acd_arr_ne
scd_ne= F_ne.scd_arr_ne
ccd_ne= F_ne.ccd_arr_ne

te_ne= F_ne.te_arr
ne_ne= F_ne.ne_arr

def acd_cal_ne(te, ne ):
    n_te = where( te>te_ne)[0][-1]+1
    n_ne = where( ne>ne_ne)[0][-1]+1
    
    acd= acd_ne[:, n_ne-1, n_te-1]\
            +(acd_ne[:, n_ne, n_te]-acd_ne[:,n_ne, n_te-1])/(te_ne[n_te]-te_ne[n_te-1] )*(te-te_ne[n_te-1] )\
            +(acd_ne[:, n_ne, n_te]-acd_ne[:,n_ne-1, n_te])/(ne_ne[n_ne]-ne_ne[n_ne-1] )*(ne-ne_ne[n_ne-1] )  
    
    return acd
#---------------------------------------------------------------------------------------------------------------------------------------

def scd_cal_ne(te, ne ):
    n_te = where( te>te_ne)[0][-1]+1
    n_ne = where( ne>ne_ne)[0][-1]+1
    
    scd= scd_ne[:, n_ne-1, n_te-1]\
            +(scd_ne[:, n_ne, n_te]-scd_ne[:,n_ne, n_te-1])/(te_ne[n_te]-te_ne[n_te-1] )*(te-te_ne[n_te-1] )\
            +(scd_ne[:, n_ne, n_te]-scd_ne[:,n_ne-1, n_te])/(ne_ne[n_ne]-ne_ne[n_ne-1] )*(ne-ne_ne[n_ne-1] )  
    
    return scd
#---------------------------------------------------------------------------------------------------------------------------------------
def ccd_cal_ne(tn0, ne ):
    n_tn0 = where( tn0>te_ne)[0][-1]+1
    n_ne = where( ne>ne_ne)[0][-1]+1
    
    ccd= ccd_ne[:, n_ne-1, n_tn0-1]\
            +(ccd_ne[:, n_ne, n_tn0]-ccd_ne[:,n_ne, n_tn0-1])/(te_ne[n_tn0]-te_ne[n_tn0-1] )*(tn0-te_ne[n_tn0-1] )\
            +(ccd_ne[:, n_ne, n_tn0]-ccd_ne[:,n_ne-1, n_tn0])/(ne_ne[n_ne]-ne_ne[n_ne-1] )*(ne-ne_ne[n_ne-1] )  
    
    return ccd
#---------------------------------------------------------------------------------------------------------------------------------------

acd_arr_ne=zeros( [ len( nz_input[:,0] )-1, len(r) ] )
scd_arr_ne=zeros( [ len( nz_input[:,0] )-1, len(r) ] )
ccd_arr_ne=zeros( [ len( nz_input[:,0] )-1, len(r) ] )


for i in range( len(acd_arr_ne[0,:] ) ):
        acd_arr_ne[:,i]=acd_cal_ne( te_input[i], ne_input[i] )
        scd_arr_ne[:,i]=scd_cal_ne( te_input[i], ne_input[i] )
        ccd_arr_ne[:,i]=ccd_cal_ne( tn0_input[i], ne_input[i] )
        
A_re_ne=acd_arr_ne*ne_input+ccd_arr_ne*n0_input
B_io_ne=scd_arr_ne*ne_input

A_re_ne=A_re_ne[::-1,:]
B_io_ne=B_io_ne[::-1,:]
