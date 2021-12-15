from numpy import float64, zeros, arange, exp, ones, copy, append, dot, mean, std, sum, asarray, max, sqrt, pi, correlate, meshgrid, abs, \
conjugate, fliplr, log10, floor, power, angle, conjugate, linspace, hamming, hanning, add, diff, where, power, real, random, inf, isfinite,\
log, diag, diff, int64, poly1d
import scipy
from scipy.interpolate import interp1d


def read_ccd(is1=10, num=100, tn0=[1.1, 5e+3], ne=[2e+18, 1.2e+20], file_name='ccd19_ne.dat'):
    u"""
    
    """
    log_tn0_in=linspace( log10(tn0[0]), log10(tn0[1]), num)
    log_ne_in=linspace( log10(ne[0]), log10(ne[1]), num)
    f = open(file_name, 'r') 
    d=f.read()
    dd=d.split('******************************************')
    header=dd[0].split()
    n_state=int64(header[0])
    n_ne=int64(header[1])
    n_tn0=int64(header[2])
    
    ddd_pre=dd[1]
    ddd_pre=ddd_pre.split('C-----------------------------------------------------------------------')[0]
    ddd=ddd_pre.split('*****MH=')
    tn0_ne=float64(ddd[0].split() )
    log_ne=tn0_ne[:n_ne]+6
    log_tn0=tn0_ne[n_ne:]
    
    output=zeros( [num, num], dtype=float64)
    output_dum=zeros( [n_ne, num], dtype=float64)
    
    data=ddd[is1]
    data=asarray( float64( data.split('*****************************')[1].split() ) )
    
    data=data.reshape(n_tn0, n_ne)
    
    for i in range(n_ne):
        f = interp1d(log_tn0, data[:,i], kind='cubic')
        output_dum[i,:]=f(log_tn0_in)
    
    for i in range(num):
        f = interp1d(log_ne, output_dum[:,i], kind='cubic')
        output[:,i]=f(log_ne_in)
    
    return 10**log_ne_in, 10**log_tn0_in, 10**(output)*1e-6

def read_acd_scd(is1=3, num = 100, te=[1.1, 5e+3], ne=[2e+18, 1e+20], file_name='acd96_ne.dat'):
    u"""
    read acd96_ne.dat and scd96_ne.dat, units will be converted to m and s
    
    is1: charge state

    return 
    ne_arr,
    te_arr,
    scd or acd
    """
    log_te_in=linspace( log10(te[0]), log10(te[1]), num)
    log_ne_in=linspace( log10(ne[0]), log10(ne[1]), num)
    f = open(file_name, 'r') 
    d=f.read()
    dd=d.replace(" == 6 ================== Behringer data =========================================", "---------------------/ IGRD= 1  / IPRT= 1  /--------/ Z1= 6   / DATE= 20/12/00").split('================================================================================')
    header=dd[0].split()
    n_state=int64(header[0])
    n_ne=int64(header[1])
    n_te=int64(header[2])

    ddd=dd[1].split('---------------------')
    te_ne=float64(ddd[0].split() )
    log_ne=te_ne[:n_ne]+6
    log_te=te_ne[n_ne:]

    output=zeros( [num, num], dtype=float64)
    output_dum=zeros( [n_ne, num], dtype=float64)

    data=ddd[is1]
    data=asarray( float64( data.split('/00')[1].split() ) )
    
    data=data.reshape(n_te, n_ne)

    for i in range(n_ne):
        f = interp1d(log_te, data[:,i], kind='cubic')
        output_dum[i,:]=f(log_te_in)
    
    for i in range(num):
        f = interp1d(log_ne, output_dum[:,i], kind='cubic')
        output[:,i]=f(log_ne_in)
    
    return 10**log_ne_in, 10**log_te_in, 10**(output)*1e-6
#plt.plot(log_ne_in, output[:,5], "*--" )
    
class Lookuptable_ne(object):
     u"""
     forward model for balmer line analysis
     """
     def __init__(self, num = 100, te=[1.1, 5.5e+3], ne=[2e+18, 1.2e+20], n_state=10, acd_file='acd96_ne.dat',\
                                                                                  scd_file='scd96_ne.dat',\
                                                                                  ccd_file='ccd19_ne.dat'):
#     def __init__(self, num = 100, te=[1.1, 5.5e+3], ne=[2e+18, 1.2e+20], n_state=10, acd_file='/afs/ipp-garching.mpg.de/u/rld/strahl/atomdat/newdat/acd96_ne.dat',\
#                                                                                  scd_file='/afs/ipp-garching.mpg.de/u/rld/strahl/atomdat/newdat/scd96_ne.dat',\
#                                                                                  ccd_file='/afs/ipp-garching.mpg.de/u/rld/strahl/atomdat/newdat/ccd19_ne.dat'):
         self.te_arr=zeros( num, dtype=float64)
         self.ne_arr=zeros( num, dtype=float64)
         self.acd_arr_ne=zeros([n_state, num, num])
         self.scd_arr_ne=zeros([n_state, num, num])
         self.ccd_arr_ne=zeros([n_state, num, num])
         
         self.ne_arr, self.te_arr, _ = read_acd_scd(is1=1, num=num, te=te, ne=ne, file_name=acd_file)
         
         for i in range(n_state):
             _, _, self.acd_arr_ne[i,:,:]=read_acd_scd(is1=i+1, num=num, te=te, ne=ne, file_name=acd_file)
             _, _, self.scd_arr_ne[i,:,:]=read_acd_scd(is1=i+1, num=num, te=te, ne=ne, file_name=scd_file)
             _, _, self.ccd_arr_ne[i,:,:]=read_ccd(is1=i+1, num=num, tn0=te, ne=ne, file_name=ccd_file)
             