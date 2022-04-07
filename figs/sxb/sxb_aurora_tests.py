import aurora
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as rbs

ion = 'Ar'
adf15files = {'H': ['pec96#h_pju#h0.dat', 1e-12],
              #'B': ['/afs/ipp/u/sciof/adas/adf15/b/fs#2000.00A_8000.00A#b1.dat',1e-15],
              'B': ['pec93#b_llu#b2.dat', 1e-12],
              'C': ['pec96#c_pju#c4.dat', 1e-15],
              #'N': ['/afs/ipp/u/sciof/adas/adf15/n/pec98#n_ssh_pju#n1.dat', 1e-12]
              'N': ['/afs/ipp/u/sciof/adas/adf15/n/pec96#n_vsu#n2.dat', 1e-11],
              'Ar': ['/afs/ipp/u/sciof/adas/adf15/ar/mbu#ar7.dat', 1e-10],
              }

ad = aurora.adas_files_dict()
fileloc = aurora.get_adas_file_loc(ad[ion]['scd'],filetype='adf11')
scd = aurora.adas_file(fileloc)

# load all transitions provided in the chosen ADF15 file:
path = aurora.get_adas_file_loc(adf15files[ion][0],filetype='adf15')
trs = aurora.read_adf15(path)

try:
    out = aurora.parse_adf15_configs(path)
except:
    out = None
    
# plot only at a single density
ne_val = 2.8e+13 #5e13 # cm^-3
#ne_val = 5e13 # cm^-3


Te_range = [1,800] #[100,1000] #[1, 20] #800]

ls = aurora.get_ls_cycle()

fig,ax = plt.subplots()
for lam in np.unique(trs['lambda [A]']):

    
    # select the excitation-driven component
    tr = trs[(trs['lambda [A]']==lam) & (trs['type']=='excit')].iloc[0]

    #aurora.plot_pec(tr)

    # ne and Te points
    ne = tr['dens pnts']  # cm^-3
    Te = tr['temp pnts']  # eV
    _pecs = tr['PEC pnts']

    pecs = rbs(Te, ne, _pecs.T, kx=1, ky=1)(Te, ne_val)[:,0]

    if np.max(pecs)<adf15files[ion][1]: # limit set for each ion PEC
        # ignore lines that don't emit enough light (arbitrary limit)
        continue

    
    q = int(adf15files[ion][0].split(f'#{ion.lower()}')[1].split('.dat')[0])
    scd_interp = rbs(scd.logT, scd.logNe, scd.data[q-1,:,:], kx=1, ky=1)

    SXB = 10**scd_interp(np.log10(Te), np.log10(ne_val))[:,0]/pecs

    # only use data in chosen Te range
    ind = slice(*Te.searchsorted(Te_range))
    #ind = slice(np.argmin(np.abs(Te-Te_range[0])),np.argmin(np.abs(Te-Te_range[1])))
    
    # plot change in SXB, rather than SXB itself
    ax.plot(Te[ind], SXB[ind]/np.max(SXB[ind]), next(ls), marker='o',
            #ax.plot(Te[ind], SXB[ind], next(ls), marker='o',
            label=fr'$\lambda={lam}A$') #, {tr["transition"].replace(" ","")}')
    ax.set_xlabel(r'$T_e$ [eV]')
    ax.set_ylabel(r'S/PEC norm')
    ax.set_title(label=f'{tr["type"]}, $n_e={ne_val:.1g}$ [m$^{{-3}}$]')
    
ax.legend(loc='best').set_draggable(True)
ax.set_xlim(Te_range)
if ion=='H':
    ax.set_ylim([0.5,None])
