'''Functions/methods to load experimental AUG data to compare to 2D edge modeling. 
'''
import sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.pyplot import cm
from IPython import embed
from scipy.interpolate import interp1d, griddata
import aug_sfutils as sf
import matplotlib as mpl
import divspec_main
import aurora_oedge
from scipy.ndimage import median_filter
from midplane_profs import aug_profile_data
from atomID.AUG.get_aug_los import read_los_coord, shot_to_year


class aug_diag:
    '''General class for AUG diagnostics. The setup is designed to enable
    effective comparison with synthetic diagnostics from a 2D model.
    '''
    def __init__(self, shot, time_s, diag_name, equ=None):
        self.shot = int(shot)
        self.time_s = float(time_s)
        self.diag_name = str(diag_name)

        # load equilibrium if not passed via argument
        self.equ = equ if equ is not None else sf.EQU(self.shot, diag='EQI')
        
    def load_data(self, t0, t1):
        self.t0 = float(t0)
        self.t1 = float(t1)
        
        self.data_expt = {}
        
    def plot_data(self):
        # method to be implemented in inhereting classes
        pass

    def setup_synth_diags_base(self, oedge_label='osm'):
        '''Basic setup of synthetic diagnostics based on OEDGE simulations.
        The `oedge_label` parameter allows specification of which simulation should be loaded 
        for a given shot and time.

        Inhereting classes for diagnostics requiring line-integration will need to supplement
        the basic setup done here.
        '''
        self.oedge_case = aurora_oedge.oedge_case(self.shot, self.time_s, label=oedge_label)
        self.oedge_case.load_output()
        
        self.synth_diags = {'oedge_case': self.oedge_case}

        self.data_model = {}
        
    def eval_local_synth(self):
        '''Evaluate a synthetic diagnostic for a local measurement, e.g. Thomson scattering.
        '''
        nc = self.oedge_case.output.nc

        ll = - np.infty
        for quant in self.data_expt['quants']:

            # find quantity label in OEDGE output
            field = self.oedge_case.output.name_maps[quant]['data']

            mask = np.nonzero(nc[field]['data'])

            # interpolate on experimental data points
            self.data_model[quant] = griddata((nc['RS']['data'][mask], nc['ZS']['data'][mask].T),
                                              nc[field]['data'][mask],
                                              (self.data_expt['R'], self.data_expt['Z']))

            # normalized error
            chi = (self.data_expt[quant] - self.data_model[quant])/(self.data_expt[f'{quant}_unc']+1e-10)
            chi_vec.append(chi)
                    
            # add ln-likelihood contribution
            ll += self.chi_to_lnlike(chi, self.data_expt[f'{quant}_unc'], self.data_expt['w'], lnlike_type=0)

        return ll

    def eval_line_int_synth(self):
        '''Evaluate synthetic diagnostics for a line-integrated signal.
        '''
        for los in self.data_expt['los_names']:

            # select cached synthetic diagnostic setup
            dcase = self.synth_diags[diag][los]

            # get species name and charge state from signal label
            imp = diag.split('_')[0].replace('D','H')
            cs = int(diag.split('_')[1])

            # get ISEL index for line of interest -- excitation first
            lines_df = divspec_main.line_adas_info()
            isel = int(lines_df.loc[(lines_df['label']==diag)&(lines_df['type']=='excit')]['isel'])
            dcase.get_single_line_intensity(ion=imp, cs=cs, isel=isel, recom=False)

            #self.data_model[line]
            self.data_model[los] = dcase.bright_los

            # now recombination
            lines_df = divspec_main.line_adas_info()
            isel = int(lines_df.loc[(lines_df['label']==diag)&(lines_df['type']=='recom')]['isel'])
            dcase.get_single_line_intensity(ion=imp, cs=cs, isel=isel, recom=False)
            self.data_model[los] += dcase.bright_los

            # add all molecular components
            line_name = lines_df.loc[lines_df['isel']==isel]['names'].tolist()[0].split('-')[1]
            if line_name in ['alpha','beta','delta','gamma']:
                # no molecular components for higher-n lines 
                dcase.get_balmer_mol_intensity(line_name=line_name)
                self.data_model[los] += dcase.balmer_mol_bright_los

        # least-square analytic minimization to rescale synthetic signals
        if LS_rescale:
            scale = 1.
        else:
            scale = np.nansum(
                self.data_model[los]*self.data_expt['sig']/self.data_expt['sig_unc']**2)/\
                (np.nansum((self.data_model[los]/self.data_expt['sig_unc'])**2)+1e-10)

        # normalized error
        model_vals = np.array(list(self.data_model.values()))
        chi = (self.data_expt['sig'] - scale*model_vals)/self.data_expt['sig_unc']
        chi_vec.append(chi)

        return self.chi_to_lnlike(chi, self.data_expt['sig_unc'], self.data_expt['w'], lnlike_type=0)

    def eval_line_ave_synth(self):
        '''Evaluate synthetic diagnostics for a line-averaged signal.
        '''
        nc = self.oedge_case.output.nc

        ll = - np.infty
        for quant in self.data_expt['quants']:

            # find quantity label in OEDGE output
            field = self.oedge_case.output.name_maps[quant]['data']

            mask = np.nonzero(nc[field]['data'])

            for los in self.data_expt['los_names']:
                # extract quantity along LOS
                pnt1 = self.data_expt['pnt1'][los]
                pnt2 = self.data_expt['pnt2'][los]

                self.data_model[quant] = np.zeros_like(self.data_expt[quant])
                for tidx in np.arange(pnt1.shape[1]):
                    # Similar to what's in solps.get_3d_path:
                    xyz = np.outer(pnt2[:,tidx] - pnt1[:,tidx],
                                   np.linspace(0, 1.0, int(101))) + pnt1[:, [tidx]]
                    pathR, pathZ = np.hypot(xyz[0], xyz[1]), xyz[2]

                    # interpolate to points along LOS
                    oedge_RZ = griddata(
                        (nc['RS']['data'][mask], nc['ZS']['data'][mask].T),
                        nc[field]['data'][mask],
                        (pathR, pathZ)
                    )

                    # path length elements
                    pathL = np.linalg.norm(xyz - pnt1[:, [tidx]], axis=0)
                    dl = np.hstack((np.array([0,]),np.diff(self.pathL)))

                    # line-average
                    self.data_model[quant][:,tidx] = np.average(oedge_RZ, weights=dl)

            # normalized error
            chi = (self.data_expt[quant] - self.data_model[quant])/\
                  (self.data_expt[f'{quant}_unc']+1e-10)
            chi_vec.append(chi)

            # add ln-likelihood contribution
            ll += self.chi_to_lnlike(
                chi, self.data_expt[f'sig_unc'], self.data_expt['w'], lnlike_type=0)

        return ll

    def plot_comparison(self):
        '''General method, to be implemented for each inheriting diagnostic class,
        which should select one of the methods 
        `plot_local_comparison`, `plot_line_int_comparison`, `plot_line_ave_comparison`.
        '''
        pass

    def get_xpnt_pos(self, t0, t1):
        ''' Load x-point coordinates within the chosen time interval.
        '''
        # get GQH equilibrium to load x-point position
        gqh = sf.SFREAD('GQH', self.shot)

        # determine if USN or LSN by checking values of psin
        _txpu = gqh.gettimebase('Rxpu')        
        tindu = slice(*_txpu.searchsorted([t0,t1]))
        txp = _txpu[tindu]
        Rxp = gqh.getobject('Rxpu')[tindu]
        Zxp = gqh.getobject('Zxpu')[tindu]

        return txp, Rxp, Zxp
            
    def plot_machine(self, ax):
        '''Plot machine contours and some flux surfaces on the provided axes.
        '''
        gc_d = sf.getgc() 
        for gc in gc_d.values(): 
            ax.plot(gc.r, gc.z, 'k-')
        plt.axis('equal')

        # in the core:
        rhop = np.linspace(0,1.0, 11)
        r2, z2 = sf.rho2rz(self.equ, rhop, t_in=self.time_s, coord_in='rho_pol')
        for jrho, rho in enumerate(rhop):                                
            ax.plot(r2[0][jrho], z2[0][jrho], 'b-')

        # now in the SOL:
        rhop = np.linspace(1.01,1.05, 3)
        r2, z2 = sf.rho2rz(self.equ, rhop, t_in=self.time_s, coord_in='rho_pol')
        for jrho, rho in enumerate(rhop):                                
            ax.plot(r2[0][jrho], z2[0][jrho], 'k-')    

    def plot_local_comparison(self, quant='ne'):
        '''Plot a comparison of experimental and modelled values for the chosen
        *local* quantity. Note that the nomenclature of the `quant` input
        must be the one expected by the `aurora_oedge.oedge_output` class.
        '''
        fig,ax = plt.subplots(figsize=(10,5), num=f'local {quant} vals ({self.diag_name})')
        set_labels = False
        if not ax.lines:
            set_labels = True                        

        if quant=='ne':
            mask = self.data_model[quant]<1e23 #1e20
        elif quant=='Te':
            mask = self.data_model[quant]<5000 #500

        if 'core_mask' in self.data_expt:
            mask *= self.data_expt['core_mask']

        # plot as a function of Z for vertical diagnostic arrays
        xx = self.data_expt[self.data_expt['plot_coord']]

        ax.plot(xx[mask], self.data_model[quant][mask], 'ro', label='model')
        ax.errorbar(xx.flatten(), self.data_expt[quant].flatten(),
                    self.data_expt[f'{quant}_unc'].flatten(),
                    fmt='.', c='k', label='expt', alpha=0.2)

        if set_labels:
            ax.set_yscale('log')
            ax.set_xlabel(r'R [m]' if self.data_expt['plot_coord']=='R' else r'Z [m]')
            label = oedge_case.output.name_maps[quant]['label']
            units = oedge_case.output.name_maps[quant]['units']
            ax.set_ylabel(f'{label} [{units}]')

        # 2D plot of normalized differences between model and experiment
        fig,ax = plt.subplots(num=f'local {quant} chi')
        set_labels = False
        if not ax.lines:
            set_labels = True

        ctr = ax.contourf(self.data_expt['R'], self.data_expt['Z'], chi)

        if set_labels:
            cbar = fig.colorbar(ctr)
            label = oedge_case.output.name_maps[quant]['label']
            units = oedge_case.output.name_maps[quant]['units']
            cbar.ax.set_ylabel(f'{label} (expt-model)/expt-unc')
            self.plot_machine(ax)
            ax.set_xlabel(r'R [m]')
            ax.set_ylabel(r'Z [m]')

    def plot_line_int_comparison(self):
        '''Plot a comparison of experimental and modelled values for the chosen
        *line-integrated* volumetric quantity, e.g. line emission along a chosen LOS.
        '''
        fig,ax = plt.subplots(num=f'line-integrated {self.diag_name}')
            
        num = len(self.data_expt['sig'])
        ax.plot(np.arange(num), self.data_expt['sig'], '.', label=self.diag_name+' expt')
        ax.plot(np.arange(num), np.array(list(self.data_model.values())),
                    '.', label=self.diag_name+' model')
        ax.legend(loc='best').set_draggable(True)
        ax.set_xticks(np.arange(num))
        ax.set_xticklabels(self.data_expt['los_names'], rotation='vertical')
        ax.set_ylabel(r'Signal [phot/m$^2$/sr/s]')
        plt.tight_layout()

    def plot_line_ave_comparison(self, quant):
        ''' Compare experimental measurements and results from synthetic diagnostics 
        for a quantity that is a weighted average over a LOS, e.g. Ti.
        ''' 
        fig,ax = plt.subplots(num=f'line-averaged {self.diag_name}')
        cols = cm.rainbow(np.linspace(0, 1, len(los_names)))
        for l,los in enumerate(self.data_expt['los_names']):
            ax.plot(self.data_expt['time'], self.data_model[quant][:,l], c=cols[l],
                    ls='o-')
            ax.errorbar(self.data_expt['time'], self.data_expt[quant][:,l],
                        self.data_expt[f'{quant}_unc'][:,l],
                        fmt='.', c=cols[l], ls='+-', alpha=0.2)

        ax.plot([],[], 'ko', label='model')
        ax.errorbar([],[],[], fmt='+', c='k', label='expt', alpha=0.2)
        ax.set_xlabel(r'time [s]')
        label = oedge_case.output.name_maps[quant]['label']
        units = oedge_case.output.name_maps[quant]['units']
        ax.set_ylabel(f'{label} [{units}]')
        ax.legend(loc='best').set_draggable(True)
        plt.tight_layout()
            
    def chi_to_lnlike(self, chi, sigmas, w, lnlike_type=0):
        '''Compute log-likelihood value from array of chi values for a certain diagnostic. 

        Based on the value of :math:`\chi`, one may compute a number of log-likelihood
        (:math:`\mathcal{L}`) values, based on the data model (Gaussian, Cauchy, etc.).

        The weight factors :math:`w_i` allow scaling of the uncertainty
        for a given diagnostic. The log-posterior is then computed as
        
        .. math::
            
            \ln p \propto -\mathcal{L} + \ln p(D, V)
        
        where :math:`\ln p(D, V)` is the log-prior.

        Parameters
        ----------
        chi : list
            Differences between experimental and modelled signals, normalized by the
            experimental uncertainty.
        w : list of floats
            Arbitrary weights for each data point
        lnlike_type : int
            Selection of log-likelihood type, currently from
            {0: simple Gaussian, 1: Cauchy}

        Returns
        -------
        float
             Log-likelihood value
        '''
        if lnlike_type == 0:
            # standard Gaussian likelihood, allowing for variable diagnostic weights
            gaussian_lnlike = +0.5*np.log(w) - np.log(np.sqrt(2*np.pi)*sigmas) - 0.5*w*chi**2   
            lnlike_val = gaussian_lnlike[~np.isnan(gaussian_lnlike)].sum()

        elif lnlike_type == 1:
            # Cauchy likelihood
            cauchy_lnlike = + 0.5*np.log(w) - np.log(np.pi*sigmas) - np.log(1.+w*chi**2)
            lnlike_val = cauchy_lnlike[~np.isnan(cauchy_lnlike)].sum()  # log-likelihood

        else:
            raise ValueError('Unrecognized log-likelihood type')

        return lnlike_val

 
class ets(aug_diag):
    '''Midplane edge Thomson Scattering (ETS)'''
    def __init__(self, shot, time_s, equ=None):
        super(ets, self).__init__(shot, time_s, 'ets', equ)

        self.sf_obj = sf.SFREAD(self.shot, 'vta')
        if not self.sf_obj.status:
            print(f'Could not load VTA for shot={shot}')
            return
        
    def load_data(self, t0, t1):
        ''' Load experimental data.
       
        Parameters
        ----------
        t0,t1 : floats
            Beginning and end of time averaging window
        plot : bool
            If True, plot data for the given time window.
        '''
        super(ets, self).load_data(t0,t1)

        # load only edge array
        _t_vta = self.sf_obj.gettimebase('Ne_e')
        tind = slice(*_t_vta.searchsorted([t0,t1]))
        t_vta = _t_vta[tind]
        
        _R_vta_tmp = self.sf_obj.getobject('R_edge') # R can change over time, but actually fixed
        _R_vta = np.mean(_R_vta_tmp) # assume R is fixed in time (true for 38996)
        Z_vta = self.sf_obj.getobject('Z_edge')  # not changing over time

        # flatten all arrays, since this workflow is for steady-state data only
        #R_vta = np.tile(_R_vta, (len(_Z_vta),1)).T[tind].flatten()
        #Z_vta = np.tile(_Z_vta, (len(_R_vta),1))[tind].flatten()
        R_vta = np.tile(_R_vta, len(Z_vta))

        ne = copy.deepcopy(self.sf_obj.getobject('Ne_e')[tind])
        ne_unc = copy.deepcopy(self.sf_obj.getobject('SigNe_e')[tind])

        Te = copy.deepcopy(self.sf_obj.getobject('Te_e')[tind])
        Te_unc = copy.deepcopy(self.sf_obj.getobject('SigTe_e')[tind])
        
        # reduce time resolution, averaging within 50 ms time windows
        gg = 6
        num = len(t_vta)//gg # reduce time res from 8 to 50 ms
        t_vta = np.nanmedian(t_vta[:gg*num].reshape(-1,gg), axis=1)
        ne = np.nanmedian(ne[:gg*num].reshape(-1,gg,ne.shape[1]),axis=1)
        ne_unc = np.nanmedian(ne_unc[:gg*num].reshape(-1,gg,ne_unc.shape[1]),axis=1)
        Te = np.nanmedian(Te[:gg*num].reshape(-1,gg,Te.shape[1]),axis=1)
        Te_unc = np.nanmedian(Te_unc[:gg*num].reshape(-1,gg,Te_unc.shape[1]),axis=1)
        
        # exclude bad points
        ne[ne<=0.] = np.nan
        Te[Te<=0.] = np.nan
        ne[ne_unc>1e20] = np.nan
        ne[ne_unc/ne>2.] = np.nan
        Te[Te_unc>100] = np.nan
        Te[Te_unc/Te>2.] = np.nan
        
        # set arbitrary (to be revised!) minimum uncertainties
        ne_unc[ne_unc<1e18] = 1e18
        Te_unc[Te_unc<1.0] = 1.0

        # find measurement points relative to x-point
        txp, Rxp, Zxp = self.get_xpnt_pos(t0,t1)

        # get (R,Z) coordinates of every ne,Te measurement relative to X-point
        Rxp_vta = interp1d(txp, Rxp)(t_vta)
        Zxp_vta = interp1d(txp, Zxp)(t_vta)

        # take average x-point as reference
        DeltaR = Rxp_vta - np.mean(Rxp_vta)
        DeltaZ = Zxp_vta - np.mean(Zxp_vta)

        # -ve sign because relative motion of diags relative x-point
        # i.e. if x-point goes up, it's like diags go down
        R_vta_corr = R_vta[None]-DeltaR[:,None] 
        Z_vta_corr = Z_vta[None]-DeltaZ[:,None]
        
        rhop = sf.rz2rho(self.equ, R_vta_corr, Z_vta_corr, t_in = t_vta, coord_out = 'rho_pol')

        core_mask = (rhop>1.0) * (Z_vta_corr>Zxp_vta[:,None])
        
        # collect data in output dictionary
        self.data_expt = {'R': R_vta_corr, 'Z': Z_vta_corr, 'rhop': rhop,
                          'quants': ['ne','Te'],
                          'plot_coord': 'Z',
                          'ne': ne, 'ne_unc': ne_unc,
                          'Te': Te, 'Te_unc': Te_unc,
                          'core_mask': core_mask,
                          'type': 'local',
                          'w' : 1.0 # diagnostic weight, modifiable
        }

    def plot_data(self):
        '''Plot experimental data.
        '''
        data = self.data_expt
        fig, ax = plt.subplots()
        ax.errorbar(data['rhop'].flatten(), data['ne'].flatten()/1e19,
                    data['ne_unc'].flatten()/1e19, fmt='.', label='ETS')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$]')

        fig, ax = plt.subplots()
        ax.errorbar(data['rhop'].flatten(), data['Te'].flatten(),
                    data['Te_unc'].flatten(), fmt='.', label='ETS')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$T_e$ [$eV$]')

        fig, ax = plt.subplots()
        self.plot_machine(ax)
        ax.scatter(data['R'], data['Z'], s=20, c='k')

    def setup_synth_diags(self):
        ''' Store the oedge_case with loaded results.
        '''
        super(ets, self).setup_synth_diags_base(oedge_label='osm')

    def eval_synth(self, oedge_case):
        '''Evaluate synthetic diagnostic.
        '''
        return self.eval_local_synth(oedge_case)

    def plot_comparison(self):
        '''Call inhereted method to plot local comparison.
        '''
        self.plot_local_comparison()


class cdm(aug_diag):
    '''Load CDM high-resolution spectrometer estimates for 
    ion temperature (usually from C, but could also be other species).
    '''
    def __init__(self, shot, time_s):
        super(cdm, self).__init__(shot, time_s, 'cdm', equ)
        
        self.sf_obj = sf.SFREAD(self.shot, 'CDM')
        if not self.sf_obj.status:
            print(f'Could not load CDM for shot={shot}')
            return

    def load_data(self, t0, t1, varname='TC_2_465', dt_av=0.1):
        '''
        Parameters
        ----------
        t0,t1 : floats
            Beginning and end of time averaging window
        varname : str
            Name of the temperature variable to be fetched.
            Default is for the C2+ temperature from a line at 465nm.
        dt_av : float
            Time interval width over which data should be averaged, i.e.
            desired time resolution. Units of [s].
        '''
        super(cdm, self).load_data(t0, t1)
        self.varname = str(varname)
        self.dt_av = float(dt_av)

        _t_cdm = self.sf_obj.gettimebase(varname)
        tind = slice(*_t_cdm.searchsorted([t0,t1]))
        t_cdm = _t_cdm[tind]

        # now fetch temperature for the 3 chords
        Tc = self.sf_obj.getobject(varname)[tind]  # eV

        # reduce time resolution, averaging within 50 ms time windows
        dt_av = 0.05 # s
        gg = int(dt_av//np.mean(np.diff(t_cdm))) # number of time points to group
        num = len(t_cdm)//gg # reduce time res from 1 to 50 ms
        t_cdm = np.nanmean(t_cdm[:gg*num].reshape(-1,gg), axis=1) # use mean
        Tc = np.nanmean(Tc[:gg*num].reshape(-1,gg,Tc.shape[1]),axis=1) # use mean
        
        # lines of sight
        _los = self.sf_obj.getparset('LOS')
        num = len(np.nonzero(_los['R1'])[0])
        los_names = [_los[f'CHAN_{i:02d}'][0].decode("utf-8").strip()\
                     for i in np.arange(1,num+1)]
        
        # find measurement points relative to x-point
        txp, Rxp, Zxp = self.get_xpnt_pos(t0,t1)

        # get (R,Z) coordinates of every ne,Te measurement relative to X-point
        Rxp_cdm = interp1d(txp, Rxp)(t_cdm)
        Zxp_cdm = interp1d(txp, Zxp)(t_cdm)

        # take average x-point as reference
        DeltaR = Rxp_cdm - np.mean(Rxp_cdm)
        DeltaZ = Zxp_cdm - np.mean(Zxp_cdm)

        # populate data dictionary
        self.data = {'los_names': los_names, # redundant because of pnt1
                     'time': t_cdm,
                     'quant': ['Ti'],
                     'sig': Tc,
                     'sig_unc': Tc*0.2, # arbitrarily assume 20% uncertainty
                     'w' : 1.0, # diagnostic weight
                     'pnt1': {}, 'pnt2': {},
                     'type': 'line-average'
        }            
                
        # LOS as moving in time as x-point shifts vertically
        for los in los_names:
            R1, phi1, Z1, R2, phi2, Z2 = read_los_coord(
                self.shot, los, to_radian=True, get_cartesian=False)      

            # shift LOS positions relative to x-point
            # If x-point goes up, it's like diags go down, so -ve sign
            R1_corr = R1-DeltaR; Z1_corr = Z1-DeltaZ
            R2_corr = R2-DeltaR; Z2_corr = Z2-DeltaZ

            # store start & end points in cartesian coordinates, needed for line-integrals
            pnt1 = np.array([R1_corr * np.cos(phi1), R1_corr * np.sin(phi1), Z1_corr])
            pnt2 = np.array([R2_corr * np.cos(phi2), R2_corr * np.sin(phi2), Z2_corr])
            
            self.data_expt['pnt1'][los] = pnt1
            self.data_expt['pnt2'][los] = pnt2

    def plot_data(self):
        '''Plot Ti as a function of time for the 3 diagnostics.
        '''
        fig, ax = plt.subplots()
        for ss,los_name in enumerate(self.data_expt['los_names']):
            ax.plot(self.data_expt['time'], self.data_expt['sig'][:,ss])
        ax.set_xlabel('time [s]')
        ax.set_ylabel(r'$T_i$ [eV]')

        # show lines of sights as they shift in time
        cols = cm.rainbow(np.linspace(0, 1, len(los_names)))
        for l,los in enumerate(los_names):
            fig,ax = plt.subplots()
            self.plot_machine(ax)
            pnt1 = self.data_expt['pnt1'][los]
            pnt2 = self.data_expt['pnt2'][los]

            for tidx in np.arange(pnt1.shape[1]):
                # Similar to what's in solps.get_3d_path:
                xyz = np.outer(pnt2[:,tidx] - pnt1[:,tidx],
                               np.linspace(0, 1.0, int(101))) + pnt1[:, [tidx]]
                pathR, pathZ = np.hypot(xyz[0], xyz[1]), xyz[2]
                ax.plot(pathR, pathZ, c=cols[l], ls='-')

    def setup_synth_diags(self):
        ''' Store the oedge_case with loaded results.
        '''
        super(cdm,self).setup_synth_diags_base(oedge_label='osm')

    def eval_synth(self, oedge_case):
        '''Evaluate synthetic diagnostics.
        '''
        return self.eval_line_ave_synth(oedge_case)

    def plot_comparison(self):
        # use inhereted method
        self.plot_line_ave_comparison()


class lin(aug_diag):
    '''Lithium beam (LIN).
    '''
    def __init__(self, shot):
        super(lin, self).__init__(shot, time_s, 'lin', equ)

        self.sf_obj = sf.SFREAD(self.shot, 'lin')
        if not self.sf_obj.status:
            print(f'Could not load LIN for shot={self.shot}')
            return        
        
    def load_data(self, t0, t1):
        '''Load lithium beam (LIN) data.
        
        Parameters
        ----------
        t0,t1 : floats
            Beginning and end of time averaging window
        '''
        super(lin, self).load_data(t0,t1)

        _t_lin = self.sf_obj.gettimebase('time')
        tind = slice(*_t_lin.searchsorted([t0,t1]))
        t_lin = _t_lin[tind]
        
        ne_lin = copy.deepcopy(self.sf_obj.getobject('ne')[:,tind]).T
        ne_lin_unc = copy.deepcopy(self.sf_obj.getobject('ne_unc')[:,tind]).T

        # reduce time resolution, averaging within 50 ms time windows
        gg = 50
        num = len(t_lin)//gg # reduce time res from 1 to 50 ms
        t_lin = np.nanmedian(t_lin[:gg*num].reshape(-1,gg), axis=1)
        ne_lin = np.nanmedian(ne_lin[:gg*num].reshape(-1,gg,ne_lin.shape[1]),axis=1)
        ne_lin_unc = np.nanmedian(ne_lin_unc[:gg*num].reshape(-1,gg,ne_lin_unc.shape[1]),axis=1)
        
        # exclude bad points
        ne_lin[ne_lin<=0.] = np.nan
        ne_lin[ne_lin_unc>1e20] = np.nan
        ne_lin[ne_lin_unc/ne_lin>2.] = np.nan
        ne_lin_unc[ne_lin_unc<1e18] = 1e18

        # map coordinate relative to x-point
        R_lin = self.sf_obj.getobject('R')[:,0]
        Z_lin = self.sf_obj.getobject('Z')[:,0]

        # find measurement points relative to x-point
        txp, Rxp, Zxp = self.get_xpnt_pos(t0,t1)

        # get (R,Z) coordinates of every ne,Te measurement relative to X-point
        Rxp_lin = interp1d(txp, Rxp, bounds_error=False, fill_value='extrapolate')(t_lin)
        Zxp_lin = interp1d(txp, Zxp, bounds_error=False, fill_value='extrapolate')(t_lin)

        # take average x-point as reference
        DeltaR = Rxp_lin - np.mean(Rxp_lin)
        DeltaZ = Zxp_lin - np.mean(Zxp_lin)

        # -ve sign because relative motion of diags relative x-point
        # i.e. if x-point goes up, it's like diags go down        
        R_lin_corr = R_lin[None]-DeltaR[:,None]
        Z_lin_corr = Z_lin[None]-DeltaZ[:,None]

        rhop = sf.rz2rho(self.equ, R_lin_corr, Z_lin_corr, t_in=t_lin, coord_out='rho_pol')
        
        core_mask = (rhop>1.0) * (Z_lin_corr>Zxp_lin[:,None])
        
        self.data_expt = {'R': R_lin_corr, 'Z': Z_lin_corr, 'rhop': rhop,
                          'plot_coord': 'Z',
                          'quants': ['ne'],
                          'ne': ne_lin, 'ne_unc': ne_lin_unc,
                          'core_mask': core_mask,
                          'type': 'local',
                          'w' : 1.0 # diagnostic weight, modifiable
        }

    def plot_data(self):
        fig, ax = plt.subplots()
        ax.errorbar(rhop.flatten(), ne_lin.flatten()/1e19, ne_lin_unc.flatten()/1e19, fmt='.', label='LB')
        ax.set_xlabel(r'$\rho_p$')
        ax.set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$]')
        
        fig, ax = plt.subplots()
        self.plot_machine(ax)
        ax.scatter(R_lin_corr.flatten(), Z_lin_corr.flatten(), s=20, c='k')

    def setup_synth_diags(self):
        ''' Store the oedge_case with loaded results.
        '''
        super(lin,self).setup_synth_diags_base(oedge_label='osm')

    def eval_synth(self, oedge_case):
        '''Evaluate synthetic diagnostics.
        '''
        return self.eval_local_synth(oedge_case)

    def plot_comparison(self):
        # use inhereted method
        self.plot_local_comparison()


class dts(aug_diag):
    '''Divertor Thomson Scattering.
    '''
    def __init__(self, shot, time_s):
        super(dts, self).__init__(shot, time_s, 'dts', equ)

        self.sf_obj = sf.SFREAD(self.shot, 'dtn')
        if not self.sf_obj.status:
            print(f'Could not load DTN for shot={shot}')
            return
        
    def load_data(self, t0, t1):
        '''
        Note that DTS measurements are taken to be at locations
        relative to the mean x-point location during the chosen
        time window.
        
        Parameters
        ----------
        t0,t1 : floats
            Beginning and end of time averaging window
        '''
        super(dts, self).load_data(t0,t1)

        txp, Rxp, Zxp = self.get_xpnt_pos(t0,t1)
        
        # now load DTN data with position relative to the x-point
        _t_dtn = self.sf_obj.gettimebase('Te_ld')
        tind = slice(*_t_dtn.searchsorted([t0,t1]))
        t_dtn = _t_dtn[tind]
        
        ne_dtn = copy.deepcopy(self.sf_obj.getobject('Ne_ld')[tind])
        ne_dtn_unc = copy.deepcopy(self.sf_obj.getobject('SigNe_ld')[tind])
        Te_dtn = copy.deepcopy(self.sf_obj.getobject('Te_ld')[tind])
        Te_dtn_unc = copy.deepcopy(self.sf_obj.getobject('SigTe_ld')[tind])

        # eliminate outliers (TODO: find better criteria)
        #ne_dtn[ne_dtn>1e21] = np.nan
        #Te_dtn[Te_dtn>np.quantile(Te_dtn,0.90)] = np.nan

        # apply median filter to data images to eliminate outliers
        median_filter(ne_dtn, size=5, output=ne_dtn, mode='nearest')
        median_filter(Te_dtn, size=5, output=Te_dtn, mode='nearest')
        
        # R and Z are indpt of time, but we want them for every time point
        R_dtn = self.sf_obj.getobject('R_ld')
        Z_dtn = self.sf_obj.getobject('Z_ld')

        # get (R,Z) coordinates of every ne,Te measurement relative to X-point
        Rxp_dtn = interp1d(txp, Rxp)(t_dtn)
        Zxp_dtn = interp1d(txp, Zxp)(t_dtn)

        # take average x-point as reference
        DeltaR = Rxp_dtn - np.mean(Rxp_dtn)
        DeltaZ = Zxp_dtn - np.mean(Zxp_dtn)

        # -ve sign because relative motion of diags relative x-point
        # i.e. if x-point goes up, it's like diags go down        
        R_dtn_corr = R_dtn[None]-DeltaR[:,None]
        Z_dtn_corr = Z_dtn[None]-DeltaZ[:,None]

        # substitute all data points that are exactly 0 with nan
        ne_dtn[ne_dtn==0.] = np.nan
        Te_dtn[Te_dtn==0.] = np.nan

        # set arbitrary (to be revised!) minimum uncertainties
        ne_dtn_unc[ne_dtn_unc<1e18] = 1e18
        Te_dtn_unc[Te_dtn_unc<1.0] = 1.0

        # collect data in output dictionary
        self.data_expt = {'R': R_dtn_corr, 'Z': Z_dtn_corr,
                          'plot_coord': 'R',
                          'type': 'local',
                          'quants': ['ne','Te'],
                          'ne': ne_dtn, 'ne_unc': ne_dtn_unc,
                          'Te': Te_dtn, 'Te_unc': Te_dtn_unc,
                          'w' : 1.0 # diagnostic weight, modifiable
        }
        
    def plot_data(self):
        
        # interpolate for better visualization in 2D
        Rgrid = np.linspace(np.min(self.data_expt['R']), np.max(self.data_expt['R']), 110)
        Zgrid = np.linspace(np.min(self.data_expt['Z']), np.max(self.data_expt['Z']), 100)
        
        # mesh on which data will be interpolated, if within the covered spatial domain
        Rmesh = np.tile(Rgrid, (len(Zgrid),1))
        Zmesh = np.tile(Zgrid, (len(Rgrid),1))
        
        ne_out = griddata((self.data_expt['R'].flatten(), self.data_expt['Z'].flatten()),
                          self.data_expt['ne'].flatten(), (Rmesh.T, Zmesh), method='linear')
        
        Te_out = griddata((self.data_expt['R'].flatten(), self.data_expt['Z'].flatten()),
                          self.data_expt['Te'].flatten(), (Rmesh.T, Zmesh), method='linear')
        
        # now plot ne first, Te second
        fig,ax = plt.subplots()
        ctr = ax.contourf(Rgrid, Zgrid, ne_out.T)
        fig.colorbar(ctr)
        ax.scatter(self.data_expt['R'].flatten(), self.data_expt['Z'].flatten(), s=5, c='k')
        self.plot_machine(ax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        
        fig,ax = plt.subplots()
        ctr = ax.contourf(Rgrid, Zgrid, Te_out.T)
        fig.colorbar(ctr)
        plt.scatter(self.data_expt['R'].flatten(), self.data_expt['Z'].flatten(), s=5, c='k')
        self.plot_machine(ax)
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        
        # plot locations of DTS measurements
        # fig,ax = plt.subplots(num='DTS',figsize=(7,7))
        # self.plot_machine(ax)
        # #p = mpl.collections.PolyCollection(
        # #    oedge_case.output.mesh,
        # #    edgecolors='k', fc='w', linewidth=0.1)
        # #ax.add_collection(p)
        # ax.plot(R_dtn, Z_dtn, 'ro', label='DTS measurement points')
        # ax.legend(loc='best').set_draggable(True)
        # plt.axis('equal')

    def setup_synth_diags(self):
        ''' Store the oedge_case with loaded results.
        '''
        super(dts,self).setup_synth_diags_base(oedge_label='osm')

    def eval_synth(self, oedge_case):
        '''Evaluate synthetic diagnostics.
        '''
        return self.eval_local_synth(oedge_case)

    def plot_comparison(self):
        # use inhereted method
        self.plot_local_comparison()


class div_balmer(aug_diag):
    def __init__(self, shot, time_s):
        super(div_balmer, self).__init__(shot, time_s, 'div_balmer', equ)

        self.xvl = {'evl': sf.SFREAD(self.shot, 'EVL'),
                    'fvl': sf.SFREAD(self.shot, 'FVL'),
                    'gvl': sf.SFREAD(self.shot, 'GVL')}

        for xx in self.xvl:
            if not self.xvl[xx].status:
                print(f'Could not load {xx.upper()} for shot={shot}')
                return

    def load_data(self, t0, t1):
        '''Load the Balmer series data from divertor spectroscopy.

        Parameters
        ----------
        t0,t1 : floats
            Beginning and end of time averaging window
        '''
        super(div_balmer, self).load_data(t0,t1)

        # 'D_0_6561': 'D-alpha', 'D_0_4860': 'D-beta',
        # 'D_0_4339': 'D-gamma', 'D_0_4101': 'D-delta',
        # 'D_0_3969': 'D-epsilon', 'D_0_3888': 'D-zi',
        # 'D_0_3834': 'D-nu'
            
        for xx in ['evl','fvl','gvl']:

            xvl = self.xvl[xx]
            time = xvl.getobject('TIME')
            _los = xvl.getparset('LOS')

            # select time range
            tind = slice(*time.searchsorted([t0,t1]))

            # select names of used LOS
            num = len(np.nonzero(_los['R1'])[0])
            los = [_los[f'CHAN_{i:02d}'][0].decode("utf-8").strip()\
                   for i in np.arange(1,num+1)]

            # fetch balmer lines
            for line in xvl.objects:
                if line.startswith('D_0'):
                    self.data_expt[line] = {
                        'los_names': los,
                        'type': 'line-int',
                        'sig': np.mean(xvl.getobject(line)[tind],axis=0)
                    }
                    
                    # arbitrarily assume 10% uncertainty
                    self.data_expt[line]['sig_unc'] = 0.1 * self.data_expt[line]['sig']

                # add default diagnostic weight
                self.data_expt[line]['w'] = 1.

    def plot_data(self):

        fig,ax = plt.subplots()

        ax.semilogy(np.arange(num), self.data_expt[line]['sig'], '.', label=line)
        
        # pick one of the D-alpha signals
        line0 = [diag for diag in self.data.keys() if diag.startswith('D_0')][0]
            
        ax.set_xticks(np.arange(num))
        ax.set_xticklabels(self.data_expt[line0]['los_names'], rotation='vertical')
        ax.set_ylabel(r'Signal [phot/m$^2$/sr/s]')
        ax.legend(loc='best').set_draggable(True)
        plt.tight_layout()

    def setup_synth_diags(self):
        '''Setup synthetic diagnostics for Balmer series measurements, 
        assuming the usual splitter setup on AUG.
        '''
        super(div_balmer,self).setup_synth_diags_base(oedge_label='osm')

        # set up only D-beta explictly
        diag = 'D_0_4860'
        synth_diag = self.synth_diags[diag] = {}

        for ll,los in enumerate(self.data_expt[diag]['los_names']):
            synth_diag[los] = divspec_main.synth_los_spectrum(
                self.shot, self.time_s, los=los)

            if ll==0:
                los0 = los
                # load OEDGE case only once
                synth_diag[los].load_oedge_case(imp='H')
            else:
                # copy OEDGE case to all other LOS (avoid repeated access to disk)
                synth_diag[los].model = 'oedge'
                synth_diag[los].imp = 'H'
                synth_diag[los].oedge = copy.deepcopy(synth_diag[los0].oedge)
                
                # update line integration for current LOS
                synth_diag[los].evaluate_sim_los()

        # now, copy synthetic diag model to all other Balmer lines
        for new_diag in self.data:
            if new_diag.startswith('D_0'):
                # use pointers, no deepcopy
                self.synth_diags[new_diag] = self.synth_diags[diag]

    def eval_synth(self, LS_rescale=False):
        '''Evaluate synthetic diagnostics.
        '''
        return self.eval_line_int_synth()

    def plot_comparison(self):
        # use inhereted method
        super(div_balmer, self).plot_line_int_comparison()




class aug_expt_comp:
    '''Class to compare 2D edge modeling results to AUG experimental data.

    Data from a number of experimental diagnostics are organized in such a 
    way to offer a standardization of local and line-integrated signals.
    '''
    def __init__(self, shot=38996, time_s=3.):
        self.shot = int(shot)
        self.time_s = float(time_s)

        self.equ = sf.EQU(self.shot, diag='EQI')
        
        self.data = {}

    def load_diags(self, t0, t1, diag_names=['ets','lin','dtn','cdm','div_balmer']):

        self.diag_names = diag_names

        self.diags = {}
        self.data_expt = {}

        if 'ets' in diag_names:
            self.diags['ets'] = ets(self.shot, self.time_s, equ=self.equ)
        if 'lin' in diag_names:
            self.diags['lin'] = lin(self.shot, self.time_s, equ=self.equ)
        if 'dtn' in diag_names:
            self.diags['dtn'] = dtn(self.shot, self.time_s, equ=self.equ)
        if 'cdm' in diag_names:
            self.diags['cdm'] = cdm(self.shot, self.time_s, equ=self.equ)
        if 'div_balmer' in diag_names:
            self.diags['div_balmer'] = div_balmer(self.shot, self.time_s, equ=self.equ)
            
        for diag_name in diag_names:
            # initialize each diagnostic class
            #exec(f'self.diags[{diag_name}] = {diag}({self.shot})')

            # load data in chosen time interval for each class
            self.diags[diag_name].load_data(t0, t1)

            # store in `data_exp` all info characterizing experimental measurement
            self.data_expt[diag_name] = self.diags[diag_name].data_expt

    def setup_synth_models(self, oedge_label='osm'):
        '''Set up synthetic models for all loaded experimental measurements.
        '''
        self.synth_diags = {}
        for diag_name in self.diag_names:
            # set up synthetic diagnostics for each loaded diagnostic
            self.diags[diag_name].setup_synth_diags()

            # store in `synth_diags` the synthetic models for all measurements
            #for meas  in self.diags.synth_diags:
            #    # overwrites "oedge_case" such that only 1 instance remains
            #    self.synth_diags[meas] = self.diags[diag_name].synth_diags[meas]
            
    def sig2lnlike(self):
        '''Compute log-likelihood for a simulation model and all loaded 
        experimental diagnostics.

        Diagnostics are separated into 3 categories:
        
        #. Local, e.g. ne measurements from Thomson scattering;
        #. Line-integrated, e.g. measurements of Balmer lines;
        #. Line-averaged, e.g. measurements of non-volumetric quantities
           that are averaged along a LOS 

        The weighted differences :math:`\chi` are given by
        
        .. math::
            
            \chi =  \frac{b_{model, i} - b_{expt, i}}{\sigma_i}
        
        Based on these, a number of likelihood models (Gaussian, Cauchy, etc.)
        are available.

        Parameters
        ----------
        LS_rescale : bool
            If True, rescale all simulated line-integrated signals by a factor 
            that minimizes the chi^2 with experimental data, using an analytic
            Least-Squares (LS) procedure. Default is False, i.e. consider the 
            absolute calibration of the diagnostic and attempt to match it.
        plot : bool
            If True, plot comparison between experimental data and OEDGE results.

        Returns
        -------
        lnlike : float
            Value of total log-likelihood for all the loaded diagnostics.
        '''
        ll = -np.infty
        
        for diag_name in self.diag_names:

            # add log-likelihood contribution
            ll += self.diags[diag_name].eval_synth()
                        
        return ll



if __name__=='__main__':

    shot = 38996
    time_s = 3. # time for simulation

    # time window for expt data
    t0 = 2.2
    t1 = 3.6

    # initialize class
    comp = aug_expt_comp(shot, time_s)

    comp.load_diags(t0, t1, diag_names=['ets']) #,'lin','dtn','cdm','div_balmer'])
    
    # set up synthetic diagnostics for all signals
    comp.setup_synth_models()

    #print('Completed setup!')
    
    # get normalized differences between expt and model
    #lnlike, chi_vec = comp.sig2lnlike(plot=True)
