''' Bayesian Impurity Transport & Spectroscopy (BITS) framework

Main components of the framework to infer D,V coefficients and uncertainties from Bayesian analysis of spectroscopic signals on Alcator C-Mod, using methods drawn from the computational statistics literature.

BITS builds on the predecessor BITE, which used individual XICS lines, while BITS uses the entire XICS spectrum for Ca ions.

F. Sciortino - 2021
'''
import sys, os
import numpy as np 
import scipy, math  # math useful for logs of arbitrary base
import scipy.interpolate
import scipy.optimize
import warnings
import MDSplus
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.gridspec as mplgs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import profiletools, eqtools
import re, copy
import pickle as pkl
import itertools
import time as time_
import shutil, multiprocessing
import emcee
from emcee.interruptible_pool import InterruptiblePool
import traceback, sobol
import xarray
from sklearn.preprocessing import PolynomialFeatures
from scipy import constants
from scipy.interpolate import interp1d,PchipInterpolator, interp2d, RectBivariateSpline, griddata
import nlopt
from scipy.constants import e as q_electron,k as k_B, h, m_p, c as c_speed
from scipy import stats

# BITS modules
import bits_helper
import bits_diags
import bits_utils
from bits_cmod_shots import get_bits_shot_info

from IPython import embed

import aurora
import pymultinest
import TRIPPy

from omfit_classes import omfit_eqdsk, omfit_gapy
from omfit_classes.omfit_mds import OMFITmdsValue

#np.seterr(all='print')

from numba import njit, prange

#@njit(parallel=True)
def get_spec_comps(lam_spec, ne_cm3_spec, lams_profs_A, theta, 
                   pec_ion, pec_exc, pec_rr, pec_dr):

    spec_ion = np.zeros((len(lam_spec), *ne_cm3_spec.shape))
    spec_exc = np.zeros((len(lam_spec), *ne_cm3_spec.shape))
    spec_rr = np.zeros((len(lam_spec), *ne_cm3_spec.shape))
    spec_dr = np.zeros((len(lam_spec), *ne_cm3_spec.shape))
    
    for il in prange(theta.shape[0]):  # spectral line
        for it in prange(spec_ion.shape[1]):  # time
            for ir in prange(spec_ion.shape[2]):   # radius
                _line_elem = np.interp(lam_spec, lams_profs_A[il,ir,:], ne_cm3_spec[it,ir]*theta[il,ir,:]) #, left=0, right=0)

                spec_ion[:,it,ir] += pec_ion[il,it,ir] * _line_elem
                spec_exc[:,it,ir] += pec_exc[il,it,ir] * _line_elem
                spec_rr[:,it,ir] += pec_rr[il,it,ir] * _line_elem
                spec_dr[:,it,ir] += pec_dr[il,it,ir] * _line_elem
                
    return spec_ion, spec_exc, spec_rr, spec_dr 


# Directory where BITS runs are created:
usr_name=os.path.expanduser('~')[6:]
usr_home=os.environ['BITS_BASE']+usr_name 

# POS vector for XEUS:
XEUS_POS = [2.561, 0.2158, 0.196, 0.1136]

# POS vector for LoWEUS:
LOWEUS_POS = [2.561, -0.2158, 0.196, -0.1136]


def tridiag(a,b,c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


# ==================================================
class Run:
    """Class to load and run BITS/aurora.
        
    If the directory bits_<SHOT>_<VERSION> does not exist, creates it.
    
    Parameters
    ----------
    shot : int
        The shot number to analyze.
    version : int
        The version of the analysis to perform. Default is 0.
    time_1 : float
        The start time of the simulation.
    time_2 : float
        The end time of the simulation.
    injections : list of object instantiations with attributes  't_inj','t_start','t_stop'
        Objects describing the injections.
    thaco_tht : int
        The THACO THT to use. Default is 0.
    debug_plots : int
        Set to 0 to suppress superfluous plots. Set to 1 to enable some plots.
        Set to 2 to enable all plots. Default is 0.
    nD : int
        The number of eigenvalues/free coefficients to use for the D profile.
    nV : int
        The number of eigenvalues/free coefficients to use for the V profile.
    roa_grid : array of float
        r/a grid to evaluate the ne, Te profiles on.
    roa_grid_DV : array of float
        r/a grid to evaluate the D, V profiles on.
    source_file : str
        If present, this is a path to a properly-formatted source file to use
        instead of the source model. This overrides the other options related
        to sources (though note that source_prior still acts as the prior for
        the temporal shift applied to the data, and hence should be univariate).

    include_loweus : bool
        If True, the data from the LoWEUS spectrometer will be included in the
        likelihood. Otherwise, LoWEUS will only be evaluated to compare the
        brightness.
    free_time_shift : array of bools
        If True, a temporal shift will be applied to the diagnostic corresponding to
        the given index and will be included as a free parameter. 
        This can help deal with issues of synchronization.  Default is False for each 
        parameter (do not shift).

    SYNTHETIC DATA
    params_true : array of float or bool, (`num_params`,), optional
        If provided, these are used to construct synthetic data (using the
        equilibrium and temperature/density profiles from the specified shot).
        If set to True (boolean), a set of true parameters of the right length will be created internally,
        based on the provided explicit D & V. Default is to use actual experimental data.
    synth_li_lines : array of int, optional
        The indices of the Li-like Ca lines to use when constructing the
        synthetic data. The default is to use the ones normally seen on XEUS.
    synth_be_lines : array of int, optional
        The indices of the Be-like Ca lines to use when constructing the
        synthetic data. The default is to use the one normally seen on LoWEUS.
    hirex_time_res : float, optional
        The time resolution (in seconds) to use for synthetic HiReX-SR data.
        Default is 1e-3.
    vuv_time_res : float, optional 
        The time resolution (in seconds) to use for synthetic VUV data. Default
        is 1e-3.
    xtomo_time_res : float, optional
        The time resolution (in seconds) to use for XTOMO data. If synthetic data is being generated, 
        this time resolution is adopted; if real data is being analyzed, the data is averaged over time 
        windows with this effective time resolution. Default is 1e-4. Note that XTOMO's true time resolution
        is ~4e-6s.
    xtomo_time_res_decay : float
        Time resolution (in seconds) to use for XTOMO data (synthetic or real) in the decay phase, i.e. 
        xtomo_decay_phase_time s after the recorded time of LBO injection. Default is 5e-3, i.e. similar to XICS. 
    xtomo_decay_phase_time : float
        Length of time after LBO injection to consider for SXR data, in ms.
        If left to 0 (default), use the same time window as for other diagnostics.
    presampling_time : float, optional
        The time (in seconds) to keep from before `time_1` when generating
        synthetic data. Default is 15e-3.
    synth_noises : array of float, optional
        The relative noise levels to use with the synthetic data. Default is
        [0.03, 0.03, 0.1] for w line, VUV and SXR. If Hirex_min_rel_unc and XEUS_min_rel_unc
        are given, their values replace the XICS and VUV ones in synth_noises. 
    normalize : bool, optional
        If True, normalized signals will be used when comparing to aurora output.
        Default is True.
    time_spec : str, optional
        The time grid specification to use when writing the param file. Default
        is `TIME_SPEC_RD` (previously: `DEFAULT_TIME_SPEC`).
    initial_params : array of bool, (`num_params`,), optional
        Initial values for all of the parameters (both free and fixed). Default
        is to use `params_true` if it is provided, or a random draw from the
        prior distribution. This exists primarily as a mechanism to set the
        values of fixed parameters.


    BASIC D,V. V/DBOUNDS:
    D_lb : float, optional
        The lower bound on the diffusion coefficient parameters. Default is 0.01.
    D_mid_ub : float, optional
        The upper bound on the diffusion coefficient parameters. Default is 30.0.
    D_axis_ub: float, optional
        The upper bound on the diffusion coefficient parameters near axis. Default is 30.0.
    D_edge_ub : float, optional
        The upper bound on the diffusion coefficient parameters at the edge. Default is 30.0.
    --------
    V_axis_lb : float, optional
        The lower bound on V near axis.
    V_axis_ub : float, optional
        The upper bound on V near axis. 
    V_mid_lb : float, optional
        The lower bound on V at midradius.
    V_mid_ub : float, optional
        The upper bound on V at midradius.
    V_edge_lb : float, optional
        The lower bound on V at the edge.
    V_edge_ub : float, optional
        The upper bound on V at the edge.
    --------
    VoD_axis_lb : float, optional
        The lower bound on V/D near axis.
    VoD_axis_ub : float, optional
        The upper bound on V/D near axis. 
    VoD_mid_lb : float, optional
        The lower bound on V/D at midradius.
    VoD_mid_ub : float, optional
        The upper bound on V/D at midradius.
    VoD_edge_lb : float, optional
        The lower bound on V/D at the edge.
    VoD_edge_ub : float, optional
        The upper bound on V/D at the edge.
    ---------
    use_truncnorm_V : bool, opt
         Choose whether to use truncated Gaussians for V priors. This uses a uniform (possibly Jeffreys') prior
         for D. Default is True. Necessary parameters follow:
    ---------------
    
    noise_type: {'proportional Gaussian', 'Poisson'}
        The noise type to use. Options are:
        * 'proportional Gaussian': Gaussian noise for which the standard
          deviation is equal to the relative noise level times the value.
        * 'Poisson' : Gaussian noise for which the standard deviation is
          equal to the relative noise level times the value divided by the
          square root of the ratio of the value to the max value. This
          simulates Poisson noise.
    explicit_D : array of float, (`M`,), optional
        Explicit values of D to construct the truth data with. Overrides the D coefficient parts of `params_true`.
        *** If set to a boolean (either True or False), both explicit (true) D and V will be set based on a random draw
        from the prior. ***
    explicit_D_roa : array of float, (`M`,), optional
        The r/a grid the explicit values of D are given on.
    explicit_V : array of float, (`M`,), optional
        Explicit values of V to construct the truth data with. Overrides the V
        coefficient parts of `params_true`.
    explicit_V_roa : array of float, (`M`,), optional
        The r/a grid the explicit values of V are given on.


    SPLINE KNOTS
    method : {'linterp', 'akima'}
        The method to use when evaluating the D, V profiles.
        * If 'linterp', (the default) piecewise linear functions will be used
          for the D and V profiles. `nD` and `nV` will then be the
          number of free values which are then linearly-interpolated. Because
          there is a slope constraint on D and a value constraint on V, the
          total number of knots is `num_eig + 1`. But the first and last knots
          should usually be at the edges of the domain, so there are only
          `num_eig - 1` free knots.
        * If 'akima', Akima1DInterpolator splines are used. The call to this is the 
          same as for 'linterp'.
        * If 'pchip', PCHIP 1D interpolation is used.
    -----
    fixed_knots: array of floats, (num of free D,V coefficients), optional
        Fixed values for the knots. The number of fixed knots provided must be 
        equal to nD-1 + nV-1, i.e. equal to the number of non-trivial
        knots (i.e. knots at 0.0 and 1.05 do not need to be given). 
        If set to None, this option has no effect. Default is None.

    -----
    free_D_knots : bool
        If True, set D spline knots to be free. Default is False.
    free_V_knots : bool
        If True, set V spline knots to be free. Default is False.
    equal_DV_knots : bool
        If True, V spline knots are set equal to the D ones. Default is False.

    DIAGNOSTIC MEASUREMENTS
    signal_mask : array of bool, optional
        Indicates which synthetic signals should be generated/used. Order is:
        [HiReX-SR, VUV, XTOMO]. If passed when creating the run for the first
        time, only the indicated synthetic signals will be created. If passed
        when restoring a run, will simply remove the synthetic signals from the
        inference.
    Hirex_min_rel_unc : float, optional
        minimum relative uncertainty for normalized Hirex-Sr data.
    Hirex_min_abs_unc : float, optional
        minimum absolute uncertainty for normalized Hirex-Sr data. 
    Hirex_unc_multiplier : float, optional
        Multiplier for all Hirex-Sr experimental uncertainties. Default is 1.
    sig_rise_enforce : float, optional
        Reinforce importance of data points in the rise of the signal by dividing 
        the uncertainty of the first 3 of these (fixed, arbitrary number) by this factor. 
        Set to 1 to make this ineffective. Default is 1.0. 
    XEUS_min_rel_unc : float, optional 
        minimum relative uncertainty for normalized XEUS data.
    XEUS_min_abs_unc : float, optional
        minimum absolute uncertainty for normalized XEUS data.
    LoWEUS_min_rel_unc : float, optional 
        minimum relative uncertainty for normalized LoWEUS data.
    LoWEUS_min_abs_unc : float, optional
        minimum absolute uncertainty for normalized LoWEUS data.
    XTOMO_min_rel_unc : float
        minimum relative uncertainty for normalized XTOMO data.
    XTOMO_min_abs_unc : float
        minimum absolute uncertainty for normalized XTOMO data
    -----
    Hirex_time_shift : float, optional
        Add an explicit time shift to the Hirex-Sr time base, possibly avoiding free_time_shift and
        thus imposing a value that has already been inferred previously. Default is 0. 
    VUV_time_shift : float, optional
        Same as above, but for VUV (XEUS + LoWEUS). Default is 0.
    ----
    vuv_rel_calibration : bool, optional
          As above, but for VUV lines. Note that XEUS and LoWEUS are always considered separately, since their sensitivity differs, 
          so it is not possible to set a relative calibration between signals from the two detectors. 
          Default is False. 

    -----
     n0x : float, opt
         Multiplier for input deuterium neutral densities, used to estimate CX recombination contribution using 
         ADAS CCD rates.
    free_n0x : bool
         If True, set a scalar multiplier of the atomic neutral density as a free parameter. Default is False. 
         Note that this only has an effect if charge exchange is included (include_neutrals_cxr=True)

    DIAGNOSTIC WEIGHTS
    d_weights : list of float, (`n_sig`,), or list of arrays of float
        The weights to use for each non-masked signal in :py:attr:`self.signals`. Each entry in the outer list 
        applies to the corresponding entry in :py:attr:`self.signals`. Each entry can either be a single,
        global value to use for the diagnostic, or a list of values for each chord. If these are different 
        than a list of 1's, then some of the diagnostics are weighed comparatively more with respect to the others. 
        This is NTH's "chi2 bias", applied at the stage of computing the log-likelihood in diffs2ln_prob.
    free_d_weights : bool, optional
        Set diagnostic weights to be free. If True, the values in d_weights are ignored. 
        The weight for Hirex-Sr chords is set to 1 and other values are inferred relative to this.  
        Default is False. 
    CDI : integer, optional
        Combined Datasets Inference (CDI). Apply analysis of Hobson, Lahav et al. MNRAS 2000, 2002 
        to marginalize over weights for each diagnostic/dataset, using analytically-derived generalizations
        to the method. This gives a different log-likelihood, but no extra free parameters. 
        {-1: Cauchy likelihood; 0: do not apply; 1: apply with exponential prior on weights; 
        2: apply with general Gamma prior with parameters a and b; 
        3: apply with Gamma prior of mean 1 with single parameter nu; 
        4: same as 3, but only applied to diagnostics other than Hirex}.
        To use options 2, the parameter CDI_a and CDI_b must be provided. 
        To use option 3 (or 4), only the parameter CDI_a is needed (CDI_b is set equal to 1/CDI_a).
        Default is 0 -- apply simple chi^2 minimization.    
    CDI_a: float, optional
         Shape parameter for Gamma prior distribution of diagnostic weights, which is marginalized over
         analytically to obtain the log-likelihood expression. This is only used if CDI==2, 3 or 4. Default is 100. 
    CDI_b : float, optional
         Rate parameter for Gamma prior distribution of diagnostic weights, which is marginalized over
         analytically to obtain the log-likelihood expression. This is only used if CDI==2. If CDI==3 or 4, 
         CDI_b is automatically set to 1/CDI_a. Default is 0.01. 

    PMMCMC PARAMETERS
    use_PMMCMC_tshifts : bool, optional
        If True, use pseudo-marginal nested sampling to handle the time shift
        parameters. Default is False (either hold fixed or sample full
        posterior).
    num_pts_PMMCMC_tshifts : int, optional
        Number of points to use when evaluating the marginal distribution with
        pseudo-marginal nested sampling. Roughly-speaking, this defines the resolution 
        with which the time shift should be inferred, based on the width of the time shift
        prior for each diagnostic. Default is 50.
    method_PMMCMC_tshifts : {'QMC', 'GHQ'}, optional
        Method to use when computing the marginal distribution with
        pseudo-marginal nested sampling. Default is 'QMC' (use quasi Monte Carlo
        sampling with a Sobol sequence). The other valid option is 'GHQ' (use
        Gauss-Hermite quadrature with a dense tensor product grid).


    MULTINEST SAMPLING:
    n_live_points : int, optional
        Number of nested sampling live points. Default is 400.
    sampling_efficiency: float, optional
         Sampling efficiency for nested sampling. Default is 0.3.
    MN_INS : bool, optional
         Boolean indicating whether Importance Nested Sampling (INS) should be used in MultiNest. 
         Default is True. Note that multimodal discovery is not possible in MultiNest with INS. 
    MN_const_eff : bool, optional
        Run MultiNest in constant efficiency mode. Default is False. 
    MN_lnev_tol : float, optional 
        Tolerance in the log-evidence of MultiNest. This gives the termination condition for nested sampling. 
        Default is 0.5. 
    copula_corr : float, optional
        Off-diagonal correlation strength between adjacent variables that are coupled by the Gaussian copula. 
        This is only used if copula_corr>0. Default is 0.5. 
    non_separable_DV_priors : bool
        If True, set priors on D, V and V/D, rather than just on V and V/D. This requires that learn_VoD=True, so
        that sampling initially occurs on D and V/D parameters; a fixed V~N(0,15m/s) prior is then applied and combined 
        with the prior on V/D (assumed to be gaussian too). Since the product of gaussian priors is another gaussian, 
        we create a new gaussian that is the combination of the mean and std from the V/D prior and from the 
        effective prior on V after a sample in D. We then evaluate the inverse CDF of this combined gaussian prior 
        at the unit hypercube samples passed by MultiNest. 

    SAWTEETH:
    sawteeth_times: bool or array
        Times at which sawtooth crashes occur. This should be used together with `sawteeth_mix_rad'.
        All times should be given in units of seconds. On Alcator C-Mod, sawtooth crash times can be
        estimated using the XTOMO-3 array. If set to False, the aurora sawtooth option is not activated,
        i.e. densities don't get flattened at sawtooth crashes. 
    sawteeth_inv_rad: bool or float
        Radius (in units of cm, real geometry) at which the sawtooth inversion radius is judged to be. This is set to
        be the same for all the times given above. If set to False, the aurora sawtooth option is not
        activated, i.e. densities don't get flattened at sawtooth crashes.  
    sawteeth_inv_mix_factor: float
        Factor that relates the sawtooth inversion and mixing radius. Default is np.sqrt(3./2.).
        Seguin et al. PRL 1983 suggested np.sqrt(2), but this appears too large.      
    adapt_rsaw_flag : bool, float
        If True, adapt the sawtooth mixing radius proportionally to the amplitude of the temperature sawtooth crash. 
        If False, all sawteeth are set to have the same mixing radius. 
    free_mixing_radius : bool, optional
        Infer a correction to the mixing radius used for aurora's sawtooth flattening model, which is normally given by 
        sawteeth_mix_rad. Default is False. 
    sawtooth_width_cm: float, optional
          Width of sawtooth ERFC crash, passed to aurora. Note that this is taken to be in rV coordinates.
    free_sawtooth_width : bool, optional
          If True, the width of the sawtooth crash is set to be a free parameter. 

    aurora GRID:
    K_spatial_grid : float, optional
        Exponent for the spatial grid definition given in STRAHL's 2018 manual, Eq. (1.27). The larger K_spatial_grid, the more
        steep is the grid variation near the gradient. Default is 6.0.
    dr_center : float, optional
        spacing (in cm) at the center of the aurora grid. This parameter is used in the write_param method. 
        Default is 0.2.
    dr_edge : float, optional
        Spacing (in cm) at the edge of the aurora grid. Set this to a low value (<1mm) for cases with very strong
        pedestal transport. Default is 0.1.
    decay_length_boundary : float, optional [cm]
        Decay length of impurity density at the last grid point. If >0, the flux is given by D_{edge} n_{z,edge}/\lam, 
        where lam is the decay length. If <0, lam is set to \sqrt{D_{edge} \tau_{||, edge}}, where \tau is the parallel
        loss time constant that is internally calculated. Default is 0.2 cm. 


    force_identifiability : bool, optional
         If True, force identifiability of posterior by mapping all unit-hypercube samples to a hyper-triangle, 
         following the method described in Handley et al. MNRAS 453, 4384-4398 (2015)
    min_knots_dist : float
         Minimum distance to be imposed between knots in the force_identifiability=True mode. 
         Default is 0.1.


    PRIOR BOUNDS ON D,V
    ped_roa_DV : float, optional
        Minimal radial location (in r/a units) of the region to be considered pedestal. This value is used to set different 
        priors for core and pedestal D,V. Default is r/a=0.95. 
    nearaxis_roa : float, optional
        Maximal radial location (in r/a units) of the region to be considered as "near-axis". This value is used to set different
        prior for core and pedestal D,V. If set to True, this is automatically set to be the sawtooth mixing radius, as defined 
        by the variable sawteeth_mix_rad (which is given in cm). If set to False, this variable is internally set to 0.
        Default is True.  (alternatively, value of r/a=0.25 is suggested).
    D_prior_dist : str
        Type of prior distribution on D. Choice of {'uniform','loguniform','truncnorm'}. Default is loguniform, but with non_separable_DV_priors
        a gaussian may be more appropriate.

    fix_D_to_tglf_neo : bool
         If True, fix parameters for D to the values given by NEO and TGLF modeling. Default is False.
    fix_V_to_tglf_neo : bool
         If True, fix parameters for V to the values given by NEO and TGLF modeling. Default is False.

    # PEDESTAL PARAMETERS
    learn_VoD: bool, optional 
        Make inferences on D and V/D, rather than D and V. All nomenclature for V/D still refers to V. 
        This option allows one to set prior bounds to V/D rather than V. Parameters are still read and interpolated 
        by the method eval_DV, but this option makes the function take D and V/D parameters and output
        D and V profiles if activated. All other functions behave the same. 
        Default is False.

    -------
    use_gaussian_Dped : bool, optional
          If True, add a Gaussian diffusion profile in the pedestal on top of the spline-derived convection profile. 
          This depends on 3 parameters, specified below.
    gaussian_D_amp : float
          Amplitude of Gaussian pedestal. Only used if use_gaussian_Dped is True. 
    gaussian_D_w : float
          Width of Gaussian D pedestal. Only used if use_gaussian_Dped is True. 
    gaussian_D_r : float
          Radial location of Gaussian D pedestal. Only used if use_gaussian_Dped is True. 
    free_gaussian_Dped : list or array of bools, (3,)
          Each element of this list sets whether correspondent parameter for gaussian pedestal is set to be free. Order of parameters is
          [gaussian_D_amp, gaussian_D_w, gaussian_D_r]. Default is [False, False, False]. If the user passes only 1 boolean, it is taken to 
          be the choice for all the 3 parameters. 
    use_gaussian_Vped : bool, optional
          If True, add a Gaussian inward pinch on top of the spline-derived convection profile. 
          This depends on 3 parameters, specified below.
    gaussian_V_amp : float
          Amplitude of Gaussian pedestal. Only used if use_gaussian_Vped is True. 
          NB: it is expected that the true sign of the convection is given, i.e. to give a pinch (inward convection), 
          gaussian_V_amp must be -ve. 
    gaussian_V_w : float
          Width of Gaussian pedestal. Only used if use_gaussian_Vped is True. 
    gaussian_V_r : float
          Radial location of Gaussian pedestal. Only used if use_gaussian_Vped is True. 
    free_gaussian_Vped : list or array of bools, (3,)
          Each element of this list sets whether correspondent parameter for gaussian pedestal is set to be free. Order of parameters is
          [gaussian_V_amp, gaussian_V_w, gaussian_V_r]. Default is [False, False, False]. If the user passes only 1 boolean, it is taken to 
          be the choice for all the 3 parameters. 
    scale_gaussian_Dped_by_Z : bool, opt
         If True, the gaussian D feature in the pedestal (only used if use_gaussian_Vped=True) is scaled by Z for each charge state.
    scale_gaussian_Vped_by_Z : bool, opt
          If True, the gaussian V feature in the pedestal (only used if use_gaussian_Vped=True) is scaled by Z for each charge state.
    
    ----
    couple_D_V_gauss_params : bool, opt
          If True, set D and V gaussian priors to be coupled such that they have the same width and central location, 
          but different amplitude. 
    #fix_final_V_knot_value : bool, opt
          If use_gaussian_Vped=True, this flag allows one to fix the final V knot value to 0.0, such that the V values at the edge are solely
          determined by the gaussian V profile.
    #fix_final_V_knot_value : bool, opt
          If use_gaussian_Dped=True, this flag allows one to fix the final D knot value to D_lb, such that the D values at the edge are mostly
          determined by the gaussian D profile and the D at the LCFS can be effectively very small.

    OTHER PARAMETERS
    print_locals : bool, optional
        Boolean indicating whether a dictionary containing all the arguments passed to this 
        function should be printed out. Useful for keeping track of runs in a logbook. 
    Te_scale : float, optional
          Scaling factor applied to the entire Te profile to test sensitivity of atomic data. Default is 1.0.
    check_particle_conservation: bool, opt
         If True, the results from every aurora iteration are checked to make sure that a reasonable fraction of 
         particles entered the plasma. 
    min_penetration_fraction: float, opt
        Minimum fraction of particles that are estimated to enter the plasma from the LBO injection. 
        Rice et al. At.Mol.Opt.Phys. 28-893 1995 estimated ~10% of Sc particles from LBO entered C-Mod. 
        The fraction for Ar puffing was thought to be significantly smaller (5%?)
        Estimates from LBO injections on DIII-D suggest higher fractions for LBO injections (>25%).
        Default is to conservatively set this parameter to a few percent. 
    max_p_conserv_loss : float, opt
         Maximum fraction of particles that may be "lost" in a aurora simulation. If a greater fraction of the particles are
         lost in a simulation, the log-likelihood is set to -inf, effectively preventing the inference from converging to 
         solutions that do not conserve particle number. 
    apply_CF_corr : bool, opt
         If True, apply centrifugal (CF) correction to diagnostic weights.
    apply_superstaging : bool, 
         If True, avoid simulating transport of selected low charge states [1-9] to save computing time.

    SIGNALS CORRELATION
    gauss_corr_coeff : float
          Correlation coefficient applied to the likelihood, multiplying the chi2. 
          Default is 0.0, for which no correlation is applied.

    EDGE/RECYCLING
    rcl : float
        Recycling coefficient [0,1]. If outside of the [0,1] range, it is not used. Default is -1.
    tau_divsol : float [ms]
         Timescale for recycling from the divertor to the SOL reservoir. Default is 50 ms. 
    tau_pump : float [ms]
         Time scale for pumping out of impurities. Default is 1000 ms (i.e. almost no pumping).
    tau_rcl_ret : float [ms]
         Time scale for wall retention and release during recycling. Default is 10 ms. 
    neutrals_file : str
         Pickle file to load edge atomic neutral deuterium profiles from. These are expected to be derived from 
         diagnostic meausurements (e.g. from Ly-alpha measurements) or from neutrals modeling (e.g. via KN1D). 

    """
    def __init__(
            self,
            shot=0,
            version=0,
            time_1=0.0,
            time_2=0.0,
            injections=[],
            thaco_tht=0,
            debug_plots=0,
            nD=5,
            nV=5,
            source_file=None,
            method='linterp',
            free_D_knots=False,
            free_V_knots=False,
            equal_DV_knots=True,
            include_loweus=False,
            LS_diag_rescaling = True, 
            use_LS_alpha_only = True,
            free_time_shift=[False,False,False],
            #### synthetic model parameters #####
            params_true=None,
            synth_li_lines=[38,54,61,],
            synth_be_lines=[33,38,66],
            hirex_time_res=6e-3, #1e-3,
            vuv_time_res=2e-3, #1e-3,
            xtomo_time_res=1e-4, # lower to 0.1ms to avoid too data #2e-6,
            xtomo_time_res_decay=5e-3, # time resolution to set in decay phase
            xtomo_decay_phase_time=3e-3, # s after which "decay phase" of XTOMO is taken to start
            presampling_time=15e-3,
            synth_noises=[0.05, 0.05, 0.1],   # NOT used if min rel and abs uncertainties are given
            ######### ---------------------------------------###############
            normalize=True,
            initial_params=None,
            fixed_knots=None, 
            ###  D prior bounds
            D_lb=0.0,
            D_mid_ub=10.0, 
            D_edge_ub=None,  
            D_axis_ub=None,  
            ###  V prior bounds
            V_axis_mean = 0.0,
            V_axis_std = +15.0,
            V_axis_lb = -50,
            V_axis_ub = +50,
            V_mid_mean = 0.0,
            V_mid_std = +15.0,
            V_mid_lb = -50.0,
            V_mid_ub =+50.0,
            V_edge_mean = 0.0,
            V_edge_std = +50.0,
            V_edge_lb = -150.0,
            V_edge_ub = +50.0,
            ### V/D prior bounds
            VoD_axis_mean = 0.0,
            VoD_axis_std = +5.0,
            VoD_axis_lb = -15.0,
            VoD_axis_ub = +15.0,
            VoD_mid_mean = 0.0,
            VoD_mid_std = +5.0,
            VoD_mid_lb = -50.0,
            VoD_mid_ub = +50.0,
            VoD_edge_mean = 0.0,
            VoD_edge_std = +50.0, 
            VoD_edge_lb = -150.0,
            VoD_edge_ub = +50.0,
            ###
            use_truncnorm_V=True,
            ###
            signal_mask=[True, True, True],
            noise_type='proportional Gaussian',
            explicit_D=None,
            explicit_D_roa=None,
            explicit_V=None,
            explicit_V_roa=None,
            use_PMMCMC_tshifts=False,
            num_pts_PMMCMC_tshifts=10,
            method_PMMCMC_tshifts='QMC',
            Hirex_min_rel_unc=0.05, 
            Hirex_min_abs_unc=0.03,
            Hirex_unc_multiplier=1.0,
            sig_rise_enforce=1.0,
            XEUS_min_rel_unc=0.05,   
            XEUS_min_abs_unc=0.1,   
            LoWEUS_min_rel_unc=0.05,
            LoWEUS_min_abs_unc=0.1,
            XTOMO_min_rel_unc= 0.1,
            XTOMO_min_abs_unc = 0.05,
            d_weights = None,
            free_d_weights = False,
            CDI=0,
            CDI_a=100.,
            CDI_b=0.01, 
            Hirex_time_shift = 0.0,
            VUV_time_shift = 0.0,
            print_locals = False,
            sawteeth_inv_rad=False,    
            sawteeth_inv_mix_factor=np.sqrt(2.0),   #####
            adapt_rsaw_flag = False,
            sawteeth_times=False,    
            learn_VoD=False,           
            ped_roa_DV = 0.95, 
            nearaxis_roa = True, 
            # MultiNest parameters:
            n_live_points = 400,
            sampling_efficiency = 0.3,
            MN_INS = True, 
            MN_const_eff = False,  
            MN_lnev_tol = 0.5,
            copula_corr = 0.5, 
            non_separable_DV_priors = True,
            free_mixing_radius = False,
            mix_rad_free_width = 1.0, 
            free_sawtooth_width = False,
            ###
            K_spatial_grid=6.0, 
            dr_center=0.3, 
            dr_edge=0.05, 
            ###  
            decay_length_boundary = 0.2, 
            ###
            force_identifiability = True,
            min_knots_dist=0.1,
            D_prior_dist = 'loguniform',
            #use_VoDcore_V_edge=False,
            fix_D_to_tglf_neo = False,
            fix_V_to_tglf_neo = False,
            ####
            Te_scale=1.0,
            sawtooth_width_cm = 1.0, #cm   # larger, 3cm?
            #####
            vuv_rel_calibration = False,
            ###
            use_gaussian_Dped = False,
            gaussian_D_amp = 1.0, #0.5, 
            gaussian_D_w = 0.02, 
            gaussian_D_r = 'wall', #1.0, 
            free_gaussian_Dped = [False, False, False],
            ###
            use_gaussian_Vped = False,
            gaussian_V_amp = -50.0, 
            gaussian_V_amp_std = 50.0, 
            gaussian_V_amp_lb = -300.0,
            gaussian_V_amp_ub = 0.0,
            gaussian_V_w = 0.025, #0.03, #!!!!!!!
            gaussian_V_r =1.0, #0.99, #0.98, 
            free_gaussian_Vped = [False, False, False],
            scale_gaussian_Dped_by_Z = True,
            scale_gaussian_Vped_by_Z = True,
            ###
            gaussia_V_amp_std = 50.0,
            ###
            couple_D_V_gauss_params = True, 
            ###
            #fix_final_V_knot_value = True,
            #fix_final_D_knot_value = True,
            gauss_corr_coeff = 0.0,
            ### 
            #D_log_base=10.0,
            ###
            check_particle_conservation=False,
            min_penetration_fraction=0.05, 
            max_p_conserv_loss=0.05,
            ####
            rcl = -1.0,     # <0: no recycling & no divertor return
            tau_divsol = 50, #[ms]
            tau_pump = 1000.0,   #[ms]
            tau_rcl_ret = 10, #[ms]
            free_rcl = False,
            free_tau_divsol = False,
            ###
            n0x=1.0, 
            zeta=1.0,
            free_n0x = False,
            free_zeta = False,
            neutrals_file=None, 
            include_neutrals_cxr=False,
            ###
            apply_CF_corr=True,
            synthetic_data=False, 
            ###
            apply_superstaging=False,

        ):
        
        if print_locals:
            saved_args = locals()
            print("Arguments passed to Run() are: ")
            print(saved_args)
        
        # add all arguments in self.arg=*
        # save_args(vals()) # not used because it would crash way the getters/setters work

        # self.roa_grid_DV will be changed into aurora grid (in r/a units)
        self.roa_grid_DV = np.linspace(0, 1.075, 100)  # used to be 1.05

        self.check_particle_conservation = bool(check_particle_conservation)
        self.min_penetration_fraction = float(min_penetration_fraction)
        self.max_p_conserv_loss = float(max_p_conserv_loss)
        
        #centrifugal force correction
        self.apply_CF_corr = bool(apply_CF_corr)

        # Recycling:
        self.rcl = float(rcl)
        self.tau_divsol = float(tau_divsol)
        self.tau_pump = float(tau_pump)
        self.tau_rcl_ret = float(tau_rcl_ret)
        
        # possible free parameters
        rcl_scale = 0.1
        a = (0.0 - rcl)/rcl_scale
        b = (1.0 - rcl)/rcl_scale
        self.rcl_prior = stats.truncnorm(a,b, loc=rcl, scale = rcl_scale)

        tau_divsol_scale = 100. #ms
        a = (10.0 - tau_divsol)/tau_divsol_scale  
        b = (1e6 - tau_divsol)/tau_divsol_scale
        self.tau_divsol_prior = stats.truncnorm(a,b, loc=tau_divsol, scale=tau_divsol_scale)

        # to use log-normal prior, use formulae from wikipedia: https://en.wikipedia.org/wiki/Log-normal_distribution
        self.mu_x_n0x = 1.0
        self.sigma_x_n0x=0.25 #0.1 #0.25
 
        # log/exp basis does not matter -- see wikipedia
        self.mu_n0x = np.log(self.mu_x_n0x**2/np.sqrt(self.mu_x_n0x**2+self.sigma_x_n0x**2))  
        self.sigma_n0x = np.sqrt(np.log(1+self.sigma_x_n0x**2/self.mu_x_n0x**2))
        self.n0x_prior = stats.lognorm(self.sigma_x_n0x, scale=np.exp(self.mu_n0x)) #
        
        # the following shows that dist is the right form of the distribution
        # uu = np.random.uniform(size=30000)
        # dist1 = stats.lognorm(sigma_x_n0x, scale=np.exp(mu_n0x))
        # res1 = np.array([dist1.ppf(uu[ii]) for ii in np.arange(len(uu))])
        # plt.figure();  plt.hist(res1, bins=50, color='g')
        # print(f'Mean {np.mean(res1)} should be {self.mu_x_n0x} if this were to work') # this works
        # print(f'Std {np.std(res1)} should be {self.sigma_x_n0x} if this were to work') # this works
    
        self.vuv_rel_calibration = vuv_rel_calibration

        self.gauss_corr_coeff = float(gauss_corr_coeff)

        # Parameters for wavelength weighting 
        self.xics_lam_wf=1. # weight multiplier at chosen lines
        self.highlight_lines = ['xmsty', ] #['wn4','k','q','r']   # XICS wavelengths to be weighted extra
        self.xics_wf_dlam_A  = 0.001 # 1 mA, wavelength range near chosen lines

        # ----------
        # For VUV:
        #- Set the correction factor (DeltaT) to be inferred within a uniform distribution of bounds [ dt/2, 3/2* dt] 
        #- Attempt to match experimental data by integrating simulated signals in the interval [t+DeltaT - dt/2, t + DeltaT + dt/2] .

        # recall: stats.uniform takes loc, scale
        self.shift_prior = [stats.uniform(-10e-3, 15e-3 + 10e-3),  # s  # allow up to 15 ms afterwards for L-mode
                            #self.shift_prior = [stats.uniform(-10e-3, 8e-3 + 10e-3),  # s
                            stats.uniform(-2e-3,3e-3+2e-3), #s  <------- appears mostly unjustified!!
                            #stats.uniform(1e-3, 3e-3 - 1e-3), #s  # [ dt/2, 3/2* dt]  < ---- supposed to be most correct!!
                            stats.uniform(-2e-3,2e-3+2e-3)] #s
        
        self.use_PMMCMC_tshifts = bool(use_PMMCMC_tshifts)
        self.signal_mask = np.array(signal_mask, dtype=bool)

        self.shot = int(shot)
        self.version = int(version)
        self.time_1 = float(time_1)
        self.time_2 = float(time_2)

        # set spatial resolution
        self.K_spatial_grid=K_spatial_grid
        self.dr_center=float(dr_center)
        self.dr_edge=float(dr_edge)
        
        # aurora edge parameters
        self.decay_length_boundary = float(decay_length_boundary)        

        # --------------------------------------
        # experimental data info
        self.injections = injections
        
        self.thaco_tht = int(thaco_tht)        
        self.debug_plots = bool(debug_plots)
        self.include_loweus = bool(include_loweus)
        self.normalize = bool(normalize)

        # option to rescale signals via a least-square analytic normalization:
        self.LS_diag_rescaling = True # set always to True, but modified temporarily when creating synthetic signals

        #choice of whether to use a single scalar multiplier (alpha) or also a constant background (beta)
        self.use_LS_alpha_only = bool(use_LS_alpha_only)

        #  ------------------------------

        # for synthetic data
        self.params_true = np.asarray(params_true, dtype=float) if params_true is not None else None
        if not isinstance(explicit_D, bool):
            self.explicit_D = np.asarray(explicit_D, dtype=float) if explicit_D is not None else None
            self.explicit_D_roa = np.asarray(explicit_D_roa, dtype=float) if explicit_D_roa is not None else None
            self.explicit_V = np.asarray(explicit_V, dtype=float) if explicit_V is not None else None
            self.explicit_V_roa = np.asarray(explicit_V_roa, dtype=float) if explicit_V_roa is not None else None
        else:
            # if explicit_D is given as a boolean, it will later be substituted with a random draw of parameters from the prior
            pass
        self.noise_type = noise_type

        if self.params_true is not None:
            # fix random seed so that synthetic data is always produced via the same seed
            import random  
            random.seed(0) 
            
        # Minimum diagnostic uncertainties        
        self.Hirex_min_rel_unc=Hirex_min_rel_unc
        self.Hirex_min_abs_unc=Hirex_min_abs_unc
        self.Hirex_unc_multiplier=Hirex_unc_multiplier
        self.sig_rise_enforce=sig_rise_enforce
        self.XEUS_min_rel_unc=XEUS_min_rel_unc
        self.XEUS_min_abs_unc=XEUS_min_abs_unc
        self.LoWEUS_min_rel_unc=LoWEUS_min_rel_unc
        self.LoWEUS_min_abs_unc=LoWEUS_min_abs_unc
        self.XTOMO_min_rel_unc = XTOMO_min_rel_unc
        self.XTOMO_min_abs_unc = XTOMO_min_abs_unc
        
        ### time resolutions
        self.hirex_time_res=float(hirex_time_res)
        self.vuv_time_res = float(vuv_time_res)
        self.xtomo_time_res=float(xtomo_time_res)
        self.xtomo_time_res_decay=float(xtomo_time_res_decay)
        self.xtomo_decay_phase_time=float(xtomo_decay_phase_time)

        # Other adaptations for comparing to diagnostic signals
        self.Hirex_time_shift = Hirex_time_shift
        self.VUV_time_shift = VUV_time_shift

        # Knot sampling options
        self.fixed_knots = fixed_knots

        # assume relation between sawtooth inversion and mixing radius
        self.sawteeth_inv_mix_factor = float(sawteeth_inv_mix_factor)
        self.sawteeth_inv_rad = float(sawteeth_inv_rad)
        self.sawteeth_mix_rad= self.sawteeth_inv_rad * self.sawteeth_inv_mix_factor
        self.adapt_rsaw_flag = bool(adapt_rsaw_flag)

        # sawteeth timing options
        self.sawteeth_times=np.array(sawteeth_times)
        
        self.sawteeth_times_str = "     "
        if not sawteeth_times[0]==False:
            self.sawteeth_times_str = self.sawteeth_times_str.join(map(str,sawteeth_times))
            self.num_sawteeth = len(sawteeth_times)
        else:
            print("No sawteeth!")
            self.num_sawteeth=0
        
        # possible freedom in mixing radius for aurora simulation
        self.mix_rad_free_width = float(mix_rad_free_width)
        loc = 0.0
        a = ( -3.*self.mix_rad_free_width - loc)/self.mix_rad_free_width
        #a = ( 0.*self.mix_rad_free_width - loc)/self.mix_rad_free_width  # don't allow inversion radius to be reduced in inference
        b = (+3.*self.mix_rad_free_width - loc)/self.mix_rad_free_width
        #b = (+3.  - loc)/self.mix_rad_free_width   # fixed maximum correction of 3 cm
        self.mixing_radius_prior = stats.truncnorm(a,b, loc=loc, scale=self.mix_rad_free_width)
 
        # Sawtooth crash specs
        self.sawtooth_width_cm = float(sawtooth_width_cm)

        # allow for free width of sawtooth crash, although this is likely not easy to explore...
        dsaw_width = 1.0
        a = (1.0 - self.sawtooth_width_cm)/dsaw_width
        b = (5.0 - self.sawtooth_width_cm)/dsaw_width
        self.dsaw_prior = stats.truncnorm(a,b, loc=self.sawtooth_width_cm, scale=dsaw_width)        

        self.copula_corr = float(copula_corr)

        #######

        self.CDI=CDI
        self.CDI_a=CDI_a
        self.CDI_b=CDI_b
        if CDI!=0 and CDI!=-1: 
            # do not allow weights to be free (they are analytically marginalized over)
            print("****** Using non-Gaussian log-likelihood! CDI = ", CDI, "  ******* ")
            free_d_weights=False  
            
        # set Gamma distribution parameters for diagnostic weights (not actually used if CDI==1). 
        if self.CDI==3 or self.CDI==4:  #in this case ignore input CDI_b and set CDI_b = 1/CDI_a
            self.CDI_b=1./self.CDI_a

        if CDI==-1:
            print(" ****** Using Cauchy likelihood function ****** ")

        # --------------------------------------------------------------------------------
        # D prior bounds
        self.D_lb = float(D_lb)
        self.D_axis_ub = float(D_axis_ub)
        self.D_mid_ub = float(D_mid_ub)
        self.D_edge_ub = float(D_edge_ub)
        
        # Options to activate truncated gaussian priors for V
        self.use_truncnorm_V=bool(use_truncnorm_V)

        # V prior bounds
        self.V_axis_mean= float(V_axis_mean)
        self.V_axis_std= float(V_axis_std)
        self.V_axis_lb = float(V_axis_lb)
        self.V_axis_ub = float(V_axis_ub)

        self.V_mid_mean= float(V_mid_mean)
        self.V_mid_std= float(V_mid_std)
        self.V_mid_lb = float(V_mid_lb)
        self.V_mid_ub = float(V_mid_ub)

        self.V_edge_mean= float(V_edge_mean)
        self.V_edge_std= float(V_edge_std)
        self.V_edge_lb = float(V_edge_lb)
        self.V_edge_ub = float(V_edge_ub)

        # V/D prior bounds
        self.VoD_axis_mean= float(VoD_axis_mean)
        self.VoD_axis_std= float(VoD_axis_std)
        self.VoD_axis_lb = float(VoD_axis_lb)
        self.VoD_axis_ub = float(VoD_axis_ub)

        self.VoD_mid_mean= float(VoD_mid_mean)
        self.VoD_mid_std= float(VoD_mid_std)
        self.VoD_mid_lb = float(VoD_mid_lb)
        self.VoD_mid_ub = float(VoD_mid_ub)

        self.VoD_edge_mean= float(VoD_edge_mean)
        self.VoD_edge_std= float(VoD_edge_std)
        self.VoD_edge_lb = float(VoD_edge_lb)
        self.VoD_edge_ub = float(VoD_edge_ub)

        # use D, V and V/D priors, rather than just D and V/D
        self.non_separable_DV_priors = bool(non_separable_DV_priors)

        if self.non_separable_DV_priors:
            # use gaussian prior for D
            self.D_prior_dist = 'truncnorm'
        else:
            # choose arbitrary distribution for 
            self.D_prior_dist = str(D_prior_dist)

        if self.D_prior_dist=='truncnorm':
            # use gaussian prior for D
            self.D_axis_std = self.D_axis_ub/3.
            self.D_mid_std = self.D_mid_ub/3.
            self.D_edge_std = self.D_edge_ub/3.

        # ------------------------------------------------------------------

        # Allow user to fix D and/or V to values obtained via NEO and TGLF
        self.fix_D_to_tglf_neo = bool(fix_D_to_tglf_neo)
        self.fix_V_to_tglf_neo = bool(fix_V_to_tglf_neo)

        # load D and V from interpolated NEO+TGLF modeling
        self.roa_grid_models, self.D_models, self.V_models = bits_helper.get_merged_transport_models(
            self.shot, rho_choice='r/a', plot=False)

        # add radial value at the end of the grid to ensure good behavior
        self.roa_grid_models = np.concatenate((self.roa_grid_models, np.array([np.max(self.roa_grid_DV),])))
        self.D_models = np.concatenate((self.D_models, np.array([self.D_models[-1],])))
        self.V_models = np.concatenate((self.V_models, np.array([-10.0,]))) # arbitrary small value

        #####
        self.use_gaussian_Dped =  bool(use_gaussian_Dped)    
        self.use_gaussian_Vped =  bool(use_gaussian_Vped)  

        if self.use_gaussian_Dped and self.use_gaussian_Vped:
            self.couple_D_V_gauss_params = bool(couple_D_V_gauss_params)
            if self.couple_D_V_gauss_params:
                # let width and location of gaussian be determined by D gaussian parameters
                free_gaussian_Vped = [True,False,False]
        else:
            self.couple_D_V_gauss_params = False
        
        # ---------------------------------------------------------------------------------------------------

        if self.use_gaussian_Dped==False:
            free_gaussian_Dped = [False, False, False]
        else:
            # if free_gaussian_D_ped was given as a single boolean, assume that all three flags are the same
            if isinstance(free_gaussian_Dped, list) and len(free_gaussian_Dped)==3:
                pass
            elif isinstance(free_gaussian_Dped, list) and len(free_gaussian_Dped)==1:
                free_gaussian_Dped = free_gaussian_Dped*3 # requires free_gaussian_Dped to be a list, NOT np.array
            elif isinstance(free_gaussian_Dped, np.ndarray) and len(free_gaussian_Dped)==1:
                free_gaussian_Dped = np.concatenate((free_gaussian_Dped,free_gaussian_Dped,free_gaussian_Dped))
            elif int(free_gaussian_Dped) == 0:   # no free parameters
                free_gaussian_Dped = [False, False, False]
            elif int(free_gaussian_Dped) == 1:   #only amplitude is free
                free_gaussian_Dped = [True, False, False]
            elif int(free_gaussian_Dped) == 2:  # amplitude and width are free
                free_gaussian_Dped = [True, True, False]
            elif int(free_gaussian_Dped) == 3:   # amplitude, width and center location are free
                free_gaussian_Dped = [True,True,True]
            else:
                raise ValueError('Unrecognized free_gaussian_Dped form!')           


        if gaussian_D_r=='wall':
            # set gaussian_D to be centered at the last grid point
            gaussian_D_r = np.max(self.roa_grid_DV)
        
            # also set 2*std to be the distance to the LCFS
            gaussian_D_w = 0.5* (gaussian_D_r - 1.0)

        self.gaussian_D_amp_prior = stats.loguniform(self.D_lb, self.D_edge_ub)

        D_w_scale  = 0.01
        a = (0.01 - float(gaussian_D_w))/D_w_scale
        b = (0.03 - float(gaussian_D_w))/D_w_scale
        self.gaussian_D_w_prior = stats.truncnorm(a, b, loc = float(gaussian_D_w), scale = D_w_scale)

        D_r_scale = 0.01
        a = (0.97 - float(gaussian_D_r))/D_r_scale
        b = (1.2 - float(gaussian_D_r))/D_r_scale   # can go outside Aurora domain
        self.gaussian_D_r_prior = stats.truncnorm(a, b, loc = float(gaussian_D_r), scale = D_r_scale)

        # --------------------------------------------------------------------------------
            
        if self.use_gaussian_Vped==False:
            free_gaussian_Vped = [False, False, False]
        else:
            # if free_gaussian_V_ped was given as a single boolean, assume that all three flags are the same
            if isinstance(free_gaussian_Vped, list) and len(free_gaussian_Vped)==3:
                pass
            elif isinstance(free_gaussian_Vped, list) and len(free_gaussian_Vped)==1:
                free_gaussian_Vped = free_gaussian_Vped*3 # requires free_gaussian_Vped to be a list, NOT np.array
            elif isinstance(free_gaussian_Vped, np.ndarray) and len(free_gaussian_Vped)==1:
                free_gaussian_Vped = np.concatenate((free_gaussian_Vped,free_gaussian_Vped,free_gaussian_Vped))
            elif int(free_gaussian_Vped) == 0:   # no free parameters
                free_gaussian_Vped = [False, False, False]
            elif int(free_gaussian_Vped) == 1:   #only amplitude is free
                free_gaussian_Vped = [True, False, False]
            elif int(free_gaussian_Vped) == 2:  # amplitude and width are free
                free_gaussian_Vped = [True, True, False]
            elif int(free_gaussian_Vped) == 3:   # amplitude, width and center location are free
                free_gaussian_Vped = [True,True,True]
            else:
                raise ValueError('Unrecognized free_gaussian_Vped form!')           

        if self.fix_D_to_tglf_neo:
            # if D should be fixed to models, don't allow any D-related parameters
            free_gaussian_Dped = [False,False,False]
        if self.fix_V_to_tglf_neo:
            free_gaussian_Vped = [False,False,False]

        # if gaussian Vped is activated, but no parameters are made free, fix Vped to ~neoclassical values for each shot
        if self.use_gaussian_Vped and np.sum(free_gaussian_Vped)==0:
            if self.shot==1101014006:
                gaussian_V_amp  = - 50.0
            elif self.shot==1101014019:
                gaussian_V_amp = - 200.0 
            elif self.shot==1101014030:
                gaussian_V_amp = -100.0
            else:
                # keep input value of gaussian_V_amp
                pass


        ###############################
        self.gaussian_V_amp_std = float(gaussian_V_amp_std)
        self.gaussian_V_amp_lb = float(gaussian_V_amp_lb)
        self.gaussian_V_amp_ub = float(gaussian_V_amp_ub)
    
        a = (self.gaussian_V_amp_lb - float(gaussian_V_amp))/self.gaussian_V_amp_std
        b = (self.gaussian_V_amp_ub - float(gaussian_V_amp))/self.gaussian_V_amp_std
        self.gaussian_V_amp_prior = stats.truncnorm(a, b, loc = float(gaussian_V_amp), scale = self.gaussian_V_amp_std)

        #V_w_scale  = 0.005 #0.01
        #a = (0.01 - float(gaussian_V_w))/V_w_scale
        #b = (0.04 - float(gaussian_V_w))/V_w_scale
        V_w_scale  = 0.01
        a = (0.01 - float(gaussian_V_w))/V_w_scale
        b = (0.1 - float(gaussian_V_w))/V_w_scale
        self.gaussian_V_w_prior = stats.truncnorm(a, b, loc = float(gaussian_V_w), scale = V_w_scale)

        V_r_scale = 0.01
        a = (0.97 - float(gaussian_V_r))/V_r_scale
        b = (1.2 - float(gaussian_V_r))/V_r_scale   # can go outside Aurora domain
        self.gaussian_V_r_prior = stats.truncnorm(a, b, loc = float(gaussian_V_r), scale = V_r_scale)

        ##################################

        # boolean to indicate whether gaussian Dped/Vped should be scaled with Z:
        self.scale_gaussian_Dped_by_Z = bool(scale_gaussian_Dped_by_Z)
        self.scale_gaussian_Vped_by_Z = bool(scale_gaussian_Vped_by_Z)

        # Free parameter for Z scaling of Vped
        # to use log-normal prior, use formulae from wikipedia: https://en.wikipedia.org/wiki/Log-normal_distribution
        # self.mu_x_zeta = 1.0
        # self.sigma_x_zeta=0.25
        # self.mu_zeta = np.log(self.mu_x_zeta**2/np.sqrt(self.mu_x_zeta**2+self.sigma_x_zeta**2))  
        # self.sigma_zeta = np.sqrt(np.log(1+self.sigma_x_zeta**2/self.mu_x_zeta**2))
        # self.zeta_prior = stats.lognorm(self.sigma_x_zeta, scale=np.exp(self.mu_zeta)) #

        self.zeta_prior = stats.uniform(0, 2) # don't allow to go -ve --> no physical motivation and bad numerical behavior

        # -------------------------------------------------------------------

        # temporary variables to enable use of get_prior()
        self.set_free_D_knots = bool(free_D_knots)
        self.equal_DV_knots = bool(equal_DV_knots)
        if self.equal_DV_knots and self.set_free_D_knots:
            # V knots must not be free, only D knots
            self.set_free_V_knots = False
            free_V_knots = False
        else:
            self.set_free_V_knots = bool(free_V_knots)
        
        # if neither of the options above are used, user can still request to learn V/D across all knots
        self.learn_VoD = bool(learn_VoD)
        self.force_identifiability = bool(force_identifiability)

        # location outside of which spline values are considered "in the pedestal" and for possible outermost knot
        self.ped_roa_DV = float(ped_roa_DV)

        # Location of innermost and outermost knot for D and V splines
        self.innermost_knot = 0.0
        #self.outermost_knot = self.ped_roa_DV  # this only affects where the free knots can fall, not how splines interpolate in between
        self.outermost_knot = 1.0 #self.roa_grid_DV.max()
        
        # minimum distance between knots, only used if force_identifiability=True:
        self.min_knots_dist=float(min_knots_dist) #0.1 #0.05

        ######## 
        self.method = str(method)   # transformation/interpolation of D,V profiles

        # Number of coefficients:
        if nD < 1:
            raise ValueError("Must have at least one free coefficient for D!")
        if nV < 1:
            raise ValueError("Must have at least one free coefficient for V!")
        
       # number of free D and V (or V/D) coefficients. Set to 0 if profile should be fixed from transport models
        self.nD = 0  if self.fix_D_to_tglf_neo else int(nD) 
        self.nV = 0  if self.fix_V_to_tglf_neo else int(nV)

        self.nkD = np.maximum(0, self.nD-1) # number of free knots associated with self.nD
        self.nkV = np.maximum(0, self.nV-1) # number of free knots associated with self.nD

        self.nDiag = 3   # always 3 true diagnostics (both BITE and BITS)

        # read in source function:
        self.source_file = str(source_file)

        # Nested sampling parameters
        self.n_live_points = int(n_live_points)
        self.sampling_efficiency = float(sampling_efficiency)
        self.MN_INS = bool(MN_INS)
        self.MN_const_eff = bool(MN_const_eff )
        self.MN_lnev_tol = float(MN_lnev_tol)
                
        # Application of impurity superstaging to reduce number of charge states
        self.apply_superstaging = bool(apply_superstaging)

        if self.apply_superstaging:
            # exclude charge states 1-9; excellent approximation for inferences since we have no diagnostics on these states
            self.superstages = np.array([0,1,10,11,12,13,14,15,16,17,18,19,20])

        # ------------------------------------------------------------
        # EFIT tree access
        try:
            self.efit_tree = eqtools.CModEFITTree(self.shot)
        except: # MDSplus can be overloaded in job arrays
            print("Trying to import EFIT a second time....")
            try:
                time_.sleep(10) # give some waiting time and try again
                self.efit_tree = eqtools.CModEFITTree(self.shot)
            except:
                raise RuntimeError('Could not import CMOD EFIT!')

        _t0 = (self.time_1 + self.time_2) / 2.0
        self._t0 = _t0
        indt = np.argmin(np.abs(self.efit_tree.getTimeBase()  - _t0))
        self.R0_t = self.efit_tree.getMagR()[indt]  # get major radius on axis in the middle of time interval
        self.Z0_t = self.efit_tree.getMagZ()[indt]


        ##################
        # roa radial grid is r_mid/a_mid, NOT r_V/a_V
        self.roa_grid = np.linspace(0, 1.1, 100)  #used to be up to 1.2
        self.roa_grid_in = copy.deepcopy(self.roa_grid )

        self.Te_scale=Te_scale

        # load geqdsk
        self.geqdsk = self.get_geqdsk()  

        # Load ne and Te profiles depending on given options
        self.load_ne_te_ti()   # gives self.ne_cm3_in, self.Te_eV_in

        # Try to load edge neutral profiles:
        self.neutrals_file  = str(neutrals_file)
        self.include_neutrals_cxr = bool(include_neutrals_cxr)
        
        if self.include_neutrals_cxr:
            # only attempt to load neutral profiles if CXR is supposed to be included:
            #self.load_lya_n0()
            self.load_solps_n0(False)


        ### Get roa_mix
        roa_mix = self.efit_tree.rho2rho('Rmid','r/a',self.R0_t+self.sawteeth_mix_rad/100.0, _t0)

        # Convert sawteeth_mix_rad from real-world-cm to circular-geometry-cm for aurora
        self.rsaw_vol_cm = np.sqrt(self.efit_tree.rho2rho('r/a','v', roa_mix,_t0)/ (2*np.pi**2 * self.R0_t)) * 100

        # set nearaxis_roa to define within which radius different "near-axis" prior should be set
        if isinstance(nearaxis_roa, bool) and nearaxis_roa==True:
            self.nearaxis_roa = roa_mix/np.sqrt(2.0)  # set prior based on value of INVERSION radius 
        elif isinstance(nearaxis_roa, bool) and nearaxis_roa==False:
            self.nearaxis_roa = 0.0  # makes this variable useless
        else:
            self.nearaxis_roa = nearaxis_roa   # take value directly from user

        ### Construct new working directory  ###
        self.working_dir='bits_%d_%d' % (self.shot, self.version)

        # If a BITS directory doesn't exist yet, create one and set it up:
        self.bits_run_dir = os.path.join(usr_home, self.working_dir)
        if os.path.isdir(self.bits_run_dir) and os.path.exists(self.bits_run_dir):
            print(f"BITS run directory {self.bits_run_dir} is in place.")
        else:
            os.mkdir(self.bits_run_dir)
            print(f"Created BITS run directory in {self.bits_run_dir}")
            
        # choose what file to load for each Ca charge state seen by XICS signals
        self.log10pec_17 = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca17.dat.fs') # atomDB/FS
        self.log10pec_18 = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca18.dat')   # atomDB (updated 5/24/21)
        self.log10pec_19 = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/atomdb/pec#ca19.dat')   # atomDB

        # wavelengths from atomDB/APEC
        self.lam_w_Ca = 3.177153   #exc and recom
        self.lam_x_Ca = 3.189099 #exc and recom
        self.lam_y_Ca = 3.192747#exc and recom
        self.lam_z_Ca = 3.211031 #exc and recom
        
        # wavelengths from C-Mod XICS spreadsheet
        self.lam_w_Ca_cmod = 3.1773
        self.lam_x_Ca_cmod = 3.1892
        self.lam_y_Ca_cmod = 3.1928
        self.lam_z_Ca_cmod = 3.2111

        # j and k lines have slightly different wavelengths recorded in atomDB for dielectronic and excitation components
        self.lam_j_Ca_exc = 3.2090  # <--- modify to DRSAT value below
        self.lam_j_Ca_drsat = 3.2102  #### best
        self.lam_k_Ca_exc = 3.2058 # <--- modify to DRSAT value below
        self.lam_k_Ca_drsat = 3.2064   #### best
        
        #store drsat wavelengths as "overall" lambda for simplicity
        #self.lam_j_Ca = 3.2102   # drsat
        #self.lam_k_Ca = 3.2064  # drsat

        # create atomdat_Ca, only to contain wavelength centers and ranges
        self.atomdat_Ca = [[],[]] 

        # for SXR:
        self.pls_prs_loc = '/home/sciortino/atomlib/atomdat_master/pue2021_data/'
        self.pls_file_Ca = 'pls_caic_mix_Ca_9.dat' #'pls_Ca_9.dat' 
        self.prs_file_Ca = 'prs_Ca_9.dat'

        # Get spectroscopic signals:
        print("Loading signals...")
        self.signals = []

        # location where signals should be stored for each shot:
        #self.spec_datafile = f"./{self.shot}/signals_{self.shot}_wxyz.pkl"
        self.spec_datafile = f"./{self.shot}/signals_{self.shot}_real.pkl"
        self.truth_datafile = self.bits_run_dir+f"/truth_data_{self.shot}_wxyz.pkl"

        #Load BSFC fits of w line and use them to obtain a line-integrated estimate of Doppler shift,
        # to apply only after line integration
        with open(f'{self.shot}/bsfc_hirex_{self.shot}_w_Ca.pkl','rb') as f:
            bsfc_fits = pkl.load(f)
            
        # calculate Doppler shifts only at the peak of signal, when they are most accurate
        peak_tind = np.int(np.mean(np.argmax(bsfc_fits['hirex_signal'],axis=0)))
        
        # approximate Doppler shift across section with wavelength of w line. Difference is negligible, and
        # ignoring it saves us a fair amount of interpolation
        self.dlam_doppler = self.lam_w_Ca * (bsfc_fits['vel'][peak_tind,:]*1e3)/ c_speed 

        try:
            # attempt to load pre-processed signals
            with open(self.spec_datafile if self.params_true is None else self.working_dir+'/signals.pkl', 'rb') as f:
                self.signals = pkl.load(f)
            print('Loaded spectroscopic data from ', self.spec_datafile)                   

            # replace XICS data as needed here
            if self.signal_mask[0]:

                # load background-substracted XICS spectrum
                lams_A, times_cut, spec_br, spec_br_unc, pos = self.load_xics_clean()

                # now store spectrum as a SpectrumSignal object
                _norm_val = np.nanmax(spec_br)
                hirex_spec = bits_utils.SpectrumSignal(
                    spec_br, spec_br_unc, 
                    spec_br/_norm_val, spec_br_unc/_norm_val, 
                    lams_A, times_cut, 'XICS', 0, pos=pos
                )

                self.signals[0] = hirex_spec

            # store info about each diagnostic's time resolution
            self.set_diag_time_res()
            
            if signal_mask[1]:
                
                # prevent excessive outliers long before injection -- this shouldn't be needed, but reduces issues
                self.signals[1].y[self.signals[1].t<0.002] = self.signals[1].y[self.signals[1].t<0.002]/3.0
                self.signals[1].y_norm[self.signals[1].t<0.002] = self.signals[1].y_norm[self.signals[1].t<0.002]/3.0

                # prevent -ve values
                #self.signals[1].y[self.signals[1].y<0] = 0.0
                #self.signals[1].y_norm[self.signals[1].y_norm<0] = 0.0

                # store center and width of wavelength ranges that were selected, so that we can write atomdat_Ca
                vuv_lams = []; vuv_lams_width=[]
                for system in self.signals[1].vuv_lines.keys():
                    for vuv_sig in self.signals[1].vuv_lines[system]:
                        vuv_lams.append(round(np.mean([vuv_sig.lam_lb,vuv_sig.lam_ub]),3))  # A
                        vuv_lams_width.append(round(np.std([vuv_sig.lam_lb, vuv_sig.lam_ub]),3))  # A 
                self.num_vuv_lines = self.signals[1].y.shape[1]

                ### add to the atomdat tuple for Ca:
                for llam,llam_width in zip(vuv_lams,vuv_lams_width):
                    self.atomdat_Ca[0].append(llam)
                    self.atomdat_Ca[1].append(llam_width)

                # choose whether to use relative calibration of VUV lines or consider each line independently
                # NB: this can be chosen differently for each inference, regardless of the normalization initially obtained
                self.signals = set_vuv_normalization(self.signals, rel_calibration=self.vuv_rel_calibration)
            else:
                self.num_vuv_lines = 0

            # ensure that correct number of signals for specific run is loaded
            for ijk in np.arange(len(self.signal_mask)):
                if self.signal_mask[ijk]==False: self.signals[ijk]=None 

            if os.path.isfile(self.truth_datafile):
                with open(self.truth_datafile, 'rb') as f:
                    self.truth_data = pkl.load(f)

            # correct atomdat_idx indices that map each line emissivity to the correspondent index in the dlines array
            self.sxr_extra = 1 if self.signal_mask[2] else 0
            self.num_xics_channels = self.signals[0].lams.shape[1]

            # --------------------------------
            # set the number of diagnostic weights that are needed - count all VUV lines as 1
            self.nW = np.sum(self.signal_mask) # 5 if 3 VUV + 4 XICS lines 

            # Initialize params array here -- fixed params will be set later
            self.params = self.prior_random_draw()

            self.fixed_params = np.zeros_like(self.params, dtype=bool)

            # -----------------------------------------------------------------
            # setup Aurora 
            self.setup_aurora(cxr_flag=self.include_neutrals_cxr)

            # store sqrtpsinorm grid for use throughout the code
            self.rhop = self.asim.rhop_grid

            # obtain sub-sampled grids:
            self.subsample_grids()
            # -----------------------------------------------------------------

            # Get diagnostics' line integration weights
            self.compute_view_data()
            if not self.signals[0].weights[0,:].any():
                # some issue computing weights, try again
                self.compute_view_data(debug_plots=True)
                print('Potential issues computing diagnostic weights!')
                embed()


        except IOError:
            if self.params_true is None:
                # Fetch and pre-process data
                if self.signal_mask[0]:

                    # load background-substracted XICS spectrum
                    lams_A, times_cut, spec_br, spec_br_unc, pos = self.load_xics_clean()

                    _norm_val = np.nanmax(spec_br)
                    # now store spectrum as a SpectrumSignal object
                    hirex_spec = bits_utils.SpectrumSignal(
                        spec_br, spec_br_unc, 
                        spec_br/_norm_val, spec_br_unc/_norm_val, 
                        lams_A, times_cut, 'XICS', 0, pos=pos
                    )
                
                    self.signals.append(hirex_spec)
                else:
                    self.signals.append(None)

                if self.signal_mask[1]:
                    vuv_data = bits_diags.VUVData(self.shot, self.injections, debug_plots=self.debug_plots)
                    self.signals.append(vuv_data.signal)
                    # add info on lines that were loaded
                    self.signals[1].vuv_lines = vuv_data.vuv_lines

                    # prevent excessive outliers long before injection -- this shouldn't be needed, but reduces issues
                    self.signals[1].y[self.signals[1].t<0.002] = self.signals[1].y[self.signals[1].t<0.002]/3.0
                    self.signals[1].y_norm[self.signals[1].t<0.002] = self.signals[1].y_norm[self.signals[1].t<0.002]/3.0

                    # prevent -ve values
                    #self.signals[1].y[self.signals[1].y<0] = 0.0
                    #self.signals[1].y_norm[self.signals[1].y_norm<0] = 0.0
                    
                    # store center and width of wavelength ranges that were selected, so that we can write atomdat_Ca
                    vuv_lams = []; vuv_lams_width=[]
                    for system in vuv_data.vuv_lines.keys():
                        for vuv_sig in vuv_data.vuv_lines[system]:
                            vuv_lams.append(round(np.mean([vuv_sig.lam_lb,vuv_sig.lam_ub]),3))  # A
                            vuv_lams_width.append(round(np.std([vuv_sig.lam_lb, vuv_sig.lam_ub]),3))  # A 
                    self.num_vuv_lines = self.signals[1].y.shape[1]
                else:
                    vuv_data = None
                    self.signals.append(None)
                    self.num_vuv_lines=0

                if self.signal_mask[2]:
                    # Soft X-Ray (XTOMO) arrays
                    xtomo_data = bits_diags.XTOMOData(self.shot, self.injections)
                    self.signals.append(xtomo_data.signal)
                else:
                    xtomo_data = None
                    self.signals.append(None)
                self.sxr_extra = 1 if self.signal_mask[2] else 0
                self.num_xics_channels = self.signals[0].lams.shape[1]

                # store info about each diagnostic's time resolution
                self.set_diag_time_res()

                ### form the atomdat tuple for Ca:
                for llam,llam_width in zip(vuv_lams,vuv_lams_width):
                    # only 1 known Be-like Ca wavelength known in LoWEUS range:
                    self.atomdat_Ca[0].append(llam)
                    self.atomdat_Ca[1].append(llam_width)

                # ---------------------------------
                # choose whether to use relative calibration of VUV lines or consider each line independently
                # NB: this can be chosen differently for each inference, regardless of the normalization initially obtained
                self.signals = set_vuv_normalization(self.signals, rel_calibration=self.vuv_rel_calibration)
                # ---------------------------------

                # set the number of diagnostic weights that are needed
                self.nW = np.sum(self.signal_mask) 

                # Initialize params array here -- fixed params will be set later
                self.params = self.prior_random_draw()

                # make sure that knots are sorted, in case forced-identifiability knots prior is being used
                #nDnV = self.nD+self.nV
                #self.params[nDnV: nDnV+self.nkD] = np.sort(self.params[nDnV: nDnV+self.nkD])
                #self.params[nDnV+self.nkD: nDnV+self.nkD+self.nkV]=\
                #                                                                          np.sort(self.params[nDnV+self.nkD: nDnV+self.nkD+self.nkV])
                self.fixed_params = np.zeros_like(self.params, dtype=bool)

                # -----------------------------------------------------------------
                # setup Aurora
                self.setup_aurora(cxr_flag=self.include_neutrals_cxr)

                # store rhop grid for use throughout the code
                self.rhop = self.asim.rhop_grid

                # obtain sub-sampled grids:
                self.subsample_grids()
                # -----------------------------------------------------------------

                # get diagnostics' line integration weights
                self.compute_view_data()


            else:   # Generate synthetic data:

                # PECs for VUV --  these are only needed here to find order of CA_17 and 18 VUV lines in these files
                log10PEC_VUV = {}
                log10PEC_VUV[16] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca16_10A_70A.dat')   #FS
                log10PEC_VUV[17] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca17_10A_70A.dat')   #FS
                CA_16_VUV_LINES_A = np.array(list(log10PEC_VUV[16].keys()))  # A, not nm
                CA_17_VUV_LINES_A = np.array(list(log10PEC_VUV[17].keys()))  # A, not nm

                # add lines in synth_li_lines (normally on XEUS) and synth_be_lines (normally on LoWEUS) in A units
                for llam_idx in synth_li_lines:
                    self.atomdat_Ca[0].append(CA_17_VUV_LINES_A[llam_idx])   # A
                    self.atomdat_Ca[1].append(0.001) 
                for llam_idx in synth_be_lines:
                    self.atomdat_Ca[0].append(CA_16_VUV_LINES_A[llam_idx])  # A
                    self.atomdat_Ca[1].append(0.001)

                # POS vectors for XICS line integration
                #pos, pos_lam = bits_diags.hirexsr_pos(self.shot, self.thaco_tht)
                
                if self.signal_mask[0]:
                    npts = int(scipy.ceil((self.time_2 - self.time_1 + presampling_time) / hirex_time_res))
                    times = np.linspace(-presampling_time, -presampling_time + hirex_time_res * (npts - 1), npts)
                    #lam = np.linspace(3.17, 3.215, 50)

                    # take true wavelengths from diagnostic
                    out = bits_diags.load_xics_data(self.shot, tht=self.thaco_tht)
                    pos, lams_A, _, _,_ = out

                    self.signals.append(
                        bits_utils.SpectrumSignal(
                            np.zeros((len(lams_A), len(times), pos.shape[0])),
                            np.zeros((len(lams_A), len(times), pos.shape[0])),
                            np.zeros((len(lams_A), len(times), pos.shape[0])),
                            np.zeros((len(lams_A), len(times), pos.shape[0])),
                            lams_A, #lam,
                            times,
                            'XICS w',
                            0,
                            pos=pos
                        )
                    )
                    self.num_xics_channels = self.signals[0].lams.shape[1]
                else:
                    self.signals.append(None)

                # VUV:
                if self.signal_mask[1]:
                    npts = int(scipy.ceil((self.time_2 - self.time_1 + presampling_time) / vuv_time_res))
                    t = scipy.linspace(-presampling_time, -presampling_time + vuv_time_res * (npts - 1), npts)
                    self.signals.append(
                        bits_utils.Signal(
                            np.zeros((len(t), len(synth_li_lines) + len(synth_be_lines))),
                            np.zeros((len(t), len(synth_li_lines) + len(synth_be_lines))),
                            np.zeros((len(t), len(synth_li_lines) + len(synth_be_lines))),
                            np.zeros((len(t), len(synth_li_lines) + len(synth_be_lines))),
                            t,
                            ['XEUS',] * len(synth_li_lines) + ['LoWEUS',] * len(synth_be_lines),
                            list(np.arange(0, len(synth_li_lines) + len(synth_be_lines))),  # does not count XICS
                            pos=scipy.vstack([XEUS_POS,] * len(synth_li_lines) + [LOWEUS_POS,] * len(synth_be_lines)),
                            blocks=list(np.arange(0, len(synth_li_lines) + len(synth_be_lines)))
                        )
                    )
                    self.num_vuv_lines = self.signals[-1].y.shape[1]
                else:
                    self.signals.append(None)
                    self.num_vuv_lines = 0

                # XTOMO:
                if self.signal_mask[2]:
                    npts = int(scipy.ceil((self.time_2 - self.time_1 + presampling_time) / xtomo_time_res))
                    t = scipy.linspace(-presampling_time, -presampling_time + xtomo_time_res * (npts - 1), npts)
                    self.signals.append(
                        bits_utils.Signal(
                            np.zeros((len(t), 38 * 2)),
                            np.zeros((len(t), 38 * 2)),
                            np.zeros((len(t), 38 * 2)),
                            np.zeros((len(t), 38 * 2)),
                            t,
                            ['XTOMO 1',] * 38 + ['XTOMO 3',] * 38,
                            self.num_vuv_lines,    # index in dlines where XTOMO predictions are stored
                            blocks=[1,] * 38 + [3,] * 38
                        )
                    )
                    self.signals[-1].weight_idxs = scipy.hstack((list(np.arange(0, 38)), list(np.arange(0, 38))))
                else:
                    self.signals.append(None)
                self.sxr_extra = 1 if self.signal_mask[2] else 0

                # store info about each diagnostic's time resolution
                self.set_diag_time_res()

                # count all VUV lines as 1 for weighting
                self.nW = np.sum(self.signal_mask) # 5 if 3 VUV + 4 XICS lines (all VUV share 1 weight)

                # first draw from lnprior to set some fixed variables
                self.params = self.prior_random_draw()

                # make sure that knots are sorted, in case forced-identifiability knots prior is being used
                #nDnV = self.nD+self.nV
                #self.params[nDnV: nDnV+self.nkD] = np.sort(self.params[nDnV: nDnV+self.nkD])
                #self.params[nDnV+self.nkD: nDnV+self.nkD+self.nkV]=\
                #                                                                          np.sort(self.params[nDnV+self.nkD: nDnV+self.nkD+self.nkV])

                if np.atleast_1d(self.params_true)[0]==True:
                    # use the randomly sampled parameters as "true data"
                    self.params_true = self.params.copy()
                    # set log-D and V to 0 here for clarity, since explicit D and V will be used anyway
                    self.params_true[:self.nD] = 0.0
                    self.params_true[self.nD:self.nD+self.nV] = 0.0
                else:
                    # make sure that input params_true is of the right length
                    assert len(self.params) == len(self.params_true)

                # set parameters to be the "true parameters" passed by user
                self.params = self.params_true.copy()
                self.fixed_params = np.zeros_like(self.params, dtype=bool)

                # -----------------------------------------------------------------
                # setup Aurora
                self.setup_aurora(cxr_flag=self.include_neutrals_cxr)
                # store rhop grid for use throughout the code
                self.rhop = self.asim.rhop_grid
                
                # obtain sub-sampled grids:
                self.subsample_grids()
                # -----------------------------------------------------------------

                # compute weights for line integration
                self.compute_view_data()

                # ------- need to get line comps here, because they are needed for cs2dlines next ----
                # for XICS:
                try:
                    with open(f'{self.shot}/spec_comps_ca17_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','rb') as f:
                        spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17 = pkl.load(f)
                    # make sure that interpolation dimensions are still the same, otherwise compute PEC again
                    assert spec_ion_17.shape[0] == len(self.lam_spec) #self.signals[0].lams.shape[0]
                    assert spec_ion_17.shape[1] == len(self.time_spec)
                    assert spec_ion_17.shape[2] == len(self.rhop_spec)
                except:
                    print('... Caching XICS spectral components for Ca17+ ...')
                    out_17 = self.cache_xics_spec_comps(self.log10pec_17)
                    spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17 = out_17
                    with open(f'{self.shot}/spec_comps_ca17_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','wb') as f:
                        pkl.dump([spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17], f)

                try:
                    with open(f'{self.shot}/spec_comps_ca18_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','rb') as f:
                        spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18 = pkl.load(f)
                    # make sure that interpolation dimensions are still the same, otherwise compute PEC again
                    assert spec_ion_18.shape[0] == len(self.lam_spec) #self.signals[0].lams.shape[0]
                    assert spec_ion_18.shape[1] == len(self.time_spec)
                    assert spec_ion_18.shape[2] == len(self.rhop_spec)
                except:
                    print('... Caching XICS spectral components for Ca18+ ...')
                    out_18 = self.cache_xics_spec_comps(self.log10pec_18)
                    spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18= out_18
                    with open(f'{self.shot}/spec_comps_ca18_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','wb') as f:
                        pkl.dump([spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18], f)

                #print('... Caching XICS spectral components for Ca19+ ...')
                #out_19 = self.cache_xics_spec_comps(self.log10pec_19)           
                #spec_ion_19, spec_exc_19, spec_rr_19, spec_dr_19, spec_cx_19 = out_19
            
                # combine elements for later fast memory access
                self.spec_Be_mult = spec_ion_17
                self.spec_Li_mult = spec_exc_17+spec_ion_18
                self.spec_He_mult = spec_rr_17+spec_dr_17+spec_exc_18
                self.spec_H_mult = spec_rr_18+spec_dr_18

                if self.signal_mask[1]:
                    # Now for VUV:
                    self.initialize_vuv_dlines()

                if isinstance(explicit_D, bool) and explicit_D:
                    # if explicit_D=True, use D and V from random prior draw as "true parameters"
                    D_z, V_z, times_DV= self.eval_DV(imp='Ca')  # uses self.params as parameters
                    self.explicit_D = copy.deepcopy(D_z[:,0,1]) 
                    self.explicit_V = copy.deepcopy(V_z[:,0,1])
                    self.explicit_D_roa = copy.deepcopy(self.roa_grid_DV)
                    self.explicit_V_roa = copy.deepcopy(self.roa_grid_DV)

                # Get the base cs_den truth data:
                if self.explicit_D_roa[0]!=0.0:
                    raise ValueError('First value in explicit_D_roa should be 0!')

                self.explicit_D_rhop = self.efit_tree.rho2rho('r/a', 'sqrtpsinorm', self.explicit_D_roa, (self.time_1 + self.time_2) / 2.0)
                self.explicit_D_rhop[0] = 0.0
                self.explicit_V_rhop = self.efit_tree.rho2rho('r/a', 'sqrtpsinorm', self.explicit_V_roa, (self.time_1 + self.time_2) / 2.0)
                self.explicit_V_rhop[0] = 0.0

                cs_den = self.DV2cs_den(
                    debug_plots=debug_plots,
                    explicit_D=self.explicit_D,
                    explicit_D_roa=self.explicit_D_roa,
                    explicit_V=self.explicit_V,
                    explicit_V_roa=self.explicit_V_roa
                )

                # Temporarily override normalization so we can get the absolute and relative signals
                self.normalize = False # First, absolute
                ls_diag_resc = copy.deepcopy(self.LS_diag_rescaling)
                self.LS_diag_rescaling = False

                # First for (time-dependent) CaF2:
                dlines, dlines_xics = self.cs_den2dlines(cs_den, debug_plots=debug_plots)
                
                sig_abs = self.dlines2sig(dlines, dlines_xics, debug_plots=debug_plots) 
                for s, ss in zip(sig_abs, self.signals):
                    if ss is not None:
                        ss.y = s

                # Now, normalized:
                self.normalize = True

                # First for (time-dependent) CaF2:
                sig_norm = self.dlines2sig(dlines, dlines_xics, debug_plots=debug_plots)
                for s, ss in zip(sig_norm, self.signals):
                    if ss is not None:
                        ss.y_norm = s

                # Now set it back:
                self.normalize = normalize
                self.LS_diag_rescaling = ls_diag_resc

                # time bases of various signals that go into truth_data, before any further downsampling
                truth_sigs_time = [self.signals[0].t if self.signal_mask[0] else None, 
                                    self.signals[1].t if self.signal_mask[1] else None,  
                                    self.signals[2].t if self.signal_mask[2] else None
                                    ]

                # instantiate object containing truth data
                self.truth_data = bits_utils.TruthData(params_true= params_true,
                                                       cs_den= cs_den,
                                                       time= self.time,
                                                       rhop= self.rhop,
                                                       dlines= dlines,
                                                       sig_abs= sig_abs,
                                                       sig_norm= sig_norm,
                                                       truth_sigs_time = truth_sigs_time,  # store here before signals' time downsampling
                                                       #cs_den_ar= cs_den_ar,
                                                       #dlines_ar= dlines_ar,
                                                       #sig_abs_ar= sig_abs_ar,
                                                       #sig_norm_ar= sig_norm_ar,
                                                       #time_ar= time_ar,
                                                       explicit_D= self.explicit_D,
                                                       explicit_D_roa= self.explicit_D_roa,
                                                       explicit_V= self.explicit_V,
                                                       explicit_V_roa= self.explicit_V_roa
                                                   )



        #################################
        # ============ Save signals ========== #
        if not hasattr(self, 'truth_data'):
            # exptl signals might have been saved for a long time window. Make them match the requested time_2

            if self.signals[0] is not None:
                tind_end = np.argmin(np.abs(self.signals[0].t -(self.time_2 - self.injections[0].t_inj)))
                self.signals[0].y = self.signals[0].y[:, 0:tind_end,:]
                self.signals[0].y_norm = self.signals[0].y_norm[:, 0:tind_end,:]
                self.signals[0].std_y = self.signals[0].std_y[:, 0:tind_end,:]
                self.signals[0].std_y_norm = self.signals[0].std_y_norm[:, 0:tind_end,:]
                self.signals[0].t = self.signals[0].t[0:tind_end] 
                # wavelengths are time independent

            if self.signals[1] is not None:
                tind_end = np.argmin(np.abs(self.signals[1].t -(self.time_2 - self.injections[0].t_inj)))
                self.signals[1].y = self.signals[1].y[0:tind_end,:]
                self.signals[1].y_norm = self.signals[1].y_norm[0:tind_end,:]
                self.signals[1].std_y = self.signals[1].std_y[0:tind_end,:]
                self.signals[1].std_y_norm = self.signals[1].std_y_norm[0:tind_end,:]
                self.signals[1].t = self.signals[1].t[0:tind_end]

            if self.signals[2] is not None:
                tind_end = np.argmin(np.abs(self.signals[2].t -(self.time_2 - self.injections[0].t_inj)))
                self.signals[2].y = self.signals[2].y[0:tind_end,:]
                self.signals[2].y_norm = self.signals[2].y_norm[0:tind_end,:]
                self.signals[2].std_y = self.signals[2].std_y[0:tind_end,:]
                self.signals[2].std_y_norm = self.signals[2].std_y_norm[0:tind_end,:]
                self.signals[2].t = self.signals[2].t[0:tind_end]

        if hasattr(self, 'truth_data'):
            # store truth data
            with open(self.truth_datafile, 'wb') as f:
                pkl.dump(self.truth_data, f)
                
            # Apply noise:
            if self.Hirex_min_rel_unc is not None:
                synth_noises[0] = self.Hirex_min_rel_unc

            if self.XEUS_min_rel_unc is not None:
                synth_noises[1] = self.XEUS_min_rel_unc

            # define synthetic data noise
            self.apply_noise(noises=synth_noises, noise_type=self.noise_type)
        
        # save line-integrated signals, real or synthetic data, ready for inference
        with open(self.bits_run_dir + '/signals.pkl', 'wb') as f:
            pkl.dump(self.signals, f)

        # now, reduce time resolution of XTOMO, if this was loaded from real data (saved signal file is kept at full resolution)
        if not hasattr(self, 'truth_data') and self.signal_mask[2]:
            # time-average (reduce time-resolution) of SXR across the entire time range
            self.subdigitize_signal(-1.0, xtomo_time_res, sig_idx=2,after=True)   # -1.0 to select entire time window

        if self.signal_mask[2]:
            # reduce time resolution to 3ms well before injection (1ms early)
            self.subdigitize_signal(-1e-3, 3e-3, sig_idx=2, after=False) 

            if xtomo_decay_phase_time is not 0.0:
                # reduce time resolution after specified time
                self.subdigitize_signal(xtomo_decay_phase_time, xtomo_time_res_decay, sig_idx=2,after=True)

                

        # =====================
        #      End of signal loading/creation
        # =====================
        if self.params_true is None:
            # pre-processing of atomic rates -- this hasn't been done for real data yet at this point           

            # for XICS:
            # first obtain sub-sampled grids:
            self.subsample_grids()

            try:
                with open(f'{self.shot}/spec_comps_ca17_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','rb') as f:
                    spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17 = pkl.load(f)
                # make sure that interpolation dimensions are still the same, otherwise compute PEC again
                assert spec_ion_17.shape[0] == len(self.lam_spec) #self.signals[0].lams.shape[0]
                assert spec_ion_17.shape[1] == len(self.time_spec)
                assert spec_ion_17.shape[2] == len(self.rhop_spec)
            except:
                print('... Caching XICS spectral components for Ca17+ ...')
                out_17 = self.cache_xics_spec_comps(self.log10pec_17)
                spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17 = out_17
                with open(f'{self.shot}/spec_comps_ca17_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','wb') as f:
                    pkl.dump([spec_ion_17, spec_exc_17, spec_rr_17, spec_dr_17], f)

            try:
                with open(f'{self.shot}/spec_comps_ca18_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','rb') as f:
                    spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18 = pkl.load(f)
                # make sure that interpolation dimensions are still the same, otherwise compute PEC again
                assert spec_ion_18.shape[0] == len(self.lam_spec)  #self.signals[0].lams.shape[0]
                assert spec_ion_18.shape[1] == len(self.time_spec)
                assert spec_ion_18.shape[2] == len(self.rhop_spec)
            except:
                print('... Caching XICS spectral components for Ca18+ ...')
                out_18 = self.cache_xics_spec_comps(self.log10pec_18)
                spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18 = out_18
                with open(f'{self.shot}/spec_comps_ca18_{self.shot}_{len(self.rhop_spec)}_{len(self.time_spec)}.pkl','wb') as f:
                    pkl.dump([spec_ion_18, spec_exc_18, spec_rr_18, spec_dr_18], f)

            #print('... Caching XICS spectral components for Ca19+ ...')
            #out_19 = self.cache_xics_spec_comps(self.log10pec_19)           
            #spec_ion_19, spec_exc_19, spec_rr_19, spec_dr_19 = out_19

            # combine elements for later fast memory access
            self.spec_Be_mult = spec_ion_17
            self.spec_Li_mult = spec_exc_17+spec_ion_18
            self.spec_He_mult = spec_rr_17+spec_dr_17+spec_exc_18
            self.spec_H_mult = spec_rr_18+spec_dr_18

            if self.signal_mask[1]:
                # Now for VUV:
                self.initialize_vuv_dlines()

        # Compute XICS spectral weights
        self.compute_xics_lam_weights()

        # Correct signal uncertainties, also obtaining info about finite time resolutions
        # NB: signal correction is NOT saved in signals.pkl! (those are pre-correction signals)
        if self.params_true is None:
            # Do not correct signals if these are synthetic 
            self.correct_signals()  

        # ========= Get starting parameters  ===========#
        # Sort the time axes, in case multiple injections have been combined:
        for s in self.signals:
            if s is not None:
                s.sort_t()

        if free_D_knots == False:
            if self.fix_D_to_tglf_neo:
                self.knotgrid_D = [] # no knots needed
            elif self.fixed_knots is not None:
                # if knots are fixed by the user, modify those picked from prior
                self.knotgrid_D = s
                # random_draw chooses new values for knotgrid_D,V
                print("Initial knotgrid_D from prior: ", self.knotgrid_D)

        if free_V_knots == False:
            # if knots are fixed by the user, modify those picked from prior
            if self.fix_V_to_tglf_neo: 
                self.knotgrid_V = [] # no knots needed
            elif self.fixed_knots is not None:
                self.knotgrid_V = self.fixed_knots[len(self.fixed_knots)//2:]
            elif self.equal_DV_knots:
                self.knotgrid_V = self.knotgrid_D
            else:
                # knotgrid_V from a random_draw of prior
                print("Initial knotgrid_V from prior: ", self.knotgrid_V)

        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
        print("Finalized D,V knots: ", self.knotgrid_D, self.knotgrid_V)
        print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

        # Now we can use the decorators!

        # ====================
        # first set all parameters to be free, then fix them one at a time
        self.fixed_params = np.zeros_like(self.params, dtype=bool)

        # Set these here, after creating self.params and self.fixed_params, to use the property setter:
        self.free_D_knots = free_D_knots
        if self.equal_DV_knots:
            self.free_V_knots = False
        else:
            self.free_V_knots = free_V_knots
        self.free_time_shift = np.array(free_time_shift)

        #if CXR is considered, allow for free parameter acting as (exponential) multiplier of n0:
        self.free_n0x = bool(free_n0x) if self.include_neutrals_cxr else False
        self.n0x =  float(n0x)   # if free_n0x==True, this is ignored; otherwise, it acts as a fixed value

        # options for pedestal Z scaling
        self.free_zeta = bool(free_zeta) if self.scale_gaussian_Vped_by_Z else False
        self.zeta = float(zeta) if self.scale_gaussian_Vped_by_Z else 0.0

        # # 5/7/21: fix D and V at the very edge -- it shouldn't matter!
        # if not self.fix_D_to_tglf_neo and not self.fix_V_to_tglf_neo:

        #     iidx = np.where(self.param_names[:] == '$C_{D,%d}$'%self.nD)[0][0]
        #     self.fixed_params[iidx] = True
        #     self.params[iidx] = 0.1 #self.D_lb   # impose minimum on D at last grid point here

        #     iidx = np.where(self.param_names[:] == '$C_{V,%d}$'%self.nV)[0][0]
        #     self.fixed_params[iidx] = True
        #     self.params[iidx] = 0.0

        # if self.non_separable_DV_priors and not self.fix_D_to_tglf_neo and not self.fix_V_to_tglf_neo:
        #     #and self.use_gaussian_Dped and self.use_gaussian_Vped:
        #     # if using non_separable_DV_priors, we must have equal numbers of D and V free coefficients
        #     # Require that both Vped and Dped gaussians are being used and fix last knot spline values.

        #     iidx = np.where(self.param_names[:] == '$C_{D,%d}$'%self.nD)[0][0]
        #     self.fixed_params[iidx] = True
        #     self.params[iidx] = self.D_lb   # impose minimum on D at last grid point here

        #     iidx = np.where(self.param_names[:] == '$C_{V,%d}$'%self.nV)[0][0]
        #     self.fixed_params[iidx] = True
        #     self.params[iidx] = 0.0

        # else:
        #     if not self.fix_D_to_tglf_neo and self.use_gaussian_Dped: # no free D params if fix_D_to_tglf_neo=True
        #         # fix the spline value at the last knot to D_lb
        #         iidx = np.where(self.param_names[:] == '$C_{D,%d}$'%self.nD)[0][0]
        #         self.fixed_params[iidx] = True
        #         self.params[iidx] = self.D_lb   # impose minimum on D at last grid point here

        #    # fix last spline value to 0 if gaussian pedestal features are set
        #     if not self.fix_V_to_tglf_neo and self.use_gaussian_Vped: # no free V params if fix_V_to_tglf_neo=True
        #         # fix the spline value at the last knot to 0
        #         iidx = np.where(self.param_names[:] == '$C_{V,%d}$'%self.nV)[0][0]
        #         self.fixed_params[iidx] = True
        #         self.params[iidx] = 0.0

        # ----------------------

        if d_weights is None:
            self.d_weights= [1,] * self.nW
        else:
            self.d_weights = d_weights

        # if self.signal_mask[2] and self.d_weights[2]==1:
        #         # set SXR weight such that SXR is weighted like a single XICS line
        #         self.d_weights[2] = np.size(self.signals[0].y)/np.size(self.signals[2].y)
        #         print('Set SXR weight to match total weight of XICS w line')

        self.free_d_weights = free_d_weights

        # ------------------------------------------------------------------------------------------------------------------------------

        # for Gaussian pedestal (set these here to access decorator after prior is created)
        self.gaussian_D_amp =  float(gaussian_D_amp)
        self.gaussian_D_w = float(gaussian_D_w)
        self.gaussian_D_r = float(gaussian_D_r)

        self.free_gaussian_Dped = np.asarray(free_gaussian_Dped)
        assert len(self.free_gaussian_Dped)==3

        self.gaussian_V_amp =  float(gaussian_V_amp)
        self.gaussian_V_w = float(gaussian_V_w)
        self.gaussian_V_r = float(gaussian_V_r)

        self.free_gaussian_Vped = np.asarray(free_gaussian_Vped)
        assert len(self.free_gaussian_Vped)==3

        if self.couple_D_V_gauss_params:
            # set D and V gaussian pedestal features to have different amplitudes, but same width and location
            # fix width and location to be the one of D gaussian feature (enforced in :py:method split_params)
            self.free_gaussian_Vped = np.array([free_gaussian_Vped[0], True, True])

            #iidx = np.where(self.param_names[:] == '$\Delta V_{ped}$')[0][0]
            #self.fixed_params[iidx] = True
            #iidx = np.where(self.param_names[:] == '$\\langle r_{V,ped}\\rangle$')[0][0]
            #self.fixed_params[iidx] = True


        # ------------------------
        if sawteeth_times[0]==False:
            # if there are no sawteeth, prevent useless free parameters
            self.free_mixing_radius = np.zeros_like(free_mixing_radius, dtype=bool)
            self.free_sawtooth_width  = np.zeros_like(free_sawtooth_width, dtype=bool)
        else:
            self.free_mixing_radius = np.array(free_mixing_radius, dtype=bool)
            self.free_sawtooth_width = np.array(free_sawtooth_width, dtype=bool)

        print("free_mixing_radius -----> ", self.free_mixing_radius)
        print("free_sawtooth_width -----> ", self.free_sawtooth_width)

        # recycling
        self.free_rcl = bool(free_rcl)
        self.free_tau_divsol = bool(free_tau_divsol)

        # ----------------------------------------

        # Set up the grids for PMMCMC_tshifts:
        # Here we use quasi Monte Carlo importance sampling, but the implementation is done
        # in a way that makes it possible to switch to sparse grid quadrature at a later date.
        if self.use_PMMCMC_tshifts:

            self.method_PMMCMC_tshifts = str(method_PMMCMC_tshifts)
            self.num_pts_PMMCMC_tshifts = int(num_pts_PMMCMC_tshifts)

            if self.method_PMMCMC_tshifts == 'QMC':

                self.dt_quad_arr = np.zeros((self.num_pts_PMMCMC_tshifts, self.signal_mask.sum()))

                for i in np.arange(0, self.num_pts_PMMCMC_tshifts):
                    # start from 1 to not include the -inf point:
                    q, dum = sobol.i4_sobol(self.signal_mask.sum(), i + 1)
                    # Pad this out to the point that I can use the get_prior:
                    u = 0.5 * np.ones(len(self.signals))
                    u[self.signal_mask] = q
                    #p = self.shift_prior.sample_u(u)
                    p = np.array([sprior.ppf(val) for sprior, val in zip(self.shift_prior, u)] )
                    self.dt_quad_arr[i, :] = p[self.signal_mask]

                # Mask out the inf/nan values:
                mask = (scipy.isinf(self.dt_quad_arr).any(axis=1)) | (scipy.isnan(self.dt_quad_arr).any(axis=1))
                self.dt_quad_arr = self.dt_quad_arr[~mask, :]

            elif self.method_PMMCMC_tshifts == 'GHQ':

                # NB: this method only works with Gaussian priors
                try:
                    assert hasattr(self.shift_prior, 'mean') and hasattr(self.shift_prior, 'std') # approx test
                except:
                    raise ValueError("PMMCMC_tshifts method GHQ only works for normal priors on the time shifts!")
                mu = np.array([sprior.mean() for sprior in self.shift_prior[self.signal_mask]] )
                sigma = np.array([sprior.std() for sprior in self.shift_prior[self.signal_mask]])
                pts, wts = np.polynomial.hermite.hermgauss(self.num_pts_PMMCMC_tshifts)
                self.dt_quad_arr = scipy.sqrt(2.0) * sigma * pts[:, None] + mu
                self.ln_dt_quad_wts = np.log(1.0 / (scipy.sqrt(2.0 * np.pi) * sigma) * wts[:, None])
    
            else:
                raise ValueError("Unknown method for PMMCMC_tshifts marginalization!")

    
        if self.copula_corr > 0.0:
            print("----------> Applying Gaussian copula to D & V spline values! <---------")

            free_D_vals = ~self.fixed_params[:self.nD]   # booleans
            free_V_vals = ~self.fixed_params[self.nD:self.nD+self.nV]   # booleans
            
            num_D = np.sum(free_D_vals)  # number of free D values
            num_V = np.sum(free_V_vals)  # number of free V values
            
            # freeze standard Gaussian distribution for use in the prior Gaussian Copula
            self.norm = scipy.stats.distributions.norm(0,1)

            # set correlations between D values using tridiagonal correlation matrix
            a = [self.copula_corr,]* (num_D -1)
            b = [1.0,]* num_D
            c = [self.copula_corr,]* (num_D - 1)
            self.copula_Sigma_D = tridiag(a,b,c)

            try:
                # Cholesky decomposition (faster, but equivalent to eigen-decomposition)
                # only works for "dominant diagonal"
                self.Sig_copula_D = np.linalg.cholesky(self.copula_Sigma_D)
            except:
                ww,vv = np.linalg.eig(self.copula_Sigma_D)
                self.Sig_copula_D = vv.dot(np.diag(ww)**0.5)

            a = [self.copula_corr,]* (num_V -1)
            b = [1.0,]* num_V
            c = [self.copula_corr,]* (num_V - 1)
            self.copula_Sigma_V = tridiag(a,b,c)

            try:
                # Cholesky decomposition (faster, but equivalent to eigen-decomposition)
                self.Sig_copula_V = np.linalg.cholesky(self.copula_Sigma_V)
            except:
                ww, vv= np.linalg.eig(self.copula_Sigma_V)
                self.Sig_copula_D = vv.dot(np.diag(ww)**0.5)

    # =========================
    # =========================
    # =========================

    def load_xics_clean(self):
        '''Load XICS spectrum and subtract time-dependent background from sides of spectrum,
        where there should be no line radiation.
        This allows one to subtract bremsstrahlung and estimate an uncertainty on the background.
        '''
        out = bits_diags.load_xics_data(self.shot, tht=self.thaco_tht)
        pos, lams_A, times,  spec_br, spec_br_unc = out

        # select only time points within injection interval
        i_start = np.argmin(np.abs(times - self.injections[0].t_start))
        i_stop = np.argmin(np.abs(times - self.injections[0].t_stop))
        times_cut = times[i_start:i_stop] - self.injections[0].t_inj  # count time from injection time

        num_ch = lams_A.shape[1]

        # Do background subtraction based times before injection
        # This is particularly useful to eliminate the Ar w4n + other Ar satellites (assumed time independent)
        i_bckg = np.argmin(np.abs(times - (self.injections[0].t_start-0.03)))  # 30 ms before injection
        spec_bckg = np.mean(spec_br[:,i_bckg:i_start,:],axis=1)
        spec_bckg_unc = 2*np.std(spec_br[:,i_bckg:i_start,:],axis=1) # decrease confidence in uncertainties with arbitrary factor of 2
        
        spec_br -= spec_bckg[:,None,:]
        spec_br_unc = np.sqrt(spec_bckg_unc[:,None,:]**2 +spec_br_unc**2)

        ###
        bckg_mean = np.zeros((i_stop-i_start,num_ch))
        bckg_std = np.zeros((i_stop-i_start,num_ch))

        for ch in np.arange(num_ch):       
            # range over which to average background (region of no line radiation)
            ind1=np.min(np.argmin(np.abs(lams_A[:,ch] - 3.170), axis=0))
            ind2=np.max(np.argmin(np.abs(lams_A[:,ch] - 3.173), axis=0))
            ind3=np.min(np.argmin(np.abs(lams_A[:,ch] - 3.2175), axis=0))
            ind4=np.max(np.argmin(np.abs(lams_A[:,ch] -3.220), axis=0))

            spec_br_slices = np.concatenate((spec_br[ind1:ind2,i_start:i_stop,ch], spec_br[ind3:ind4,i_start:i_stop,ch]))

            bckg_mean[:,ch] = np.mean(spec_br_slices,axis=0)
            bckg_std[:,ch] = np.std(spec_br_slices,axis=0)

        # background subtraction:
        spec_br_clean = spec_br[:,i_start:i_stop,:] - bckg_mean
        spec_br_unc_clean = np.sqrt(spec_br_unc[:,i_start:i_stop,:]**2 + bckg_std**2)

        # limit wavelength range -- this is tricky because each channel has different wavelengths, but here's a safe workaround
        #lam_bounds = (3.173, 3.214)
        lam_bounds = (3.173, 3.215) # make it slightly safer to prevent cutting wavelengths in method below

        # evaluate number of lambda points on first channel
        _ind1=np.min(np.argmin(np.abs(lams_A[:,0] - lam_bounds[0]), axis=0))
        _ind2=np.max(np.argmin(np.abs(lams_A[:,0] - lam_bounds[1]), axis=0)) 
        lams_A_final = np.zeros((_ind2 - _ind1, num_ch))
        spec_br_final = np.zeros((_ind2 - _ind1, len(times_cut), num_ch))
        spec_br_unc_final = np.zeros((_ind2 - _ind1, len(times_cut), num_ch))

        for ch in np.arange(num_ch):
            ind1=np.min(np.argmin(np.abs(lams_A[:,ch] - lam_bounds[0]), axis=0))
            ind2 = ind1+ (_ind2 - _ind1) # keeps number of lambda points constant across channels, all starting near 3.173

            lams_A_final[:,ch] = lams_A[ind1:ind2,ch]
            spec_br_final[:,:,ch] = spec_br_clean[ind1:ind2,:,ch]
            spec_br_unc_final[:,:,ch] = spec_br_unc_clean[ind1:ind2,:,ch]

        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))


        if self.shot==1101014006: # no Ar w4n line in 1101014030 and 1101014019...

            for ch in np.arange(num_ch):
                # Ar w4n, near the Ca q line
                q_mask = (lams_A_final[:,ch]>3.198)&(lams_A_final[:,ch]<3.202)
                #tmp_unc = copy.deepcopy(spec_br_unc_final[q_mask,:,ch] )
                #spec_br_unc_final[q_mask,:,ch] = 2*np.max(tmp_unc)  # arbitrary factor of 2
                
                # clear from the data that we're missing ~1% subtraction in normalized units near the q line
                # Fit a gaussian to the first time slice, always with no Ca signal
                norm = np.nanmax(spec_br_final)
                lam_Ar_wn4 = 3.1997
                popt, pcov = scipy.optimize.curve_fit(gauss_function, lams_A_final[q_mask,ch], spec_br_final[q_mask,0,ch], 
                                                      p0 = [0.01*norm, lam_Ar_wn4, 5e-4])
                
                # now subtract gaussian from every time slice, assuming time-independence of Ar w4n line
                for tidx in np.arange(spec_br_final.shape[1]):
                    spec_br_final[q_mask,tidx,ch] -= gauss_function(lams_A_final[q_mask,ch], *popt)
                    
                # Sr XXXVI, 3.1790 +/- 0.0020 A according to NIST -- near Ca wn4 lines
                #sr_mask = (lams_A[:,ch]>3.176)&(lams_A[:,ch]<3.180)
                #tmp_unc = copy.deepcopy(spec_br_unc[sr_mask,:,ch] )
                #spec_br_unc[sr_mask,:,ch] =  np.max(tmp_unc)*2
                
                # Mo XXXIII (?), near  near the Ca m-s-t satellites
                #mo_mask = (lams_A[:,ch]>3.191)&(lams_A[:,ch]<3.194)
                #tmp_unc = copy.deepcopy(spec_br_unc[mo_mask,:,ch] )
                #spec_br_unc[mo_mask,:,ch] =  np.max(tmp_unc)*2
                

        return lams_A_final, times_cut, spec_br_final, spec_br_unc_final, pos


    def subsample_grids(self):
        '''Create sub-sampled radial and temporal grids to allow for efficient caching and interpolation of spectra
        for BITS forward modeling.
        '''
        # get sawteeth times so that we always resolve them with time-dept ne,Te, even if sawteeth are not explicitly modeled in Aurora
        info_out = get_bits_shot_info(self.shot, sqrt_rinv_correction=False) 
        t_1, t_2, source_function, t_clusters, time_shifts, sawteeth_inv_rad,\
            sawteeth_times, min_abs_unc, hirex_info, NEO_lims, neutrals_file = info_out

        sawteeth_times = np.array(sawteeth_times)
        saw_times = sawteeth_times[(sawteeth_times>self.time_1)*(sawteeth_times<self.time_2)]

        # first create timing as for Aurora simulations with sawteeth
        nml_timing= copy.deepcopy(self.namelist['timing'])
        nml_timing['times'] = np.array([self.time_1] + list(saw_times) + [self.time_2]) #s
        nml_timing['dt_increase'] = np.array([1.01] + [1.01,]*len(saw_times) + [1.0])   
        nml_timing['dt_start'] = np.array([1e-5] + [1e-4,]*len(saw_times) + [1e-4])   #s
        nml_timing['steps_per_cycle'] = np.array([1.0] + [1.0,]*len(saw_times) + [1.0])

        # Now create lower time resolution than Aurora simulation, still by using aurora's create_time_grid function
        # to better resolve fast Te changes at sawteeth
        nml_timing['dt_increase'] = [2 for dt_increase in nml_timing['dt_increase']] # 5 gives ~33 points, 2 gives ~59 points
        nml_timing['dt_start'][0] = 1e-3
        self.time_spec, _ = aurora.create_time_grid(nml_timing, plot=False) # should give ~150 time points, sawtooth-resolved

        # try to keep full time resolution
        #self.time_spec = self.time   ## unfeasible, it never ends even if using numba...

        # set higher resolution in pedestal, but not as high as in Aurora forward model
        #nml = {'dr_0': 1.0, 'dr_1': 0.1, 'K': 20,
        #       'rvol_lcfs': self.namelist['rvol_lcfs'], 'bound_sep': self.namelist['bound_sep'], 'lim_sep': self.namelist['lim_sep']}
        #rvol_spec, pro_grid, qpr_grid, prox_param = aurora.create_radial_grid(nml, plot=False)
        #self.rhop_spec = interp1d(self.asim.rvol_grid, self.asim.rhop_grid,
        #                          bounds_error=False, fill_value='extrapolate')(rvol_spec)  # extrapolation needed because changing dr0 and dr1

        # use a flat rhop grid for line integration -- allows more detail on core, where most signal is actually measured
        self.rhop_spec = np.linspace(0, np.max(self.rhop), 80)

        # try to keep full spatial resolution
        #self.rhop_spec = copy.deepcopy(self.rhop)

        # interpolate ne and Te on minimally (but sufficiently) resolved time and radial grids
        self.ne_cm3_spec = interp2d(self.rhop, self.time, self.ne_cm3)(self.rhop_spec, self.time_spec)
        self.Te_eV_spec =  interp2d(self.rhop, self.time, self.Te_eV)(self.rhop_spec, self.time_spec) 

        # Add Ti and vi for Doppler broadening
        self.Ti_eV_spec = np.interp(self.rhop_spec, self.rhop_in, self.Ti_eV_in)
        self.vi_ms_spec = np.interp(self.rhop_spec, self.rhop_in, self.vi_ms_in)

        # get spectral components on equally-spaced wavelength grid
        self.lam_spec = np.linspace(3.172, 3.215, 200) # covers range of He-like Ca transitions

        # Add instrumental width
        self.Ti_eV_spec += 200.0 #300.0  # eV
        


    def cache_xics_spec_comps(self, log10pec_dict):
        '''Initialize all arrays that are necessary to compute diagnostic line emission.
        This is done here to avoid interpolating atomic rates at every BITS iteration, i.e. we pre-cache atomic rates
        for fixed ne,Te. Note that this effectively prevents one from varying ne,Te at every Aurora iteration.

        Parameters
        -----------------
        log10pec_dict : dict
            Dictionary of log-10(PEC) interpolant for various atomic processes. This should be the output of the
            :py:fun:`aurora.radiation.read_adf15` function.

        Returns
        -----------
        spec_ion : array, (wavelengths,times,radii)
            Ionization-driven spectral components.
        spec_exc : array, (wavelengths,times,radii)
            Excitation-driven spectral components.
        spec_rr : array, (wavelengths,times,radii)
            Radiative recombination-driven spectral components.
        spec_dr : array, (wavelengths,times,radii)
            Dielectronic recombination-driven spectral components.
        #spec_cx : array, (wavelengths,times,radii)
        #    Charge exchange recombination-driven spectral components.    
        '''

        log10pec_dict_red = {}
        for pec_line in log10pec_dict:
            if pec_line>3.172 and pec_line<3.215:
                log10pec_dict_red[pec_line] = log10pec_dict[pec_line]

        wave_A = np.zeros((len(list(log10pec_dict_red.keys()))))
        pec_ion = np.zeros((len(list(log10pec_dict_red.keys())), *self.ne_cm3_spec.shape))
        pec_exc = np.zeros((len(list(log10pec_dict_red.keys())), *self.ne_cm3_spec.shape))
        pec_rr = np.zeros((len(list(log10pec_dict_red.keys())), *self.ne_cm3_spec.shape))
        #pec_cx = np.zeros((len(list(log10pec_dict_red.keys())), *self.ne_cm3_spec.shape))
        pec_dr = np.zeros((len(list(log10pec_dict_red.keys())), *self.ne_cm3_spec.shape))
        
        for ii,lam in enumerate(log10pec_dict_red):
            wave_A[ii] = lam
            if 'ioniz' in log10pec_dict_red[lam]:
                pec_ion[ii,:,:] = 10**log10pec_dict_red[lam]['ioniz'].ev(np.log10(self.ne_cm3_spec),np.log10(self.Te_eV_spec))
            if 'excit' in log10pec_dict_red[lam]:
                pec_exc[ii,:,:] = 10**log10pec_dict_red[lam]['excit'].ev(np.log10(self.ne_cm3_spec),np.log10(self.Te_eV_spec))
            if 'recom' in log10pec_dict_red[lam]:
                pec_rr[ii,:,:] = 10**log10pec_dict_red[lam]['recom'].ev(np.log10(self.ne_cm3_spec),np.log10(self.Te_eV_spec))
            #if 'chexc' in log10pec_dict_red[lam]:
            #    pec_cx[ii,:,:] = 10**log10pec_dict_red[lam]['checx'].ev(np.log10(self.ne_cm3_spec),np.log10(self.Te_eV_spec))
            if 'drsat' in log10pec_dict_red[lam]:
                pec_dr[ii,:,:] = 10**log10pec_dict_red[lam]['drsat'].ev(np.log10(self.ne_cm3_spec),np.log10(self.Te_eV_spec))


        # Doppler broadening 
        ion_A = 40 # Ca
        mass = m_p * ion_A
        dnu_g = np.sqrt(2.*(self.Ti_eV_spec[None,:]*q_electron)/mass)*(c_speed/wave_A[:,None])/c_speed

        # set a variable delta lambda based on the width of the broadening
        dlam_A = wave_A[:,None]**2/c_speed* dnu_g * 3 # 3 standard deviations should be enough

        lams_profs_A =np.linspace(wave_A[:,None] - dlam_A, wave_A[:,None] + dlam_A, 30, axis=2) # 30 points per line may be enough

        theta_tmp = 1./(np.sqrt(np.pi)*dnu_g[:,:,None])*\
                    np.exp(-((c_speed/lams_profs_A-c_speed/wave_A[:,None, None])/dnu_g[:,:,None])**2)

        # Normalize Gaussian profile
        theta = np.einsum('ijl,ij->ijl', theta_tmp, 1./np.trapz(theta_tmp,x=lams_profs_A,axis=2))

        spec_ion, spec_exc, spec_rr, spec_dr = get_spec_comps(self.lam_spec,self.ne_cm3_spec,
                                                               lams_profs_A, theta, pec_ion, pec_exc, pec_rr, pec_dr)

        return spec_ion, spec_exc, spec_rr, spec_dr 
        



    def initialize_vuv_dlines(self):
        '''Initialize all arrays that are necessary to compute diagnostic line emission.
        This is done here to avoid interpolating atomic rates at every BITS iteration, i.e. we pre-cache atomic rates
        for fixed ne,Te. Note that this effectively prevents one from varying ne,Te at every Aurora iteration.
        '''
        # pre-compute PECs as a function of radius and time, to be multiplied by nZ at every iteration
        # This saves the computation time involved in applying interpolations at every iteration
        self.dlines_vuv_comps = np.zeros((len(self.time), 21, len(self.rhop), self.num_vuv_lines))   # hardcoded for Ca

        # don't save PEC data in memory, it can always be reloaded if necessary
        log10PEC_VUV = {}
        log10PEC_VUV[10] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca10_10A_70A.dat')   #FS
        log10PEC_VUV[11] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca11_10A_70A.dat')   #FS
        log10PEC_VUV[12] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca12_10A_70A.dat')   #FS
        log10PEC_VUV[13] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca13_10A_70A.dat')   #FS
        log10PEC_VUV[14] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca14_10A_70A.dat')   #FS
        log10PEC_VUV[15] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca15_10A_70A.dat')   #FS
        log10PEC_VUV[16] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca16_10A_70A.dat')   #FS
        log10PEC_VUV[17] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca17_10A_70A.dat')   #FS
        log10PEC_VUV[18] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca18_10A_70A.dat')   #FS
        log10PEC_VUV[19] = aurora.read_adf15('/home/sciortino/atomlib/atomdat_master/adf15/ca/fs#ca19_10A_70A.dat')   #FS

        for ii in np.arange(self.num_vuv_lines):
            cw = self.atomdat_Ca[0][ii]
            hw = self.atomdat_Ca[1][ii]

            for Z in log10PEC_VUV.keys():

                self.dlines_vuv_comps[:,Z-1,:,ii] += compute_emiss(log10PEC_VUV[Z], cw, hw, self.ne_cm3, self.Te_eV,
                                                                  nZ_ioniz = 1.0, nZ_exc = 0.0, nZ_rec = 0.0)

                self.dlines_vuv_comps[:,Z,:,ii] += compute_emiss(log10PEC_VUV[Z], cw, hw, self.ne_cm3, self.Te_eV,
                                                                   nZ_ioniz = 0.0, nZ_exc = 1.0, nZ_rec = 0.0)
                
                self.dlines_vuv_comps[:,Z+1,:,ii] += compute_emiss(log10PEC_VUV[Z], cw, hw, self.ne_cm3, self.Te_eV,
                                                                   nZ_ioniz = 0.0, nZ_exc = 0.0, nZ_rec = 1.0)



    def subdigitize_signal(self, dig_time,  dig_time_res, sig_idx=2, after=True):
        ''' Sub-digitize a signal, identified by "sig_idx" within self.signals, 
        to a time resolution given by "dig_time_res" [s].

        If "after=True", subdigitize after a certain time "dig_time" [s], else do it before this time. 
        '''
        # signal dt, to us resolution
        dt_sig = np.round(np.mean(np.diff(self.signals[sig_idx].t)), 6) # round to us resolution

        ds_dig = int(dig_time_res/dt_sig)
        remainder = len(self.signals[sig_idx].t)%ds_dig

        if remainder!=0:
            t = np.mean(self.signals[sig_idx].t[:-remainder].reshape(-1,ds_dig),axis=1) 
            y = np.mean(self.signals[sig_idx].y[:-remainder,:].reshape(-1,ds_dig,self.signals[sig_idx].y.shape[1]),axis=1)
            y_norm = np.mean(
                self.signals[sig_idx].y_norm[:-remainder,:].reshape(-1,ds_dig,self.signals[sig_idx].y_norm.shape[1]),axis=1)
            std_y= np.sqrt(
                np.mean(self.signals[sig_idx].std_y[:-remainder,:].reshape(-1,ds_dig,self.signals[sig_idx].std_y.shape[1])**2,axis=1)+
                np.std(self.signals[sig_idx].y[:-remainder,:].reshape(-1,ds_dig,self.signals[sig_idx].y.shape[1]),axis=1)**2
                )
            std_y_norm = np.sqrt(
                np.mean(self.signals[sig_idx].std_y_norm[:-remainder,:].reshape(-1,ds_dig, self.signals[sig_idx].std_y_norm.shape[1])**2,axis=1)+
                np.std(self.signals[sig_idx].y_norm[:-remainder,:].reshape(-1,ds_dig,self.signals[sig_idx].y_norm.shape[1]),axis=1)**2
                )

        else:
            t = np.mean(self.signals[sig_idx].t.reshape(-1,ds_dig),axis=1) 
            y = np.mean(
                self.signals[sig_idx].y.reshape(-1,ds_dig,self.signals[sig_idx].y.shape[1]),axis=1)
            y_norm = np.mean(
                self.signals[sig_idx].y_norm.reshape(-1,ds_dig,self.signals[sig_idx].y_norm.shape[1]),axis=1)
            std_y = np.sqrt(
                np.mean(self.signals[sig_idx].std_y.reshape(-1,ds_dig,self.signals[sig_idx].std_y.shape[1])**2,axis=1)+
                np.std(self.signals[sig_idx].y.reshape(-1,ds_dig,self.signals[sig_idx].y.shape[1]),axis=1)**2
                )
            std_y_norm = np.sqrt(
                np.mean(self.signals[sig_idx].std_y_norm.reshape(-1,ds_dig,self.signals[sig_idx].std_y_norm.shape[1]),axis=1)+
                np.std(self.signals[sig_idx].y_norm.reshape(-1,ds_dig,self.signals[sig_idx].y_norm.shape[1]),axis=1)**2
                )

        ind_t = np.argmin(np.abs(self.signals[sig_idx].t - dig_time))
        ind_t_new = np.argmin(np.abs(t - dig_time))

        if after:
            # subdigitize after given time
            self.signals[sig_idx].t = np.concatenate((self.signals[sig_idx].t[:ind_t], t[ind_t_new:]))
            self.signals[sig_idx].y = np.concatenate((self.signals[sig_idx].y[:ind_t,:], y[ind_t_new:]))
            self.signals[sig_idx].y_norm = np.concatenate((self.signals[sig_idx].y_norm[:ind_t,:], y_norm[ind_t_new:]))
            self.signals[sig_idx].std_y = np.concatenate((self.signals[sig_idx].std_y[:ind_t,:], std_y[ind_t_new:]))
            self.signals[sig_idx].std_y_norm = np.concatenate((self.signals[sig_idx].std_y_norm[:ind_t,:], std_y_norm[ind_t_new:]))
        else:
            # subdigitize before given time
            self.signals[sig_idx].t = np.concatenate(( t[:ind_t_new], self.signals[sig_idx].t[ind_t:]))
            self.signals[sig_idx].y = np.concatenate((y[:ind_t_new], self.signals[sig_idx].y[ind_t:,:]))
            self.signals[sig_idx].y_norm = np.concatenate((y_norm[:ind_t_new], self.signals[sig_idx].y_norm[ind_t:,:]))
            self.signals[sig_idx].std_y = np.concatenate((std_y[:ind_t_new], self.signals[sig_idx].std_y[ind_t:,:]))
            self.signals[sig_idx].std_y_norm = np.concatenate((std_y_norm[:ind_t_new], self.signals[sig_idx].std_y_norm[ind_t:,:]))
        

    ####
    def get_geqdsk(self):
        ''' Load geqdsk file using omfit_eqdsk, either from disk or from MDS+ '''

        time_ms = self._t0 *1e3  # ms
        try:
            with open(f'/home/sciortino/BITS/{self.shot:d}/geqdsk_{self.shot:d}_{time_ms:.3f}.pkl','rb') as f:
                geqdsk = pkl.load(f)
        except:
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(device='CMOD',shot=self.shot,
                                                              time=time_ms,
                                                              SNAPfile='EFIT20',   
                                                              fail_if_out_of_range=False,time_diff_warning_threshold=20)

            with open(f'/home/sciortino/BITS/{self.shot:d}/geqdsk_{self.shot:d}_{time_ms:.3f}.pkl','wb') as f:
                pkl.dump(geqdsk,f)
            print('Stored geqdsk in BITS shot local directory')

        return geqdsk
     


    def create_aurora_namelist(self, cxr_flag=True, geqdsk=None):
        ''' Create a new aurora namelist using defaults '''

        # get default aurora "namelist" available
        namelist = aurora.default_nml.load_default_namelist()
        
        namelist['imp'] = 'Ca'

        # Atomic data files
        namelist['acd'] = "acd89_ca.type_a_large" #"acd85_ca.dat"
        namelist['scd'] = "scd50_ca.dat" #"scd85_ca.dat"
        namelist['ccd'] = "ccd89_w.dat"  # Aurora only uses first 20 charge states

        # superstaging:
        namelist['superstages'] = self.superstages if self.apply_superstaging else []

        # Now fill in namelist (all quantities in MKS)
        namelist['device'] = 'CMOD'
        namelist['shot'] = self.shot
        namelist['time'] = self._t0 *1e3  # ms
        namelist['Baxis'] = np.abs(self.efit_tree.rz2BT(self.R0_t,self.Z0_t, self._t0))
        namelist['K'] = self.K_spatial_grid

        # source
        namelist['source_type'] = 'file'
        namelist['source_file'] = self.source_file

        # shape of source: set both params to -1 to have delta source (some particles will be lost)
        namelist['source_width_in'] =0.0 # -1.0
        namelist['source_width_out'] = 0.0 #-1.0
        namelist['imp_energy'] = 3.0 # eV
        
        # sawtooth model
        saw = namelist['saw_model']
        saw['saw_flag'] = True if (len(self.sawteeth_times) and self.sawteeth_times[0]!=False) else False
        saw['rmix'] = self.rsaw_vol_cm  # rV cm
        saw['times'] = self.sawteeth_times[(self.sawteeth_times>self.time_1)*(self.sawteeth_times<self.time_2)]
        saw['crash_width'] = self.sawtooth_width_cm  # cm

        # timing settings: 
        timing = namelist['timing']
        timing['times'] = np.array([self.time_1] + list(saw['times']) + [self.time_2]) #s
        #timing['dt_increase'] = np.array([1.01] + [1.03,]*len(saw['times']) + [1.0])   
        #timing['dt_increase'] = np.array([1.005] + [1.01,]*len(saw['times']) + [1.0])    
        timing['dt_increase'] = np.array([1.01] + [1.01,]*len(saw['times']) + [1.0])   
        timing['dt_start'] = np.array([1e-5] + [1e-4,]*len(saw['times']) + [1e-4])   #s
        timing['steps_per_cycle'] = np.array([1.0] + [1.0,]*len(saw['times']) + [1.0])
                
        namelist['dr_0'] = self.dr_center   # cm
        namelist['dr_1'] = self.dr_edge     # cm
        namelist['SOL_decay'] = self.decay_length_boundary # cm
        
        # --------------------
        # edge/recycling
        #recycling_flag enables both recycling and divertor return flows
        namelist['recycling_flag'] = self.rcl>=0
        namelist['wall_recycling'] = self.rcl  # actual R value if flag>0, but if <0 turns off return flows
        namelist['divbls'] = 0.0

        namelist['tau_div_SOL'] = self.tau_divsol
        namelist['tau_pump'] = self.tau_pump
        namelist['tau_rcl_ret'] = self.tau_rcl_ret

        # ------------------
        # Fetch and process EFIT geqdsk using omfit_eqdsk
        if geqdsk is None and not hasattr(self, 'geqdsk'):
            self.geqdsk = self.get_geqdsk()  

        # set edge parameters via EFIT-based estimates
        clen_divertor, clen_limiter = aurora.estimate_clen(self.geqdsk)
        
        rbound_rlcfs, rlim_rlcfs = aurora.estimate_boundary_distance(self.shot, 'CMOD', self._t0*1e3)
    
        namelist['bound_sep'] = rbound_rlcfs
        namelist['lim_sep'] = rlim_rlcfs
        namelist['clen_divertor'] = clen_divertor
        namelist['clen_limiter'] = clen_limiter
        namelist['source_cm_out_lcfs'] = 0.0 # 3/9/21 Place source at LCFS to avoid influence of SOL transport and atomic physics on source

        # Kinetic profiles --- set on initial grid, roa_grid_in, to ensure that we can run this function more than once if useful
        kin_profs = namelist['kin_profs']

        kin_profs['ne']['rhop'] = self.efit_tree.rho2rho('r/a','sqrtpsinorm', self.roa_grid_in, namelist['time']/1e3)
        kin_profs['Te']['rhop'] = copy.deepcopy(kin_profs['ne']['rhop'] )
        
        # enforce exact zero on axis:
        kin_profs['ne']['rhop'][0] = 0.0
        kin_profs['Te']['rhop'][0] = 0.0 
    
        # treatment of SOL kinetic profiles in interpolation:
        kin_profs['ne']['fun'] = 'interpa'  # log-interpolation also outside LCFS
        kin_profs['Te']['fun'] = 'interp'    # log-interpolation only inside LCFS

        # decay length of 1cm in SOL for Te and Ti
        kin_profs['Te']['decay'] = np.ones(len(self.time_Te))*1.0   # use Te time base

        # use Te time base
        kin_profs['ne']['times'] = copy.deepcopy(self.time_Te)
        kin_profs['Te']['times'] = copy.deepcopy(self.time_Te)

        # interpolate all profiles on the same time grid (the Te one, being more detailed)
        kin_profs['ne']['vals'] = interp2d(self.roa_grid_in, self.time_ne, self.ne_cm3_in)(
            self.roa_grid_in, self.time_Te)
        kin_profs['Te']['vals'] = copy.deepcopy(self.Te_eV_in)

        # flags
        namelist['cxr_flag'] = cxr_flag
        namelist['nbi_cxr_flag'] = False

        if cxr_flag:
            # Add atomic D neutrals profiles [cm^-3] if provided
            kin_profs['n0'] = {}
            kin_profs['n0']['fun'] = 'interpa'
            kin_profs['n0']['rhop'] = self.roa_n0 #copy.deepcopy(kin_profs['ne']['rhop'] )
            kin_profs['n0']['times'] = copy.deepcopy(kin_profs['ne']['times'])
            kin_profs['n0']['vals'] =  np.tile(self.n0_cm3,(len(kin_profs['ne']['times']),1))
            
            # require Ti for CX rates evaluation
            kin_profs['Ti'] = {}
            kin_profs['Ti']['rhop'] = copy.deepcopy(kin_profs['ne']['rhop'] )
            kin_profs['Ti']['fun'] = 'interp'     # log-interpolation only inside LCFS
            kin_profs['Ti']['decay'] = np.ones(len(self.time_Te))*1.0    # use Te time base            
            kin_profs['Ti']['times'] = copy.deepcopy(self.time_Te)
            kin_profs['Ti']['vals'] = copy.deepcopy(kin_profs['Te']['vals']) # set Ti=Te everywhere, only used for CX (eV)

        # radiation flags
        #namelist['prad_flag'] = False
        #namelist['thermal_cx_rad_flag'] = False
        #namelist['spectral_brem_flag'] = False
        #namelist['sxr_flag'] = False
        #namelist['main_ion_brem_flag'] = False

        return namelist



    def setup_aurora(self, namelist=None, cxr_flag=True, update_grids=True):
        ''' 
        Setup aurora simulations.

        INPUTS:
        namelist: aurora namelist. If left to None, this is created using defaults. 
             A simple way to modify this is to first load the default and then change only components of it as desired.
        cxr_flag : if True, uses atomic hydrogen neutrals stored in self.n0 to include charge exchange recombination
            in the effective recombination rate for impurities. 
        update_grids : if True, update BITS spatial and temporal grids to be the same as those in the aurora simulation
            setup.
        '''      
        if cxr_flag: assert hasattr(self,'n0_cm3') # n0 profiles must be available for CXR to be included

        # Fetch and process EFIT geqdsk using omfit_eqdsk
        if not hasattr(self, 'geqdsk'):
            self.geqdsk = self.get_geqdsk() 

        if namelist is None:
            self.namelist = self.create_aurora_namelist(cxr_flag=cxr_flag, geqdsk=self.geqdsk)
        else:
            self.namelist = namelist
        imp = self.namelist['imp'] 

        # -------------------------------------------
        # now use the namelist and geqdsk data to setup aurora simulation
        asim = self.asim = aurora.core.aurora_sim(self.namelist, geqdsk=self.geqdsk)

        if cxr_flag:
            # store some info about RR+DR and CXR -- this is useful for fast iteration when modifying neutral density between iterations
            lne = np.log10(asim._ne)
            lTe = np.log10(asim._Te)
            lTi = np.log10(asim._Ti)
            
            self.R_rates = aurora.interp_atom_prof(asim.atom_data['acd'], lne, lTe, x_multiply=True)
            self.alpha_CX_rates = aurora.interp_atom_prof(asim.atom_data['ccd'], lne, lTi, x_multiply=False)

            # store original "_n0" field within the aurora_sim object for possible rescaling
            self.asim_n0 = copy.deepcopy(asim._n0)

            # get electron impact ionization and radiative recombination rates in units of [s^-1]
            _, self.asim_S, self.asim_R, self.asim_cx = aurora.get_cs_balance_terms(asim.atom_data, ne_cm3=asim._ne, 
                                                      Te_eV = asim._Te, Ti_eV= asim._Ti, include_cx=asim.namelist['cxr_flag'])

            #if len(superstages):
            #    self.superstages, R, S, self.fz_upstage = \
            #                                              atomic.superstage_rates(R, S, superstages,save_time=self.save_time)

        # -------------------------------------------
        print(f'Setup aurora simulations with {len(asim.rvol_grid)} radial points and {len(asim.time_out)} time points')

        if update_grids:
         
            # aurora radial and time grids:
            self.time = copy.deepcopy(asim.time_out)
            self.rhop = copy.deepcopy(asim.rhop_grid)

            # transform aurora rhop grid to rmid/a grid 
            roa_grid_out = self.efit_tree.rho2rho('sqrtpsinorm', 'r/a', self.rhop,(self.time_1 + self.time_2) / 2.0)
            roa_grid_out[0] = 0.0

            # interpolate ne, Te profiles on r,t grids of aurora
            self.ne_cm3 = interp2d(self.roa_grid_in,self.time_ne,self.ne_cm3_in)(roa_grid_out,self.time)
            self.Te_eV = interp2d(self.roa_grid_in,self.time_Te,self.Te_eV_in)(roa_grid_out,self.time)
            self.Ti_eV = interp1d(self.rhop_in, self.Ti_eV_in)(self.rhop)
            self.omega_s = interp1d(self.rhop_in, self.omega_s_in)(self.rhop)
            self.vi_ms = interp1d(self.rhop_in, self.vi_ms_in)(self.rhop)

            # update radial grid for inference to be the one of aurora
            self.roa_grid = copy.deepcopy(roa_grid_out)
            self.roa_grid_DV = copy.deepcopy(self.roa_grid)

            # Cache SXR rates
            if self.signal_mask[2]:
                
                # Use Puetterich atomic data to find SXR radiation rates for the 50um Be filter specific to XTOMO
                atom_data = aurora.get_atom_data('Ca', {'pls': self.pls_prs_loc+self.pls_file_Ca})
                self.pls = aurora.interp_atom_prof(atom_data['pls'], np.log10(self.ne_cm3), np.log10(self.Te_eV)) # W
                
                atom_data = aurora.get_atom_data('Ca', {'prs': self.pls_prs_loc+self.prs_file_Ca})
                self.prs = aurora.interp_atom_prof(atom_data['prs'], np.log10(self.ne_cm3), np.log10(self.Te_eV)) # W


    @property
    def num_params(self):
        return len(self.fixed_params)
    
    @property
    def num_free_params(self):
        """Returns the number of free parameters.
        """
        return sum(~self.fixed_params)
    
    @property
    def free_param_idxs(self):
        """Returns the indices of the free parameters in the main arrays of parameters, etc.
        """
        return scipy.arange(0, self.num_params)[~self.fixed_params]
    
    @property
    def free_params(self):
        """Returns the values of the free parameters.
        
        Returns
        -------
        free_params : :py:class:`Array`
            Array of the free parameters, in order.
        """
        return np.array([self.params[idx] for idx in self.free_param_idxs])
    
    @free_params.setter
    def free_params(self, value):
        self.params[self.free_param_idxs] = np.asarray(value, dtype=float)
    
    
    @property
    def free_param_names(self):
        """Returns the names of the free parameters.
        
        Returns
        -------
        free_param_names : :py:class:`Array`
            Array of the names of the free parameters, in order.
        """
        return np.array([self.param_names[idx] for idx in self.free_param_idxs])
    
    @property
    def param_names(self):
        return np.asarray(self.get_labels(), dtype=str)
    
    @property
    def free_D_knots(self):
        return (~self.fixed_params[self.nD+self.nV:self.nD+self.nV+self.nkD]).any()

    @property
    def free_V_knots(self):
        return (~self.fixed_params[self.nD+self.nV+self.nkD:self.nD+self.nV+self.nkD +self.nkV]).any()

    @free_D_knots.setter
    def free_D_knots(self, val):
        self.fixed_params[self.nD+self.nV:self.nD+self.nV+self.nkD] = not val

    @free_V_knots.setter
    def free_V_knots(self, val):
        #if self.equal_DV_knots:  # let only D knots be free; see also self.split_params()
        #    self.fixed_params[self.nD+self.nV+self.nkD:self.nD+self.nV+self.nkD+self.nkV] = [False,]*self.nkV
        #else:
        self.fixed_params[self.nD+self.nV+self.nkD:self.nD+self.nV+self.nkD+self.nkV] = not val

    #####

    @property
    def free_time_shift(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV:self.nD+self.nV+self.nkD +self.nkV +self.nDiag]
    
    @free_time_shift.setter
    def free_time_shift(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV:self.nD+self.nV+self.nkD +self.nkV +self.nDiag] = ~np.asarray(bools, dtype=bool)
    
    @property
    def time_shifts(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV:self.nD+self.nV+self.nkD +self.nkV +self.nDiag]
    
    @time_shifts.setter
    def time_shifts(self, vals):
        self.params[self.nD+self.nV+self.nkD +self.nkV:self.nD+self.nV+self.nkD +self.nkV +self.nDiag] = np.asarray(vals, dtype=float)
    

    ###############
    @property
    def d_weights(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW]
        
    @d_weights.setter
    def d_weights(self, vals):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag:
                    self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW]=np.asarray(vals,dtype=float)

    @property
    def free_d_weights(self):
        return (~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW]).any()
        
    @free_d_weights.setter
    def free_d_weights(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW] = not bools

    ###############
    @property
    def mix_rad_corr(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW+ 1]
        
    @mix_rad_corr.setter
    def mix_rad_corr(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW + 1]=np.asarray(val,dtype=float)

    @property
    def free_mixing_radius(self):
        return (~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW + 1]).any()
        
    @free_mixing_radius.setter
    def free_mixing_radius(self, bool):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW +1] = not bool


    ##################

    @property
    def gaussian_D_amp(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +1:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +2]
        
    @gaussian_D_amp.setter
    def gaussian_D_amp(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +1:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +2]=np.asarray(val,dtype=float)

    @property
    def gaussian_D_w(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +2:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +3]
        
    @gaussian_D_w.setter
    def gaussian_D_w(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +2:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +3]=np.asarray(val,dtype=float)

    @property
    def gaussian_D_r(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +3:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4]
        
    @gaussian_D_r.setter
    def gaussian_D_r(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +3:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4]=np.asarray(val,dtype=float)

    ##################

    @property
    def gaussian_V_amp(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +5]
        
    @gaussian_V_amp.setter
    def gaussian_V_amp(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +5]=np.asarray(val,dtype=float)

    @property
    def gaussian_V_w(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +5:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +6]
        
    @gaussian_V_w.setter
    def gaussian_V_w(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +5:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +6]=np.asarray(val,dtype=float)

    @property
    def gaussian_V_r(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +6:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7]
        
    @gaussian_V_r.setter
    def gaussian_V_r(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +6:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7]=np.asarray(val,dtype=float)

    ####################

    @property
    def free_gaussian_Dped(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +1:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4]
        
    @free_gaussian_Dped.setter
    def free_gaussian_Dped(self, vals):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +1:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4] = ~vals   # negation works for each np.ndarray element

    #####################
    @property
    def free_gaussian_Vped(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7]
        
    @free_gaussian_Vped.setter
    def free_gaussian_Vped(self, vals):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +4:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7] = ~vals  

    ###############
    @property
    def sawtooth_width(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7:
                           self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8]
                                   
        
    @sawtooth_width.setter
    def sawtooth_width(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8]=np.asarray(val,dtype=float)

    @property
    def free_sawtooth_width(self):
        return (~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8]).any()
        
    @free_sawtooth_width.setter
    def free_sawtooth_width(self, bool):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +7:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8] = not bool
    
        
    ############
    # Recycling
    @property
    def free_rcl(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8+1] 
        
    @free_rcl.setter
    def free_rcl(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8+1] = ~np.array(bools)

    @property
    def rcl_val(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8:
                           self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8+1]

    @rcl_val.setter
    def rcl_val(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +8+1]=np.asarray(val,dtype=float)

    ########
    @property
    def free_tau_divsol(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9+1] 
        
    @free_tau_divsol.setter
    def free_tau_divsol(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9+1] = ~np.array(bools)

    @property
    def tau_divsol_val(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9:
                           self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9+1]

    @tau_divsol_val.setter
    def tau_divsol_val(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +9+1]=np.asarray(val,dtype=float)

    ########
    @property
    def free_n0x(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1] 
        
    @free_n0x.setter
    def free_n0x(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1] = ~np.array(bools)

    @property
    def n0x(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10:
                           self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1]

    @n0x.setter
    def n0x(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1]=np.asarray(val,dtype=float)

    ########
    @property
    def free_zeta(self):
        return ~self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1:
                                   self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1+1] 
        
    @free_zeta.setter
    def free_zeta(self, bools):
        self.fixed_params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1+1] = ~np.array(bools)

    @property
    def zeta(self):
        return self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1:
                           self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1+1]

    @zeta.setter
    def zeta(self, val):
        self.params[self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1:
                          self.nD+self.nV+self.nkD +self.nkV +self.nDiag +self.nW +10+1+1]=np.asarray(val,dtype=float)


    ###############
    @property
    def knotgrid_D(self):
        """Grid of knots to use when evaluating the D profile.
        
        Takes the (internal) knots given in :py:attr:`self.params` and puts the
        boundary knots given by the extreme values of :py:attr:`self.roa_grid_DV`
        at either end.
        """
        D_vals, V_vals, knots_D, knots_V, param_tshifts,d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
            rcl, tau_divsol, n0x, zeta = self.split_params()
        
        return scipy.concatenate(([self.roa_grid_DV[0],], knots_D, [self.roa_grid_DV[-1],]))
        #return scipy.concatenate(([self.roa_grid_DV[0],], knots_D, [self.ped_roa_DV,]))   # only let splines go until pedestal top
    

    @knotgrid_D.setter
    def knotgrid_D(self, value):
        """ Set the knot-grid for D-coefficients to specific locations.

        The user should give the complete grid. Only the internal knots will 
        be saved in the self.params structure.
        """
        self.params[self.nD+self.nV:self.nD+self.nV+self.nkD] = np.asarray(value[1:-1], dtype=float)

    @property
    def knotgrid_V(self):
        """Grid of knots to use when evaluating the V profile.
        
        Takes the (internal) knots given in :py:attr:`self.params` and puts the
        boundary knots given by the extreme values of :py:attr:`self.roa_grid_DV`
        at either end.
        """
        D_vals, V_vals, knots_D, knots_V, param_tshifts, d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
            rcl, tau_divsol, n0x, zeta = self.split_params()
        
        return scipy.concatenate(([self.roa_grid_DV[0],], knots_V, [self.roa_grid_DV[-1],]))
        #return scipy.concatenate(([self.roa_grid_DV[0],], knots_V, [self.ped_roa_DV,]))
    
    @knotgrid_V.setter
    def knotgrid_V(self, value):
        """ Set the knot-grid for V-coefficients to specific locations.

        The user should give the complete grid. Only the internal knots will 
        be saved in the self.params structure.
        """
        self.params[self.nD+self.nV+self.nkD:self.nD+self.nV+self.nkD +self.nkV] = np.asarray(value[1:-1], dtype=float)





    # =====================

    def set_diag_time_res(self):
        ''' Cache details about finite time resolution of diagnostics
        '''
        # get time resolution from the data. Fill in fields for all diagnostics to conform with other BITS conventions
        self.diag_time_res = np.zeros((3))
        
        if self.params_true is not None:
            print('NB: setting hirex_readout_time of 3ms! Change this for synthetic data as desirable.')

        hirex_readout_time = 0.003   # cannot have readout time larger than time resolution in synth data (real res: 6ms)
        vuv_readout_time = 0.0
        xtomo_readout_time = 0.0

        if self.signal_mask[0]:                
            self.diag_time_res[0] = [round(i,3) for i in np.diff(self.signals[0].t)][0] - hirex_readout_time
            
        if self.signal_mask[1]:    
            self.diag_time_res[1] =[round(i,3) for i in np.diff(self.signals[1].t)][0] - vuv_readout_time
            
        if self.signal_mask[2]:
            self.diag_time_res[2] = [round(i,6) for i in np.diff(self.signals[2].t)][0] - xtomo_readout_time
        
        if (self.diag_time_res<0).any():
            raise ValueError('Set negative diagnostic time resolution! Check for errors.')


    def correct_signals(self):
        ''' Correct signal uncertainties by setting minumum values  based on user input 
        '''
        if self.signal_mask[0]:                
            # Shift Hirex-Sr timebase
            self.signals[0].t=self.signals[0].t + self.Hirex_time_shift 

        if self.signal_mask[1]:    
            # Shift XEUS & LoWEUS timebase
            self.signals[1].t=self.signals[1].t + self.VUV_time_shift
            
        # no need to shift SXR -- time base is very accurate
        
        # ---------------------------
        # Safe way of setting minimum uncertainties without issues with nan's: loop
        if self.signal_mask[0]:
            # Attempt correction for underestimated Hirex-Sr uncertainties 
            HIREX_rel_norm_unc = self.signals[0].std_y_norm/(self.signals[0].y_norm+1e-10)
            for ll in np.arange(HIREX_rel_norm_unc.shape[0]): # loop over wavelengths
                for ii in np.arange(HIREX_rel_norm_unc.shape[1]):  #loop over times
                    for jj in np.arange(HIREX_rel_norm_unc.shape[2]):   # loop over chords
                        if ~np.isnan(HIREX_rel_norm_unc[ll,ii,jj]) and HIREX_rel_norm_unc[ll,ii,jj]<self.Hirex_min_rel_unc:
                            self.signals[0].std_y_norm[ll,ii,jj] = self.Hirex_min_rel_unc * np.abs(self.signals[0].y_norm[ll,ii,jj])  # NB abs here

            # min absolute value of uncertainty for Hirex-Sr
            for ll in np.arange(self.signals[0].y_norm.shape[0]):
                for ii in np.arange(self.signals[0].y_norm.shape[1]): # loop in time
                    for jj in np.arange(self.signals[0].y_norm.shape[2]):   # loop over chords
                        if ~np.isnan(self.signals[0].std_y_norm[ll,ii,jj]) and self.signals[0].std_y_norm[ll,ii,jj] < self.Hirex_min_abs_unc:
                            self.signals[0].std_y_norm[ll,ii,jj] = self.Hirex_min_abs_unc


        if self.signal_mask[1]:
            # Correction for underestimated XEUS uncertainties 

            for ll in np.arange(self.num_vuv_lines):  # for each recorded line on XEUS and LoWEUS
                rel_norm_unc = self.signals[1].std_y_norm[:,ll]/(self.signals[1].y_norm[:,ll]+1e-10)

                if self.signals[1].name[ll].lower()=='xeus':
                    min_abs_unc = copy.deepcopy(self.XEUS_min_abs_unc)
                    min_rel_unc = copy.deepcopy(self.XEUS_min_rel_unc)
                elif self.signals[1].name[ll].lower()=='loweus':
                    min_abs_unc = copy.deepcopy(self.LoWEUS_min_abs_unc)
                    min_rel_unc = copy.deepcopy(self.LoWEUS_min_rel_unc)
                else:
                    raise ValueError('Unrecognized VUV diagnostic!')

                for i in np.arange(len(self.signals[1].y_norm[:,ll])):  # loop over each data point

                    # min relative value of uncertainty
                    if ~np.isnan(rel_norm_unc[i]) and rel_norm_unc[i]< min_rel_unc:
                        self.signals[1].std_y_norm[i,ll] = min_rel_unc * self.signals[1].y_norm[i,ll]

                    # min absolute value of uncertainty 
                    if ~np.isnan(self.signals[1].std_y_norm[i,ll]) and self.signals[1].std_y_norm[i,ll]< min_abs_unc:
                        self.signals[1].std_y_norm[i,ll] = min_abs_unc

        if self.signal_mask[2]:
            # correction for XTOMO uncertainties
            XTOMO_rel_norm_unc = self.signals[2].std_y_norm/(self.signals[2].y_norm+1e-10)
            for i in np.arange(XTOMO_rel_norm_unc.shape[0]):  #loop over times
                for j in np.arange(XTOMO_rel_norm_unc.shape[1]):   # loop over chords
                    if ~np.isnan(XTOMO_rel_norm_unc[i,j]) and XTOMO_rel_norm_unc[i,j]<self.XTOMO_min_rel_unc:
                            self.signals[2].std_y_norm[i,j] = self.XTOMO_min_rel_unc * self.signals[2].y_norm[i,j]
                            
            # min absolute value of uncertainty for XTOMO
            for i in np.arange(self.signals[2].y_norm.shape[0]): # loop in time
                for j in np.arange(self.signals[2].y_norm.shape[1]):   # loop over chords
                    if ~np.isnan(self.signals[2].std_y_norm[i,j]) and self.signals[2].std_y_norm[i,j] < self.XTOMO_min_abs_unc:
                            self.signals[2].std_y_norm[i,j] = self.XTOMO_min_abs_unc       

        # -------------------------------

        if self.signal_mask[0]:
            # Multiplier for all experimental uncertainties  (AFTER min_rel and _abs corrections!)
            self.signals[0].std_y_norm = self.signals[0].std_y_norm * self.Hirex_unc_multiplier
            self.signals[0].std_y = self.signals[0].std_y * self.Hirex_unc_multiplier
            
            # Reinforce importance of Hirex data in the signal rise by reducing their uncertainty  (AFTER min_rel and _abs corrections!)
            self.signals[0].std_y_norm[:5,:] /= self.sig_rise_enforce   
            self.signals[0].std_y[:5,:] /= self.sig_rise_enforce   

        if self.signal_mask[1]:
            # do the same for XEUS, but do more points because first 3-4 are actually just 0 (AFTER min_rel and _abs corrections!)
            self.signals[1].std_y_norm[:10,:] /= self.sig_rise_enforce   
            self.signals[1].std_y[:10,:] /= self.sig_rise_enforce   


        print("*************************************")
        print("Set minimum relative and absolute uncertainties!")
        print("*************************************")
    # ==================


    def apply_noise(self, noises=[0.03, 0.03, 0.1], noise_type='proportional Gaussian'):
        """Apply random noise to the data.
        
        Parameters
        ----------
        noises : array of float, optional
            The relative noise level to apply to each signal in
            :py:attr:`self.signals`. The first element is also used for the
            argon data. Default is [0.03, 0.03, 0.1].
        noise_type: {'proportional Gaussian', 'Poisson'}
            The noise type to use. Options are:
            
            * 'proportional Gaussian': Gaussian noise for which the standard
              deviation is equal to the relative noise level times the value.
            * 'Poisson' : Gaussian noise for which the standard deviation is
              equal to the relative noise level times the value divided by the
              square root of the ratio of the value to the max value. This
              simulates Poisson noise.
        """
        for i, (n, s) in enumerate(zip(noises, self.signals)):
            if s is not None:
                if noise_type == 'proportional Gaussian':
                    s.y = self.truth_data.sig_abs[i] * (1.0 + n * scipy.randn(*self.truth_data.sig_abs[i].shape))
                    s.y[s.y < 0.0] = 0.0
                    s.std_y = n * self.truth_data.sig_abs[i]
                    s.std_y[s.std_y <= 1e-4 * np.nanmax(s.y)] = 1e-4 * np.nanmax(s.y)
                    
                    s.y_norm = self.truth_data.sig_norm[i] * (1.0 + n * scipy.randn(*self.truth_data.sig_norm[i].shape))
                    s.y_norm[s.y < 0.0] = 0.0
                    s.std_y_norm = n * self.truth_data.sig_norm[i]
                    s.std_y_norm[s.std_y_norm <= 1e-4 * np.nanmax(s.y_norm)] = 1e-4 * np.nanmax(s.y_norm)
                elif noise_type == 'Poisson':
                    sig_max = np.nanmax(self.truth_data.sig_abs[i])
                    s.std_y = n * scipy.sqrt(sig_max * self.truth_data.sig_abs[i])
                    s.std_y[s.std_y <= 1e-4 * sig_max] = 1e-4 * sig_max
                    s.y = self.truth_data.sig_abs[i] + s.std_y * scipy.randn(*self.truth_data.sig_abs[i].shape)
                    s.y[s.y < 0.0] = 0.0
                    
                    sig_max_norm = np.nanmax(self.truth_data.sig_norm[i])
                    s.std_y_norm = n * scipy.sqrt(sig_max_norm * self.truth_data.sig_norm[i])
                    s.std_y_norm[s.std_y_norm <= 1e-4 * sig_max_norm] = 1e-4 * sig_max_norm
                    s.y_norm = self.truth_data.sig_norm[i] + s.std_y_norm * scipy.randn(*self.truth_data.sig_norm[i].shape)
                    s.y_norm[s.y_norm < 0.0] = 0.0
                else:
                    raise ValueError("Unknown noise type!")
        
 
        # Needs special handling since sig_*_ar just has a single timepoint:
        if hasattr(self, 'ar_signal') and len(self.ar_signal):


            # FS : TO UPDATE FOR W AND Z LINES!


            if noise_type == 'proportional Gaussian':
                self.ar_signal.y[:, :] = self.truth_data.sig_abs_ar * (1.0 + noises[0] * scipy.randn(*self.ar_signal.y.shape))
                self.ar_signal.y[self.ar_signal.y < 0.0] = 0.0
                self.ar_signal.std_y[:, :] = noises[0] * self.truth_data.sig_abs_ar
                self.ar_signal.std_y[self.ar_signal.std_y < 1e-4 * np.nanmax(self.ar_signal.y)] = 1e-4 * np.nanmax(self.ar_signal.y)
                
                self.ar_signal.y_norm[:, :] = self.truth_data.sig_norm_ar * (1.0 + noises[0] * scipy.randn(*self.ar_signal.y.shape))
                self.ar_signal.y_norm[self.ar_signal.y_norm < 0.0] = 0.0
                self.ar_signal.std_y_norm[:, :] = noises[0] * self.truth_data.sig_norm_ar
                self.ar_signal.std_y_norm[self.ar_signal.std_y_norm < 1e-4 * np.nanmax(self.ar_signal.y_norm)] = 1e-4 * np.nanmax(self.ar_signal.y_norm)
            elif noise_type == 'Poisson':
                sig_max_ar = np.nanmax(self.truth_data.sig_abs_ar)
                self.ar_signal.std_y[:, :] = noises[0] * scipy.sqrt(sig_max_ar * self.truth_data.sig_abs_ar)
                self.ar_signal.std_y[self.ar_signal.std_y < 1e-4 * sig_max_ar] = 1e-4 * sig_max_ar
                self.ar_signal.y[:, :] = self.truth_data.sig_abs_ar + self.ar_signal.std_y * scipy.randn(*self.truth_data.sig_abs_ar.shape)
                self.ar_signal.y[self.ar_signal.y < 0.0] = 0.0
                
                sig_max_ar_norm = np.nanmax(self.truth_data.sig_norm_ar)
                self.ar_signal.std_y_norm[:, :] = noises[0] * scipy.sqrt(sig_max_ar_norm * self.truth_data.sig_norm_ar)
                self.ar_signal.std_y_norm[self.ar_signal.std_y_norm < 1e-4 * sig_max_ar_norm] = 1e-4 * sig_max_ar_norm
                self.ar_signal.y_norm[:, :] = self.truth_data.sig_norm_ar + self.ar_signal.std_y_norm * scipy.randn(*self.truth_data.sig_norm_ar.shape)
                self.ar_signal.y_norm[self.ar_signal.y_norm < 0.0] = 0.0
            else:
                raise ValueError("Unknown noise type!")

                
    
    def eval_DV_splines(self, params=None, plot=False, lc=None, label=''):
        """Evaluate the D, V profiles for the given parameters.

        Note that input parameters in self.free_params might refer to (D, V/D) or (D, V), 
        but the output of eval_DV_splines will always be (D,V). 
        
        Parameters
        ----------
        params : array of float
            The parameters to evaluate at.
        plot : bool, optional
            If True, a plot of D, V and V/D will be produced. Default is False.
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params
        
        # transform parameters still requiring mapping to physical space
        D_vals = self.params[:self.nD]
        V_vals = self.params[self.nD:self.nD+self.nV]        
        
        if not np.allclose(self.knotgrid_D, np.sort(self.knotgrid_D)) or not np.allclose(self.knotgrid_V, np.sort(self.knotgrid_V)):
            # knotgrid is not sorted, supposedly because force_identifiability=True and this function was called without 
            # multinest_prior, which is where sorting should occur.
            print('Warning! Knotgrids were not sorted.')
            self.knotgrid_D = np.sort(self.knotgrid_D)
            self.knotgrid_V = np.sort(self.knotgrid_V)

        # interpolate profiles
        if self.fix_D_to_tglf_neo:
            D = interp1d(self.roa_grid_models, self.D_models, bounds_error=False,\
                         fill_value=(self.D_models[0], self.D_models[-1]))(self.roa_grid_DV)
        else:
            D = get_D_interpolation(self.method, self.knotgrid_D, D_vals, self.roa_grid_DV)

        if self.fix_V_to_tglf_neo:
            # obtain V from TGLF+NEO interpolated models
            V = interp1d(self.roa_grid_models, self.V_models, bounds_error=False,\
                         fill_value=(self.V_models[0], self.V_models[-1]))(self.roa_grid_DV)

        else:
            # interpolate either V or V/D, based on which one is sampled (don't interpolate ratios of sampled parameters
            # or else strange features will be obtained...)
            _V = get_V_interpolation(self.method, self.knotgrid_V, V_vals, self.roa_grid_DV)

            # this function must always return V (m/s), not V/D
            V = _V * D if self.learn_VoD else _V

        if plot:
            f = plt.figure()
            a_D = f.add_subplot(3, 1, 1)
            a_V = f.add_subplot(3, 1, 2, sharex=a_D)
            a_VD = f.add_subplot(3, 1, 3, sharex=a_D)
            a_D.set_xlabel('$r/a$')
            a_V.set_xlabel('$r/a$')
            a_VD.set_xlabel('$r/a$')
            a_D.set_ylabel('$D$ [m$^2$/s]')
            a_V.set_ylabel('$V$ [m/s]')
            a_VD.set_ylabel('$V/D$ [1/m]')
            
            a_D.plot(self.roa_grid, D, color=lc, label=label + ' splines')
            a_V.plot(self.roa_grid, V, color=lc, label=label + ' splines')
            a_VD.plot(self.roa_grid, V / D, color=lc, label=label+ ' splines')

            a_D.legend(loc='best').set_draggable(True)

        return D,V
    




    def eval_DV(self, params=None, imp='Ca', plot=False, lc=None, label='', 
                apply_D_sawteeth=False):
        """Evaluate the D, V profiles for the given parameters, combining spline values,
        possible gaussian pedestal features and sawtooth variations. 

        Note that input parameters in self.free_params might refer to (D, V/D), 
        but the output of eval_DV will always be (D,V). 
        All nomenclature follows (D,V) inference, but if self.learn_VoD==True the function
        eval_DV_splines() internally converts parameters to D,V. 
        Gaussian pedestal parameters and sawtooth variations always refer to V, not V/D.
        
        Parameters
        ----------
        params : array of float
            The parameters to evaluate at.
        imp : str
            Impurity atomic symbol.
        plot : bool, opt
            If True, a plot of D and V will be produced. Default is False.
        lc : str, opt
            Line color, only used if plot=True
        label : str, opt
            Label used in plotting, if plot=True
        apply_D_sawteeth : bool, opt
            Modify D at times where sawteeth are expected, to reproduce their phenomenology 
            via a diffusive effect. This may need further testing...        
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params

        # Evaluate D and V from splines
        D, V = self.eval_DV_splines(plot=plot)

        # form time dept D,V
        times_DV = [self.time_1]
        D_tot = [D]
        V_tot = [V]

        if apply_D_sawteeth:  # full diffusive sawteeth
            # not fully tested. Parameters currently arbitrarily fixed
            D_fs_amp = 10.0; width=0.5*self.asim.mixing_radius  # width of half mixing radius
            gauss = D_fs_amp * np.exp(-(self.asim.rvol_grid - 0.0)**2/(2.* width**2))
            D_gauss = np.array([round(val,5) for val in gauss])
            D_saw = D+D_gauss
            D_saw[0] = D_saw[1] 

            for fs_time in self.sawteeth_times:
                pre_saw_t = self.asim.time_grid[np.argmin(np.abs(self.asim.time_grid-fs_time))-1]
                times_DV.append(pre_saw_t)
                times_DV.append(fs_time)
                post_saw_t = self.asim.time_grid[np.argmin(np.abs(self.asim.time_grid-fs_time))+1]  # time steps in total
                times_DV.append(post_saw_t)

            for fs_time in self.sawteeth_times:
                D_tot.append(D); V_tot.append(V)
                D_tot.append(D_saw); V_tot.append(V)
                D_tot.append(D); V_tot.append(V)

            # if using this option, turn off sawtooth flattening model inside aurora
            self.asim.saw_on = np.zeros_like(self.asim.saw_on)

        # sort times
        sort_idxs = np.argsort(times_DV)
        times_DV = np.array(times_DV)[sort_idxs]
        D_tot = np.array(D_tot)[sort_idxs,:]
        V_tot = np.array(V_tot)[sort_idxs,:]

        # -------------------
        # first set all charge states to have the same transport (include elements for neutrals)
        D_z = np.tile(np.atleast_3d(D),(1,1,self.asim.Z_imp+1)).transpose(1,0,2)  
        V_z = np.tile(np.atleast_3d(V),(1,1,self.asim.Z_imp+1)).transpose(1,0,2)  

        # -----------------------------
        if self.use_gaussian_Dped:   
            # add gaussian D feature in the pedestal
            D_amp = self.gaussian_D_amp if self.use_gaussian_Dped else 0.0

            gauss = D_amp * np.exp(-(self.roa_grid_DV - self.gaussian_D_r)**2/(2.* self.gaussian_D_w**2))
            D_gauss = np.array([round(val,5) for val in gauss]) # get rid of extremely small digits/accuracy

            if self.scale_gaussian_Dped_by_Z:
                # simple z-scaling of Gaussian D pedestal  (D decreases linearly with Z)
                Z = np.arange(1,self.asim.Z_imp+1)
                
                # option to make D_gauss magnitude decrease with Z when it's positive:
                D_z[:,:,1:] +=  D_gauss[:,None,None] *( 1. - (Z - 1.)/self.asim.Z_imp)   # no effect on neutrals

                # option to make D_gauss magnitude increase with Z when it's negative:
                #D_z[:,:,1:] +=  D_gauss[:,None,None] * Z/self.asim.Z_imp  # no effect on neutrals
            else:
                D_z+=D_gauss[:,None,None]   # same for all times and all charge states
            D_z[0,:,:] = D_z[1,:,:] # boundary condition grad(D)=0 at center
                
            # 10/22/20: enforce minimum of D (max charge state) to prevent gaussian feature from making D<0
            D_z[D_z<self.D_lb] = self.D_lb

        if self.use_gaussian_Vped:  
            # gaussian Vped must be added here, because it represents a V/D quantity
            gauss = self.gaussian_V_amp * np.exp(-(self.roa_grid_DV - self.gaussian_V_r)**2/(2.* self.gaussian_V_w**2))
            V_gauss = np.array([round(val,5) for val in gauss])

            if self.scale_gaussian_Vped_by_Z:
                # z-scaling of Gaussian v pedestal. Exact scaling is set by zeta (fixed or free) parameter
                Z = np.arange(1,self.asim.Z_imp+1)
                V_z[:,:,1:] +=  V_gauss[:,None,None]*(Z/self.asim.Z_imp)**self.zeta
            else:
                V_z += V_gauss[:,None,None]
            V_z[0,:,:] = 0.0   # boundary condition v(0)=0


        if plot:
            f = plt.figure()
            a_D = f.add_subplot(3, 1, 1)
            a_V = f.add_subplot(3, 1, 2, sharex=a_D)
            a_VD = f.add_subplot(3, 1, 3, sharex=a_D)
            a_D.set_xlabel('$r/a$')
            a_V.set_xlabel('$r/a$')
            a_VD.set_xlabel('$r/a$')
            a_D.set_ylabel('$D$ [m$^2$/s]')
            a_V.set_ylabel('$V$ [m/s]')
            a_VD.set_ylabel('$V/D$ [1/m]')
            
            a_D.plot(self.roa_grid, D_z[:,0,1], color=lc, label=label + ' Z=1, tidx=0')
            a_V.plot(self.roa_grid, V_z[:,0,1], color=lc, label=label + ' Z=1, tidx=0')
            a_VD.plot(self.roa_grid, V_z[:,0,1] / D_z[:,0,1], color=lc, label=label + ' Z=1, tidx=0')
            a_D.plot(self.roa_grid, D_z[:,0,-1], color=lc, label=label + f' Z={D_z.shape[-1]-1}, tidx=0')
            a_V.plot(self.roa_grid, V_z[:,0,-1], color=lc, label=label + f' Z={D_z.shape[-1]-1}, tidx=0')
            a_VD.plot(self.roa_grid, V_z[:,0,-1] / D_z[:,0,1], color=lc, label=label + f' Z={D_z.shape[-1]-1}, tidx=0')
            
            a_D.legend(loc='best').set_draggable(True)
        
        return D_z, V_z, times_DV



    
    def split_params(self, params=None):
        """Split the given param vector into its constituent parts.
        
        Any parameters which are infinite are set to `1e-100 * sys.float_info.max`.
        
        Refer to the code itself to see how parameters are unwrapped in practice.

        Parameters
        ----------
        params : array of float, (`num_params`,), optional
            The parameter vector to split. If not provided,
            :py:attr:`self.params` is used. This vector should contain all of
            the parameters, not just the free parameters.
        
        Returns
        -------
        split_params : tuple
            Tuple of arrays of params, split as described above.
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                # this is the expected input: full list of parameters, free and fixed
                self.params = params
            elif len(params) == self.num_free_params:
                # assume that user passed vector of free parameters (instructed not to do so, but... just to make sure). 
                self.free_params = params
            else:
                raise ValueError('Number of params not recognized!')

        # at this point, we take all the params in the self.params structure:
        params = self.params

        # Try to avoid some stupid issues:
        params[params == scipy.inf] = 1e-100 * sys.float_info.max
        params[params == -scipy.inf] = -1e-100 * sys.float_info.max

        # Split up:
        D_vals = params[:self.nD]
        V_vals = params[self.nD:self.nD+self.nV]
        knots_D = params[self.nD+self.nV:self.nD+self.nV+self.nkD]
        knots_V = params[self.nD+self.nV+self.nkD:self.nD+self.nV+self.nkD +self.nkV]

        if self.equal_DV_knots:
            # force V knots to be equal to D knots
            knots_V = knots_D
            
        param_tshifts = params[
            self.nD+self.nV+self.nkD +self.nkV:self.nD+self.nV+self.nkD +self.nkV +self.nDiag
        ]
        d_weights = params[
            self.nD+self.nV+self.nkD +self.nkV +self.nDiag:
            self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW
        ]
        mix_rad_corr = params[
            self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW:
            self.nD+self.nV+self.nkD +self.nkV +self.nDiag+self.nW + 1 
        ]
        D_gauss = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+1:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+4
        ]
        V_gauss = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+4:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+7
        ]
        sawtooth_width = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+7:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+8
        ]
        rcl = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+8:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+9
        ]
        tau_divsol = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+9:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+10
        ]
        n0x = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+10:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+11
        ]
        zeta = params[
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+11:
            self.nD+self.nV+self.nkD+self.nkV+self.nDiag+self.nW+12
        ]
        
        if self.couple_D_V_gauss_params:
            # set width and location of V gaussian pedestal feature to be the same as for D
            V_gauss[1] = D_gauss[1]
            V_gauss[2] = D_gauss[2]
            
        return D_vals, V_vals, knots_D, knots_V, param_tshifts,\
            d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
            rcl, tau_divsol, n0x, zeta
        
        


    def get_spline_knots_prior(self, q='D'):
        '''Get scipy.stats priors for D or V spline knots
        '''
        nk = self.nkD if q=='D' else self.nkV
        free_knots  = self.set_free_D_knots if q=='D' else self.set_free_V_knots

        if free_knots:
            if self.force_identifiability:
                # force identifiability of uniform knots priors by re-mapping uniform samples to a hyper-triangle
                # The following prior is actually ignored in :py:func:`multinest_prior`
                prior = [stats.uniform(self.innermost_knot, self.outermost_knot-self.innermost_knot), ]*nk

            else:
                # sampling within Gaussian distributions. 

                if self.fixed_knots is not None:
                    # create knots around locations that have been provided
                    fixedKnots = self.fixed_knots[0:len(self.fixed_knots)//2] if q=='D' else self.fixed_knots[len(self.fixed_knots)//2:]
                    fixedSpacing = np.diff(fixedKnots)

                    for i,knot in enumerate(fixedKnots[1:-1]):
                        if i==0:
                            # more freedom to the left, towards the innermost (r/a=0) knot
                            prior = [stats.uniform(knot - fixedSpacing[i]*2.0/3.0, fixedSpacing[i+1]/3.0 + fixedSpacing[i]*2.0/3.0), ] 

                        elif i==len(fixedKnots[1:-2]):
                            # more freedom to the right, towards outermost (fixed) knot
                            prior += [stats.uniform(knot - fixedSpacing[i]/3.0, fixedSpacing[i+1]*2.0/3.0 + fixedSpacing[i]/3.0),]

                        else:
                            # standard
                            prior += [stats.uniform(knot - fixedSpacing[i]/3.0, fixedSpacing[i+1]/3.0 + fixedSpacing[i]/3.0) , ]

                else:

                    # create knots within uniform distributions around equally spaced locations
                    central_knots = np.linspace(self.innermost_knot, self.outermost_knot, nk+2)[1:-1]
                    spacing = np.diff(np.linspace(self.innermost_knot, self.outermost_knot, nk+2))[0]

                    for i,knot in enumerate(central_knots):
                        if i==0:
                            prior = [stats.uniform(knot - spacing/3.0, 2*spacing/3.0),]
                        else:
                            prior += [stats.uniform(knot - spacing/3.0,  2.*spacing/3.0), ]

        # Alternative: fix knots at start of inference
        else: 
            knots_grid = np.linspace(self.innermost_knot, self.outermost_knot, nk+2)[1:-1]

            # set D,V radial knots as equally spaced in r/a. 
            for i,knot in enumerate(knots_grid):
                if i==0:
                    prior = [stats.uniform(knot, 1e-10),]
                else:
                    prior += [stats.uniform(knot, 1e-10),]

        return prior



    #######
    def get_prior(self):                      
        """Returns a numpy array containing frozen scipy distributions for each prior dimension.
        """
        try:
            knotgrid_D = copy.deepcopy(self.knotgrid_D)
            knotgrid_V = copy.deepcopy(self.knotgrid_V) 
        except:
            # The first time that get_prior() is called, knotgrid_D and knotgrid_V are not available
            # Get rough, temporary values of num_edge_D/V_coeffs based on equally-spaced knots
            knotgrid_D = np.linspace(0.0, self.outermost_knot, self.nkD+2)   # includes knot on axis and end of grid
            knotgrid_V = np.linspace(0.0, self.outermost_knot, self.nkV+2)   # includes knot on axis and end of grid

        # separate into 3 regions: near-axis, core, pedestal
        # Exclude knot at 0 (which is fixed by boundary conditions for both D and V):
        if self.fix_D_to_tglf_neo:  # no free D spline values
            self.num_axis_D_coeffs = self.num_mid_D_coeffs = self.num_edge_D_coeffs = 0
        else:
            self.num_axis_D_coeffs = len(knotgrid_D[1:][knotgrid_D[1:] <= self.nearaxis_roa])
            self.num_mid_D_coeffs = len(knotgrid_D[1:][(knotgrid_D[1:] > self.nearaxis_roa)*(knotgrid_D[1:] < self.ped_roa_DV)])
            self.num_edge_D_coeffs = len(knotgrid_D[1:][knotgrid_D[1:] >= self.ped_roa_DV])
            
        if self.fix_V_to_tglf_neo:  # no free V spline values
            self.num_axis_V_coeffs = self.num_mid_V_coeffs = self.num_edge_V_coeffs = 0
        else:
            self.num_axis_V_coeffs = len(knotgrid_V[1:][knotgrid_V[1:] <= self.nearaxis_roa])
            self.num_mid_V_coeffs = len(knotgrid_V[1:][(knotgrid_V[1:] > self.nearaxis_roa)*(knotgrid_V[1:] < self.ped_roa_DV)])
            self.num_edge_V_coeffs = len(knotgrid_V[1:][knotgrid_V[1:] >= self.ped_roa_DV])

        ######

        # Begin by getting priors for D and V spline coefficients and knots
        # assume self.nD>0 to start creation of prior
        if self.D_prior_dist == 'loguniform':
            prior = [stats.loguniform(self.D_lb, self.D_axis_ub), ] * self.num_axis_D_coeffs
            prior += [stats.loguniform(self.D_lb, self.D_mid_ub), ] * self.num_mid_D_coeffs
            prior += [stats.loguniform(self.D_lb, self.D_edge_ub), ] * self.num_edge_D_coeffs
        elif self.D_prior_dist == 'uniform':
            prior = [stats.uniform(self.D_lb, self.D_axis_ub - self.D_lb), ] * self.num_axis_D_coeffs
            prior += [stats.uniform(self.D_lb, self.D_mid_ub - self.D_lb), ] * self.num_mid_D_coeffs
            prior += [stats.uniform(self.D_lb, self.D_edge_ub - self.D_lb), ] * self.num_edge_D_coeffs
        elif self.D_prior_dist == 'truncnorm':
            # set std for D gaussian sampling to be 1/3 of upper bound
            a = (self.D_lb - 0.0)/self.D_axis_std
            b = (self.D_axis_ub - 0.0)/self.D_axis_std
            prior = [stats.truncnorm(a,b, loc=0.0, scale = self.D_axis_std),] * self.num_axis_D_coeffs
            a = (self.D_lb - 0.0)/self.D_mid_std
            b = (self.D_mid_ub - 0.0)/self.D_mid_std
            prior += [stats.truncnorm(a,b, loc=0.0, scale = self.D_mid_std),] * self.num_mid_D_coeffs
            a = (self.D_lb - 0.0)/self.D_edge_std
            b = (self.D_edge_ub - 0.0)/self.D_edge_std
            prior += [stats.truncnorm(a,b, loc=0.0, scale = self.D_edge_std),] * self.num_edge_D_coeffs
        else:
            raise ValueError('Unrecognized D_prior_dist type!')

        if self.nV>0:

            _V_axis_mean = self.VoD_axis_mean if self.learn_VoD else self.V_axis_mean
            _V_axis_std = self.VoD_axis_std if self.learn_VoD else self.V_axis_std
            _V_axis_lb = self.VoD_axis_lb if self.learn_VoD else self.V_axis_lb
            _V_axis_ub = self.VoD_axis_ub if self.learn_VoD else self.V_axis_ub

            _V_mid_mean = self.VoD_mid_mean if self.learn_VoD else self.V_mid_mean
            _V_mid_std = self.VoD_mid_std if self.learn_VoD else self.V_mid_std
            _V_mid_lb = self.VoD_mid_lb if self.learn_VoD else self.V_mid_lb
            _V_mid_ub = self.VoD_mid_ub if self.learn_VoD else self.V_mid_ub

            _V_edge_mean = self.VoD_edge_mean if self.learn_VoD else self.V_edge_mean
            _V_edge_std = self.VoD_edge_std if self.learn_VoD else self.V_edge_std
            _V_edge_lb = self.VoD_edge_lb if self.learn_VoD else self.V_edge_lb
            _V_edge_ub = self.VoD_edge_ub if self.learn_VoD else self.V_edge_ub

            # near-axis
            a = (_V_axis_lb - _V_axis_mean)/_V_axis_std
            b = (_V_axis_ub - _V_axis_mean)/_V_axis_std
            prior += [stats.truncnorm(a, b, loc=_V_axis_mean, scale=_V_axis_std), ] * self.num_axis_V_coeffs

            # midradius
            a = (_V_mid_lb - _V_mid_mean)/_V_mid_std
            b = (_V_mid_ub - _V_mid_mean)/_V_mid_std
            prior += [stats.truncnorm(a, b, loc=_V_mid_mean, scale=_V_mid_std) ,] *self.num_mid_V_coeffs

            # edge - use midradius priors if gaussian feature is being used
            mean =_V_mid_mean if self.use_gaussian_Vped else _V_edge_mean
            std = _V_mid_std if self.use_gaussian_Vped else _V_edge_std
            lb = _V_mid_lb if self.use_gaussian_Vped else _V_edge_lb
            ub = _V_mid_ub if self.use_gaussian_Vped else _V_edge_ub

            a = (lb - mean)/std
            b = (ub - mean)/std
            prior += [stats.truncnorm(a, b, loc=mean, scale=std),] * self.num_edge_V_coeffs


        # if self.force_identifiability==True, knots samples here are ignored (see :py:meth:`multinest_prior`)
        if self.nkD>0:
             prior += self.get_spline_knots_prior(q='D')

        if self.nkV>0:
            prior += self.get_spline_knots_prior(q='V')

        #######
        # Time base shifts:
        prior +=  self.shift_prior

        # Prior on diagnostic weights for ALL active diagnostics
        #prior += [stats.gamma(1.0, loc=0.0, scale=1.), ] * self.nW # exponential, Hobson MNRAS 2002
        prior +=  [stats.gamma(self.CDI_a, loc=0.0, scale=1./self.CDI_a), ] * self.nW

        # prior on mixing radius correction (units of cm)
        prior +=  [self.mixing_radius_prior,]

        # prior on Gaussian D profile amplitude, width and radial location parameters
        prior +=  [self.gaussian_D_amp_prior,]
        prior +=  [self.gaussian_D_w_prior,]
        prior +=  [self.gaussian_D_r_prior,]

        # prior on Gaussian V profile amplitude, width and radial location parameters
        prior +=  [self.gaussian_V_amp_prior,]
        prior +=  [self.gaussian_V_w_prior,]
        prior +=  [self.gaussian_V_r_prior,]

        # prior on width of sawtooth crash
        prior +=  [self.dsaw_prior,]

        # recycling
        prior +=  [self.rcl_prior,]
        prior +=  [self.tau_divsol_prior,]

        # neutral density multiplier
        prior +=  [self.n0x_prior,]

        # zeta parameter for pedestal Z scaling
        prior +=  [self.zeta_prior,]

        return np.array(prior)

        # ==============================
        

    def prior_random_draw(self, prior=None):
        '''Draw random sample from a set of priors, expected to be given as argument in the form
        of a 1D list or array.

        If prior=None, draws prior from the entire list of priors given by :py:meth:`get_prior`.
        '''       
        if prior is None:
            prior = self.get_prior()

        uu = np.random.uniform(size=len(prior))

        return np.array([prior[ii].ppf(uu[ii]) for ii in np.arange(len(uu))])




    def stretch_profs(self, r_vec, time_vec, ne_cm3, Te_keV, Te_eV_LCFS=75.0):
        '''
        Stretch in x direction to match chosen temperature (in eV) at LCFS.
        Note that ne and Te msut be on the same radial and time bases!

        Note that Te 
        '''
        TeShifted = copy.deepcopy(Te_keV); neShifted = copy.deepcopy(ne_cm3); 

        # ensure a temperature of 75 eV at each time slice
        for ti,tt in enumerate(time_vec):
            x_of_TeSep = interp1d(TeShifted[ti,:], r_vec, bounds_error=False)(Te_eV_LCFS*1e-3)
            xShifted = r_vec/x_of_TeSep
            TeShifted[ti,:] = interp1d(xShifted, TeShifted[ti,:], bounds_error=False)(r_vec)
            neShifted[ti,:] = interp1d(xShifted, neShifted[ti,:], bounds_error=False)(r_vec)

            # without extrapolation, some values at the edge may be set to nan. Set them to boundary value:
            whnan = np.isnan(TeShifted[ti,:])
            if np.sum(whnan): 
                tmp = TeShifted[ti,~whnan]
                TeShifted[ti, whnan]= tmp[-1]

            whnan = np.isnan(neShifted[ti,:])
            if np.sum(whnan): 
                tmp = neShifted[ti,~whnan]
                neShifted[ti,whnan] = tmp[-1]
        
        return neShifted, TeShifted


    def load_ne_te_ti(self):
        """ 
        Load time-dept ne and Te profiles, and time-independent Ti.

        ne is cached in units of cm^-3 and T is in units of eV
        """
        with open(f'./{self.shot}/2D_fits_{self.shot}.pkl','rb') as f:
            ne_profs,Te_profs = pkl.load(f)

        # Interpolate profiles on common time grid to be able to stretch for Te=75eV condition
        t_vec = np.linspace(np.min([ne_profs['t'].min(),Te_profs['t'].min()]),
                            np.max([ne_profs['t'].max(),Te_profs['t'].max()]),
                            np.max([len(ne_profs['t']),len(Te_profs['t'])])
        )

        # keep time of ne and Te formally separate for the moment, but may join them later on. 
        self.time_ne = copy.deepcopy(t_vec)
        self.time_Te = copy.deepcopy(t_vec)

        _Te_in = interp2d(Te_profs['x'],Te_profs['t'],Te_profs['Te'])(self.roa_grid_in, self.time_Te)
        _ne_in = interp2d(ne_profs['x'],ne_profs['t'],ne_profs['ne'])(self.roa_grid_in, self.time_ne) 

        # simple up/down shift of Te to make sure that Te goes through 75 eV at the LCFS
        #for ti,tt in enumerate(self.time_Te):
        #    Te_at_LCFS = interp1d(self.roa_grid, self.Te_in[ti,:])(1.0)
        #    self.Te_in[ti,:] += (0.075 - Te_at_LCFS)

        # stretch pofiles to have Te=75 eV at LCFS
        ne_in, Te_in = self.stretch_profs(self.roa_grid_in, t_vec, _ne_in, _Te_in)

        #only use mean profile outside of LCFS (too much noise)
        ind_lcfs = np.argmin(np.abs(self.roa_grid_in - 1.0))
        Te_in[:,ind_lcfs:] = np.mean(Te_in[:,ind_lcfs:], axis=0)
        Te_in[Te_in<0.01] = 0.01  # set minimum of 10 eV
        print("Successfully loaded time-dependent kinetic profiles!")

        # allow for scaling of the entire Te profile to test atomic data sensitivity
        Te_in *= self.Te_scale        

        # finally, ensure that ne and Te don't have any negative values:
        ne_in[ne_in < 0.0] = 0.001
        Te_in[Te_in < 0.0] = 0.001
            
        # both of the methods load ne in 10^20 m^-3 and Te in keV
        self.ne_cm3_in = ne_in*1e14  # 10^20 m^-3 --> cm^-3
        self.Te_eV_in = Te_in*1e3  # keV --> eV

        # now load time-independent Ti
        ip = omfit_gapy.OMFITinputprofiles(f'{self.shot}/input.profiles.{self.shot}')
        rhop_ip = np.sqrt((ip['polflux'])/ip['polflux'][-1])
      
        self.rhop_in = aurora.rad_coord_transform(self.roa_grid_in, 'r/a', 'rhop', self.geqdsk)
        # extrapolate Ti in SOL as a fixed value
        self.Ti_eV_in = interp1d(rhop_ip, ip['Ti_1']*1e3, bounds_error=False, fill_value=20)(self.rhop_in)

        # omega points near axis have some issue, ignore and extrapolate
        self.omega_s_in = interp1d(rhop_ip[2:], ip['omega0'][2:], bounds_error=False, fill_value='extrapolate')(self.rhop_in)
        self.vi_ms_in = self.omega_s_in * self.geqdsk['fluxSurfaces']['R0']

        #Zeff =  interp1d(rhop_ip[1:-1], ip['z_eff'][1:-1], bounds_error=False, fill_value='extrapolate')(rhop_grid)

    



    def load_solps_n0(self, debug_plots=False):
        '''Load FSA neutral profiles from the appropriate SOLPS run.
        '''

        if self.shot==1101014006:
            # L-mode
            solps_shot = 1100308004; solps_run='Attempt14'
            path = f'/home/sciortino/SOLPS/Lmode_1100308004/'
        elif self.shot==1101014019:
            # H-mode
            solps_shot = 1100305023; solps_run='Attempt24'  # H-mode  (old: 23)
            path = '/home/sciortino/SOLPS/Hmode_1100305023'
        elif self.shot==1101014030:
            # I-mode
            solps_shot = 1080416025; solps_run='Attempt15N'
            path = '/home/sciortino/SOLPS/Imode_1080416025'
        else:
            raise ValueError(f'SOLPS-ITER/EIRENE neutral data not available for shot {self.shot}')

        solps_geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(device='CMOD',shot=solps_shot,
                                                                time=self._t0*1000,  #ms
                                                                SNAPfile='EFIT20',   
                                                                fail_if_out_of_range=False,time_diff_warning_threshold=20)
        
        # load SOLPS results
        #so = aurora.solps_case(path, solps_geqdsk, case_num=case_num, form='extracted')
        so = aurora.solps_case(path, solps_geqdsk, solps_run=solps_run, form='full')

        # fetch and process SOLPS output 
        #so.process_solps_data(plot=debug_plots)
        
        # compare SOLPS results at midplane with FSA
        #rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_n0_profiles()
        rhop_fsa, neut_fsa, rhop_LFS, neut_LFS, rhop_HFS, neut_HFS = so.get_radial_prof(so.quants['nn'])

        # use FSA neutrals inside LCFS, extrapolate exponentially outside
        self.roa_n0 = np.linspace(np.min(self.roa_grid_in),np.max(self.roa_grid_in),1000)
        self.rhop_n0 =aurora.rad_coord_transform(self.roa_n0,'r/a','rhop', solps_geqdsk)
        n0_m3 = np.exp(interp1d(rhop_fsa[~np.isnan(neut_fsa)], np.log(neut_fsa[~np.isnan(neut_fsa)]), 
                                  bounds_error=False, fill_value='extrapolate')(self.rhop_n0))

        # don't let extrapolation of EIRENE neutrals give significant neutral densities in the very core
        #mask = self.rhop_n0<np.nanmin(rhop_fsa[~np.isnan(neut_fsa)])
        
        # don't allow densities in the outer pedestal or SOL determine the core decay length
        mask = (self.rhop_n0<np.nanmin(rhop_fsa[~np.isnan(neut_fsa)])) | (self.rhop_n0>0.95)

        # use exponential fit into the core:
        a,b = np.polyfit(self.rhop_n0[~mask], np.log(n0_m3[~mask]), 1)
        n0_m3[mask] = np.exp(a*self.rhop_n0[mask] + b)  
        
        self.n0_cm3 = n0_m3* 1e-6 # m^-3 --> cm^-3


        # get ne and Te on rhop_n0 grid, extrapolating in the SOL by the last value of ne (use mean in time)
        _ne_cm3 = interp1d(self.rhop_in, np.mean(self.ne_cm3_in,0), bounds_error=False, fill_value=np.mean(self.ne_cm3_in,0)[-1])(self.rhop_n0)
        _Te_eV = interp1d(self.rhop_in, np.mean(self.Te_eV_in,0), bounds_error=False, fill_value=np.mean(self.Te_eV_in,0)[-1])(self.rhop_n0)

        # get minimum fractional abundance based on H ionization equilibrium
        atom_data = aurora.get_atom_data('H',['scd','acd','ccd'])
        logTe, fz = aurora.get_frac_abundances(atom_data, _ne_cm3, _Te_eV, 
                                                      #include_cx=False,  # no H-to-H CX, symmetric process
                                                      plot=False)

        # prevent extrapolation to very small values by setting a minimum based on ionization fractions of H
        #self.n0_cm3[self.n0_cm3<1] = 1.0
        mask2 = self.n0_cm3/_ne_cm3<fz[:,0]
        self.n0_cm3[mask2] = fz[mask2,0]*_ne_cm3[mask2]
        self.n0_by_ne = self.n0_cm3/_ne_cm3

        if debug_plots:
            # plot log values (rather than on log scale) to correctly visualize uncertainties
            fig,ax = plt.subplots()
            ax.plot(rhop_fsa, np.log10(neut_fsa)*1e-6, c='r', label='Aurora-SOLPS output') 
            ax.plot(self.rhop_n0, np.log10(self.n0_cm3), c='k', label='BITS interpolation')
            ax.set_xlabel(r'$\rho_p$')
            ax.set_ylabel(r'$log_{10}(n_0 [cm^{-3}])$')


    def load_lya_n0(self, debug_plots=False):
        ''' Load edge neutral profiles, either derived from spectroscopic measurements (e.g. Lyman-alpha) 
        or from neutral simulations (e.g. KN1D). 
        '''
        if self.neutrals_file is None:
            print('No neutrals file was provided!')
            return

        with open(self.neutrals_file,'rb') as f:
            out = pkl.load(f)

        rhop,roa,R, N1_prof,N1_prof_unc,ne_prof,ne_prof_unc,Te_prof,Te_prof_unc  = out

        # interpolate profiles on aurora radial grid and expolate on exp base
        roa_r = roa[N1_prof>1e8]   # anything smaller is likely Ly-alpha noise
        N1_prof_unc_r = N1_prof_unc[N1_prof>1e8]
        N1_prof_r = N1_prof[N1_prof>1e8]

        # cap relative uncertainties to reasonable values
        rel_unc = N1_prof_unc_r/N1_prof_r
        rel_unc[rel_unc>2.] = 2.
        N1_prof_unc_r = rel_unc*N1_prof_r

        # use Pchip interpolator to obtain some smoothness on a fine grid
        self.roa_n0 = np.linspace(np.min(self.roa_grid_in),np.max(self.roa_grid_in),1000)
        self.rhop_n0 = aurora.rad_coord_transform(self.roa_n0, 'r/a','rhop', self.geqdsk)
        self.n0_cm3 = np.exp(interp1d(roa_r, np.log(N1_prof_r), bounds_error=False, fill_value='extrapolate')(self.roa_n0))
        self.n0_cm3_unc = interp1d(roa_r, N1_prof_unc_r,  bounds_error=False, fill_value='extrapolate')(self.roa_n0)
        
        # prevent extrapolation to very small values by setting a minimum [cm^-3]
        self.n0_cm3[self.n0_cm3<1] = 1.0

        if debug_plots:
            # plot log values (rather than on log scale) to correctly visualize uncertainties
            fig,ax = plt.subplots()
            ax.plot(roa_r, np.log10(N1_prof_r))
            ax.errorbar(roa_r, np.log10(N1_prof_r), rel_unc) 
            ax.plot(self.roa_n0, np.log10(self.n0_cm3), 'k')
            ax.set_xlabel('r/a')
            ax.set_ylabel(r'$log_{10}(n_0 [cm^{-3}])$')




    # ----------------------------------------#
    def DV2cs_den(self, params=None, unstage=False,
        explicit_D=None, explicit_D_roa=None, explicit_V=None, explicit_V_roa=None,
        debug_plots=False, return_reservoirs=False, return_res_ds=False):
        """Calls Aurora with the given parameters and returns the charge state densities. 
        
        D,V profiles are evaluated from cached inference parameters through eval_DV(), which
        returns (D,V) even if (D,V/D) are being inferred (i.e. if learn_VoD=True).
        
        Parameters
        ----------
        params : array of float, optional
            The parameters to use when evaluating the model. See split_params() for a
            breakdown. If absent, :py:attr:`self.params` is used.
        unstage : bool, optional
            If True, any superstages are unstaged at the end of an AURORA run. This can be 
            kept as False (default) during inferences, since indexing has been set up to look at 
            charge states from the top down. However, it may be useful to unstage to look at all
            the impurity charge state densities.
        explicit_D : array of float, optional
            Explicit values of D to use. Overrides the profile which would have
            been obtained from the parameters in `params` (but the scalings/etc.
            from `params` are still used). Assumed function of radius (given on explicit_D_roa).
        explicit_D_roa : array of float, optional
            Grid of r/a which `explicit_D` is given on.
        explicit_V : array of float, optional
            Explicit values of V to use. Overrides the profile which would have
            been obtained from the parameters in `params` (but the scalings/etc.
            from `params` are still used). Assumed function of radius (given on explicit_D_roa).
        explicit_V_roa : array of float, optional
            Grid of r/a which `explicit_V` is given on.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        return_reservoirs : bool, optional
            If True, return aurora output for particle densities in each simulation reservoir. 
        return_res_ds : bool, optional
            If True, return xarray dataset with aurora results for postprocessing (e.g. radiation modeling).
        
        Returns
        ------------
        cs_den : array (`n_time`, `n_cs`, `n_space`) 
            Charge state densities returned by Aurora.
        """
        if params is not None:
            # setting self.params to the parameters passed externally (una tantum or via sampler)
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params
        else:
            params = self.params

        D_vals, V_vals, knots_D, knots_V, param_tshifts, d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
            rcl, tau_divsol, n0x, zeta = self.split_params()
        
        # Use aurora simulation setup for ion of interest. No need to deepcopy here.
        asim = self.asim 

        if asim.namelist['cxr_flag'] and self.free_n0x:
            # reproduce here key steps made in asim.get_time_dept_atomic_rates() to set recombination rates with CXR
            # use cached RR+DR and CXR rates at given ne,Te,Ti
            #R = self.R_rates + self.n0x * asim._n0[:,None] * self.alpha_CX_rates[:,:self.R_rates.shape[1],:]

            # S and R for the Z+1 stage must be zero for the forward model.
            #R_rates = np.zeros((R.shape[2], R.shape[1] + 1, asim.time_grid.size), order='F')
            #R_rates[:, :-1] = R.T
            #asim.R_rates = R_rates

            # Get an effective recombination rate by summing radiative & CX recombination rates with n0x scale factor
            asim_R = self.asim_R + self.n0x * self.asim_cx*(asim._n0/asim._ne)[:,None] 
            asim_S = copy.deepcopy(self.asim_S)

            if len(asim.superstages):
                _, asim_R, asim_S, asim.fz_upstage = aurora.superstage_rates(
                    asim_R, self.asim_S, asim.superstages, save_time = asim.save_time)

            # S and R for the Z+1 stage must be zero for the forward model.            
            # the following initialization is only needed for debugging/tests when turning on/off superstaging
            #asim.R_rates = np.zeros((asim_R.shape[2], asim_R.shape[1] + 1, asim.time_grid.size), order='F')
            #asim.S_rates = np.zeros((asim_S.shape[2], asim_S.shape[1] + 1, asim.time_grid.size), order='F')

            asim.R_rates[:, :-1] = asim_R.T
            asim.S_rates[:, :-1] = asim_S.T


        # update aurora sim setup with new inputs
        if self.free_sawtooth_width:
            asim.sawtooth_erfc_width = copy.deepcopy(self.sawtooth_width)

        if self.free_rcl:
            asim.wall_recycling = rcl #self.rcl_val   # always sampled as >0
            # -ve values deactivate all recycling and divertor return
            
        if self.free_tau_divsol:
            asim.tau_div_SOL_ms = self.tau_divsol    # ms

        if self.free_mixing_radius:
            asim.mixing_radius = self.rsaw_vol_cm + mix_rad_corr

        if (explicit_D is None) or (explicit_V is None):
            # get time- and charge-state-dependent D,V
            D_z, V_z, times_DV = self.eval_DV(imp='Ca')
        else:
            # Handle explicit D_z and V_z
            if explicit_D.ndim==3:
                # assume that D, V and their grids were already given on the right time,space and nZ dependent form
                D_z = copy.deepcopy(explicit_D)
                V_z = copy.deepcopy(explicit_V)

                # get D & V time changes (normally, just a single value)
                _, _, times_DV = self.eval_DV(imp='Ca')
            else:
                # convert input r/a grids for explicit D and V into rhop grids
                explicit_D_rhop = self.efit_tree.rho2rho('r/a', 'sqrtpsinorm', explicit_D_roa, (self.time_1 + self.time_2) / 2.0)
                explicit_D_rhop[0] = 0.0
                explicit_V_rhop = self.efit_tree.rho2rho('r/a', 'sqrtpsinorm', explicit_V_roa, (self.time_1 + self.time_2) / 2.0)
                explicit_V_rhop[0] = 0.0

                D_z = interp1d(explicit_D_rhop, explicit_D, bounds_error=False,\
                               fill_value=(explicit_D[0],explicit_D[-1]))(asim.rhop_grid)  # 1D, handled internally by Aurora
                D_z[D_z<0.01] = 0.01 # avoid issues with extrapolation
                V_z = interp1d(explicit_V_rhop, explicit_V, bounds_error=False,\
                               fill_value=(explicit_V[0],explicit_V[-1]))(asim.rhop_grid) 
                times_DV = None  # Aurora will assume time independence

        # effective diffusion to wall at last grid point
        if self.decay_length_boundary<0:
            # compute dlen as sqrt(D/tau_||) at the last grid point, as in STRAHL, for every iteration
            asim.decay_length_boundary = round(np.sqrt(1e4*D[-1]/asim.par_loss_rate[-1]),5)

        # ------------------------------------------------------------------------------------------------------------
        # run forward model:
        #from bits_decorators import debug
        # @debug    
        out = asim.run_aurora(
            D_z*1e4, V_z*1e2,    #(space, time,nZ) in CGS units
            times_DV,
            alg_opt=1, # Linder algorithm
            evolneut=False, #True, # adds some runtime and doesn't matter in our sims
            plot=False,
            unstage = unstage, # apply unstaging only if requested, not necessary during inference
        )
        # ------------------------------------------------------------------------------------------------------------

        if return_reservoirs:
            # useful to check particle conservation:
            return out

        # unwrap output from different aurora reservoirs
        cs_den, N_wall, N_div, N_pump, N_ret, N_tsu, N_dsu, N_dsul, rcld_rate, rclw_rate = out
        cs_den = cs_den.transpose(2,1,0)


        if return_res_ds:
            # return results dataset for aurora postprocessing (e.g. radiation modeling)

            # source function on output time grid
            source_time_history_out = interp1d(asim.time_grid, asim.source_time_history)(asim.time_out)
            
            # collect aurora result into a clear dataset
            res_ds = xarray.Dataset({'impurity_density': ([ 'time', 'charge_states','rvol_grid'], cs_den),
                                     'total_impurity_density': ([ 'time','rvol_grid'], np.nansum(cs_den, axis=1)),
                                     'source_time_history': (['time'], source_time_history_out), 
                                     'particles_in_divertor': (['time'], N_div), 
                                     'particles_in_pump': (['time'], N_pump), 
                                     'parallel_loss': (['time'], N_dsu), 
                                     'parallel_loss_to_limiter': (['time'], N_dsul), 
                                     'edge_loss': (['time'], N_tsu), 
                                     'particles_at_wall': (['time'], N_wall), 
                                     'particles_retained_at_wall': (['time'], N_ret), 
                                     'recycling_from_wall':  (['time'], rclw_rate), 
                                     'recycling_from_divertor':  (['time'], rcld_rate), 
                                     'pro': (['rvol_grid'], asim.pro_grid), 
                                     'rhop_grid': (['rvol_grid'], self.rhop)
                                 },
                                    coords={'time': asim.time_out, 
                                            'rvol_grid': asim.rvol_grid,
                                            'charge_states': np.arange(cs_den.shape[1])
                                        })
            
            return res_ds


        if self.check_particle_conservation:

            # obtain number of particles in each reservoir:
            pres = asim.check_conservation(plot=False)

            # time index of maximum of particles in plasma
            indmax = np.argmax(pres['plasma_particles'])

            # find penetration fraction when the plasma particles reach a max, likely before any LBO clusters
            # This fraction is slightly higher than if we took np.max(plasma_particles)/np.max(integ_source)
            self.particle_penetration_ratio = pres['plasma_particles'][indmax]/pres['integ_source'][indmax]

            # next, check particle conservation at the end of the simulation:
            self.particle_loss_ratio = abs((pres['total'][-1]-pres['integ_source'][-1])/pres['integ_source'][-1])
        else:

            # effectively turn off check on particle penetration:
            self.particle_penetration_ratio = 1.0
            
            # assume particles are conserved:
            self.particle_loss_ratio = 0.0


        ##################
        if debug_plots:
            
            if self.check_particle_conservation:
                pres,axs = asim.check_conservation(plot=True)

                # update particle_loss_ratio
                self.particle_loss_ratio = abs((pres['total'][-1]-pres['integ_source'][-1])/pres['integ_source'][-1])
                print('Particle penetration ratio: ', self.particle_penetration_ratio)
                print('Fraction of particles lost at the end of the simulation: ', self.particle_loss_ratio)

            from mpl_toolkits.mplot3d import Axes3D
            X,Y = np.meshgrid(self.time,self.rhop)
            figg1 =plt.figure()
            axx1=figg1.add_subplot(111,projection='3d')
            axx1.plot_surface(X,Y,cs_den[:,-3,:].T, rstride=10, cstride=10)
            axx1.set_title('He-like Ca')

            figg2 =plt.figure()
            axx2=figg2.add_subplot(111,projection='3d')
            axx2.plot_surface(X,Y,cs_den[:,-4,:].T, rstride=10, cstride=10)
            axx2.set_title('Li-like Ca')

            # Plot the ne, Te profiles:
            f = plt.figure()
            a_ne=f.add_subplot(1,2,1, projection='3d')
            a_Te=f.add_subplot(1,2,2, projection='3d')

            a_ne.locator_params(axis='z', nbins=3)
            a_ne.locator_params(axis='y', nbins=4)
            a_ne.locator_params(axis='x', nbins=6)
            a_Te.locator_params(axis='z', nbins=3)
            a_Te.locator_params(axis='y', nbins=4)
            a_Te.locator_params(axis='x', nbins=6)

            X,T = np.meshgrid(self.rhop,self.time)
            a_ne.plot_surface(X,T,self.ne_cm3, rstride=10, cstride=10) #cmap='brg')
            a_Te.plot_surface(X,T,self.Te_eV, rstride=10, cstride=10) #cmap='hsv')
            a_ne.set_zlabel(r'$n_e$ [$cm^{-3}]$', labelpad=10)
            a_ne.set_xlabel(r'$\rho_p$', labelpad=20); a_ne.set_ylabel(r'$t$ [s]', labelpad=20)
            a_Te.set_zlabel(r'$T_e$ [$eV$]', labelpad=10)
            a_Te.set_xlabel(r'$\rho_p$', labelpad=20); a_Te.set_ylabel(r'$t$ [s]', labelpad=20)

            plt.tight_layout()
                
            # Plot the charge state densities:
            aurora.slider_plot(
                self.roa_grid, #self.rhop,
                self.time,
                scipy.rollaxis(cs_den.T, 1),
                xlabel=r'$r/a$', #r'$\rho_p$',
                ylabel=r'$t$ [s]',
                zlabel=r'$n_z$ [cm$^{-3}$]',
                labels=[str(i) for i in np.arange(0, cs_den.shape[1])],
                plot_sum=True
            )
            
            # Plot the total impurity content:
            volnorm_grid = self.efit_tree.psinorm2volnorm(
                self.rhop**2.0,
                (self.time_1 + self.time_2) / 2.0
            )
            V = self.efit_tree.psinorm2v(1.0, (self.time_1 + self.time_2) / 2.0)
            mask = ~scipy.isnan(volnorm_grid)
            volnorm_grid = volnorm_grid[mask]
            nn = cs_den.sum(axis=1)[:, mask]
            # Use the trapezoidal rule:
            N = V * 0.5 * ((volnorm_grid[1:] - volnorm_grid[:-1]) * (nn[:, 1:] + nn[:, :-1])).sum(axis=1)
            
            f = plt.figure(123)
            a = f.add_subplot(1, 1, 1)
            a.plot(self.time, N, '.-',label='Total impurity content')
            a.set_xlabel('$t$ [s]')
            a.set_ylabel('$N$')
            #a.set_title("Total impurity content")
            
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
  
            cmap = copy.copy(mpl.cm.get_cmap("plasma"))
            #cmap = 'plasma'
            pcm = a.pcolormesh(self.roa_grid, self.time, cs_den.sum(axis=1), cmap=cmap, 
                               vmax=cs_den.sum(axis=1)[:, 0].max(), shading='auto')
            pcm.cmap.set_over('white')
            f.colorbar(pcm, extend='max')
            a.set_xlabel(r"$r/a$")
            a.set_ylabel(r"$t$ [s]")
            a.set_title("Tot impurity density")
            a.set_xlim([0.0,1.0])
            a.set_ylim([self.injections[0].t_inj,plt.gca().get_ylim()[1]])
        
        # cs_den must have shape (n_time, n_cs, n_space)
        return cs_den



    def cs_den2dlines(self, cs_den, params=None, debug_plots=False):
        """Predicts the local emissivities that would arise from the given charge state densities.
        
        Parameters
        ----------
        cs_den : array of float, (`n_time`, `n_cs`, `n_space`)
            The charge state densities as computed by aurora.
        params : array of float
            The parameters to use. If absent, :py:attr:`self.params` is used.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params
        

        if self.signal_mask[0]:

            dlines_xics = np.zeros((self.signals[0].y.shape[0], len(self.signals[0].t), len(self.rhop_spec)))

            n_H = cs_den[:,-2,:]
            n_He = cs_den[:,-3,:]
            n_Li = cs_den[:,-4,:]
            n_Be = cs_den[:,-5,:]

            if np.array_equal(self.rhop_spec, self.rhop) and np.array_equal(self.time_spec, self.time):
                # spectral components were pre-computed on rhop and time grids of Aurora
                n_H_s = n_H
                n_He_s = n_He
                n_Li_s = n_Li
                n_Be_s = n_Be
            elif np.array_equal(self.rhop_spec, self.rhop):
                # 1D interpolation of time grid
                n_H_s = interp1d(self.time, n_H, axis=0, copy=False, assume_sorted=True)(self.time_spec)
                n_He_s = interp1d(self.time, n_He, axis=0, copy=False, assume_sorted=True)(self.time_spec)
                n_Li_s = interp1d(self.time, n_Li, axis=0, copy=False, assume_sorted=True)(self.time_spec)
                n_Be_s = interp1d(self.time, n_Be, axis=0, copy=False, assume_sorted=True)(self.time_spec)
            elif np.array_equal(self.time_spec, self.time):
                # 1D interpolation of rhop grid
                n_H_s = interp1d(self.rhop, n_H, axis=1, copy=False, assume_sorted=True)(self.rhop_spec)
                n_He_s = interp1d(self.rhop, n_He, axis=1, copy=False, assume_sorted=True)(self.rhop_spec)
                n_Li_s = interp1d(self.rhop, n_Li, axis=1, copy=False, assume_sorted=True)(self.rhop_spec)
                n_Be_s = interp1d(self.rhop, n_Be, axis=1, copy=False, assume_sorted=True)(self.rhop_spec)
            else:
                # interpolate Aurora result of cs_den to the lower-resolution time and rhop grids on which PECs were computed
                n_H_s = RectBivariateSpline(self.time, self.rhop, n_H, kx=1, ky=1)(self.time_spec, self.rhop_spec)  #~5-6 ms
                n_He_s = RectBivariateSpline(self.time, self.rhop, n_He, kx=1, ky=1)(self.time_spec, self.rhop_spec)
                n_Li_s = RectBivariateSpline(self.time, self.rhop, n_Li, kx=1, ky=1)(self.time_spec, self.rhop_spec)
                n_Be_s = RectBivariateSpline(self.time, self.rhop, n_Be, kx=1, ky=1)(self.time_spec, self.rhop_spec)

            # fast multiplications of large arrays -- dimensions of wavelength, time, radius; ~5-6 ms as well
            dlines_xics = np.einsum('kl,ikl->ikl', n_Be_s, self.spec_Be_mult, optimize='optimal')+\
                    np.einsum('kl,ikl->ikl', n_Li_s, self.spec_Li_mult, optimize='optimal')+\
                    np.einsum('kl,ikl->ikl', n_He_s, self.spec_He_mult, optimize='optimal')+\
                    np.einsum('kl,ikl->ikl', n_H_s, self.spec_H_mult, optimize='optimal')

        else:
            dlines_xics = None


        if self.signal_mask[1]: #VUV
            # sum contributions from all charge states to each of the VUV lines (accounting for possible superstaging)
            dlines = np.einsum('ijkl,ijk->lik', self.dlines_vuv_comps[:,self.superstages,:,:], cs_den)
            
        
        if self.signal_mask[2]:  #SXR
            # Compute the emissivity seen through the core XTOMO filters:

            # SXR line radiation for each charge state
            sxr_line_rad = np.maximum(cs_den[:,:-1] * self.pls[:,self.superstages[:-1],:], 1e-60)

            # SXR continuum radiation for each charge state
            sxr_cont_rad = cs_den[:,1:] * self.prs[:,self.superstages[1:],:]

            # SXR total rad -- VUV lines are "unwrapped" in dlines
            dlines[self.num_vuv_lines, :, :] = sxr_line_rad.sum(1) + sxr_cont_rad.sum(1)

        if self.signal_mask[1]==False and self.signal_mask[2]==False:
            # only XICS is being modelled
            dlines = None



        if debug_plots:

            if dlines_xics is not None: # XICS spectra

                # slider to look at key line profiles over time
                four_lines = np.zeros((4, len(self.rhop_spec), len(self.time_spec)))

                w_ind = np.argmin(np.abs(self.lam_spec - self.lam_w_Ca))
                x_ind = np.argmin(np.abs(self.lam_spec - self.lam_x_Ca))
                y_ind = np.argmin(np.abs(self.lam_spec - self.lam_y_Ca))
                z_ind = np.argmin(np.abs(self.lam_spec - self.lam_z_Ca))

                four_lines[0,:,:] = np.max(dlines_xics[w_ind-3:w_ind+3],axis=0).T
                four_lines[1,:,:] = np.max(dlines_xics[x_ind-3:x_ind+3],axis=0).T
                four_lines[2,:,:] = np.max(dlines_xics[y_ind-3:y_ind+3],axis=0).T
                four_lines[3,:,:] = np.max(dlines_xics[z_ind-3:z_ind+3],axis=0).T

                aurora.slider_plot(
                    self.rhop_spec,
                    self.time_spec,
                    four_lines,
                    xlabel=r'$\rho_p$',
                    ylabel=r'$t$ [s]',
                    zlabel=r'$\epsilon_w$ [W/cm$^3$]',
                    labels=['w','x','y','z'] #str(i) for i in np.arange(0, dlines.shape[1])]
                )


            if dlines is not None: # VUV and SXR signals

                # normalize wrt diagnostic max over its time history (and over space)
                _dlines = dlines.transpose(0,2,1)
                _norm = np.nanmax(dlines, (1,2))

                # Plot the emissivity profiles:
                aurora.slider_plot(
                    self.roa_grid,
                    self.time,
                    _dlines/_norm[:,None,None],
                    xlabel=r'$r/a$', #r'$\sqrt{\psi_n}$',
                    ylabel=r'$t$ [s]',
                    zlabel=r'$\epsilon$ [W/cm$^3$]',
                    labels=[str(i) for i in np.arange(0, dlines.shape[1])]
                )


        
        return dlines, dlines_xics
    
     

    def dlines2sig(self, dlines, dlines_xics, params=None, debug_plots=False, sigsplines=None):
        """Computes the diagnostic signals corresponding to the given local emissivities.
        
        Takes each signal in :py:attr:`self.signals`, applies the weights (if
        present), interpolates onto the correct timebase and (if appropriate)
        normalizes the interpolated signal.
        
        Returns an array, `signals`, with one entry for each element in
        :py:attr:`self.signals`. Each entry has shape (`n_time`, `n_chan`).
        
        Parameters
        ----------
        dlines : array of float, (`n_lines`,`n_time`, `n_space`)
            The spatial profiles of local emissivities of VUV and SXR.
        dlines_xics : array of float, (`n_lam`,`n_lines`,`n_time`, `n_space`)
            The spatial profiles of local emissivities of XICS spectra.
        params : array of float
            The parameters to use. If absent, :py:attr:`self.params` is used.
        debug_plots : bool, optional
            If True, plots of the various steps will be generated. Default is
            False (do not produce plots).
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params
        
        D_vals, V_vals, knots_D, knots_V, param_tshifts, d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
            rcl, tau_divsol, n0x, zeta = self.split_params()        
        
        sig = []

        if self.signal_mask[0]:

            # find XICS signals by line integration
            sig_xics = np.zeros_like(self.signals[0].std_y)

            # do some time binning
            postinj = self.signals[0].t >= -  param_tshifts[0]
            time_bins = np.array([np.linspace(self.signals[0].t[postinj][t_bin] +  param_tshifts[0] - self.diag_time_res[0]/2.0 ,
                                              self.signals[0].t[postinj][t_bin] +  param_tshifts[0] + self.diag_time_res[0]/2.0, 
                                              10, endpoint=True) for t_bin in np.arange(len(self.signals[0].t[postinj]))])

            _sig_xics = np.zeros((len(self.lam_spec), len(self.signals[0].t), self.num_xics_channels))

            # line integrate and time-average within bins (list comps and einsum for speed)
            _sig_xics[:,postinj,:] = np.nanmean([
                interp1d(self.time_spec-self.time_1, 
                         np.einsum('ijk,lk->ijl', dlines_xics, self.signals[0].weights), 
                         # extrapolation is needed because time_bins can go beyond the time range of splines
                         axis=1, copy=False, assume_sorted=True, bounds_error=False, fill_value='extrapolate')(x) 
                    for x in time_bins], axis=2).transpose(1,0,2)

            # Now interpolate line integrated spectra to XICS wavelength axes for each chord, considering Doppler shifts
            sig_xics = np.array([
                interp1d(self.lam_spec + self.dlam_doppler[ch], _sig_xics[:,:,ch], 
                         axis=0, copy=False, assume_sorted=True, bounds_error=False, fill_value=0.0)(
                             self.signals[0].lams[:,ch])
                for ch in np.arange(self.num_xics_channels)
                ]).transpose(1,2,0)

            if self.normalize:
                # normalization seems necessary for LS rescaling to work?
                sig_xics /= np.nanmax(sig_xics)

            if self.LS_diag_rescaling: # LS renormalization 

                # recale all spectral data points with equal weight in LS minimization
                if self.use_LS_alpha_only:
                    # use only a scalar multiplier, alpha --- only works well if we use this on the brightest lines, not the lower amplitude ones
                    mask = self.signals[0].y_norm> 0.05  # use rescaling such that we attempt to match only main lines
                    scale = np.nansum(sig_xics[mask] * self.signals[0].y_norm[mask] /self.signals[0].std_y_norm[mask]**2)/\
                            (np.nansum((sig_xics[mask] / self.signals[0].std_y_norm[mask])**2)+1e-10)
                    sig_xics *= scale

                else:
                    # LS scaling with both multiplier and flat background - no masking of low photon levels
                    # see F. Sciortino thesis, chapter 5
                    gam = np.nansum(self.signals[0].y_norm/self.signals[0].std_y_norm**2)/\
                          (np.nansum((1.0 / self.signals[0].std_y_norm)**2)+1e-10)
                    zet = np.nansum(sig_xics/self.signals[0].std_y_norm**2)/\
                          (np.nansum((1.0 / self.signals[0].std_y_norm)**2)+1e-10)
                    alph = np.nansum((sig_xics - gam)*self.signals[0].y_norm/self.signals[0].std_y_norm**2)/\
                           (np.nansum((self.signals[0].y_norm**2 - zet*self.signals[0].y_norm)/self.signals[0].std_y_norm**2))
                    bet = gam - alph * zet

                    sig_xics = alph * sig_xics + bet

            sig.append(sig_xics)

            if debug_plots: # XICS signals plotting

                # original data and fit
                self.signals[0].plot_data(y_synth=sig[0], norm=self.normalize)

                if hasattr(self, 'truth_data') and self.truth_data is not None:
                    ax.plot(
                        self.signals[0].t,  self.truth_data.sig_norm[0][:, k] if self.normalize else self.truth_data.sig_abs[0][:, k],
                        '.--' )

        if dlines is None:
            # only XICS is being modelled:
            return sig

        #######################
        # Now for VUV and SXR:      
        
        if sigsplines is None:
            sigsplines = self.dlines2sigsplines(dlines, self.time, self.signals[1:])

        # add None for XICS, not actually used in sigsplines currently
        sigsplines = [None,]+sigsplines

        # collect synthetic and expt signals in dict for VUV and SXR only
        comp = {}
        for j, s in enumerate(self.signals):
            if j==0: 
                continue

            if s is not None and self.signal_mask[j]:

                sspl = sigsplines[j]

                # store experimental signals in comp dict to allow more flexibility on normalization
                comp[j] = {}
                comp[j]['t'] = copy.deepcopy(s.t)
                comp[j]['y'] = copy.deepcopy(s.y_norm if self.normalize else s.y)
                comp[j]['std_y'] = copy.deepcopy(s.std_y_norm if self.normalize else s.std_y)

                out_arr = np.zeros_like(s.y)

                # Use postinj to zero out before the injection:
                postinj = s.t >= -  param_tshifts[j]

                # average signals in bins corresponding to diagnostic time resolution 
                time_bins = np.array([np.linspace(s.t[postinj][t_bin] +  param_tshifts[j] - self.diag_time_res[j]/2.0 ,
                                                  s.t[postinj][t_bin] +  param_tshifts[j] + self.diag_time_res[j]/2.0, 
                                                  10, endpoint=True) for t_bin in np.arange(len(s.t[postinj]))])

                out_arr[postinj, :] = np.array([np.nanmean([sspl[i](x,extrapolate=True) for x in time_bins],axis=1) 
                                                for i in np.arange(0, s.y.shape[1])]).T

                # last time points cannot be larger than previous ones (in decay phase). Assume extrapolation issue:
                #for i in np.arange(0, s.y.shape[1]):
                #    if out_arr[-2,i]>out_arr[-3,i]: out_arr[-2,i]=out_arr[-3,i]
                #    if out_arr[-1,i]>out_arr[-2,i]: out_arr[-1,i]=out_arr[-2,i]

                # It is possible that sometimes interpolating spline function above gives -ve signals at the end of a time series
                out_arr[out_arr<0] =  0.0

                # Do the normalization and scaling for each normalization block:
                for b in np.unique(s.blocks):
                    mask = s.blocks == b

                    # ----------------------------
                    # Normalization:
                    if self.normalize:
                        norm_val = np.nanmax(out_arr[:, mask])
                        out_arr[:, mask] = out_arr[:, mask] / norm_val

                    if (j==1 or j==2) and self.LS_diag_rescaling:
                        # least-squares analytic minimization, only for VUV or XTOMO (see below for XICS)
                        scale = np.nansum(out_arr[:, mask] * s.y_norm[:, mask]/ s.std_y_norm[:, mask]**2)/\
                                (np.nansum((out_arr[:, mask]/ s.std_y_norm[:, mask])**2)+1e-10)
                        out_arr[:, mask] *= scale      

                comp[j]['y_synth'] = copy.deepcopy(out_arr)


        #store signals. If one is masked out, add an empty list
        for j, s in enumerate(self.signals):
            if j==0: 
                continue
            if s is not None and self.signal_mask[j]:
                sig.append(comp[j]['y_synth'])
            else:
                sig.append([])


        if debug_plots: # plotting of VUV and SXR signals

            for i, s in enumerate(self.signals):
                if i==0: # XICS is dealt with above
                    continue
                if s is not None and self.signal_mask[i]:
                    # plot on time bases that are *not* shifted by param_tshifts (for simplicity)
                    f, a = s.plot_data(norm=self.normalize, ncol=int(np.ceil(np.sqrt(s.y.shape[1]))))
                    srt = s.t.argsort()
                    
                    for k, ax in enumerate(a):
                        ax.plot(s.t[srt], sig[i][srt, k], '.-')  

                    if hasattr(self, 'truth_data') and self.truth_data is not None:
                        ax.plot(
                            self.truth_data.xtomo_time if i==2 else s.t,
                            self.truth_data.sig_norm[i][:, k] if self.normalize else self.truth_data.sig_abs[i][:, k],
                            '.--'
                        )

        return sig


    def compute_xics_lam_weights(self):
        '''Compute wavelength weights to increase/decrease the importance of some wavelength ranges
        over others in the XICS Ca K-alpha spectrum.
        '''
        with open('/home/sciortino/BITS/hirexsr_wavelengths.csv', 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])

        # select only lines from Ca
        xics_lams = lineLam[lineZ==20]
        xics_names = lineName[lineZ==20]
        
        xics_names = xics_names[(xics_lams>3.17)&(xics_lams<3.215)]
        xics_lams = xics_lams[(xics_lams>3.17)&(xics_lams<3.215)]
        
        self.xics_lines_info = xics_lines_info = {}
        for ii,name in enumerate(xics_names):
            xics_lines_info[name] = xics_lams[ii]

        # assign weights at every wavelength
        xics_lam_weights = np.ones_like(self.signals[0].lams)

        if 'xmsty' in self.highlight_lines:
            # increase weight of spectral region containing x,m,s,t and y

            for ch in np.arange(self.signals[0].lams.shape[1]):            
                lam_xmin = 3.186 #A
                lam_ymax = 3.195 #A
                wvls_idx0 = np.argmin(np.abs(self.signals[0].lams[:,ch] - lam_xmin))
                wvls_idx1 = np.argmin(np.abs(self.signals[0].lams[:,ch] - lam_ymax))
                
                xics_lam_weights[wvls_idx0:wvls_idx1,ch] *= self.xics_lam_wf

        else:
            # increase weight of specific lines of interest

            # wavelengths of interest
            wvls = np.zeros(len(self.highlight_lines))
            for ii,name in enumerate(self.highlight_lines):
                wvls[ii] = xics_lines_info[name]

            for ch in np.arange(self.signals[0].lams.shape[1]):
                for wvl in wvls:
                    # increase weights of any other lines
                    wvls_idx0 = np.argmin(np.abs(self.signals[0].lams[:,ch] - (wvl - self.xics_wf_dlam_A)))
                    wvls_idx1 = np.argmin(np.abs(self.signals[0].lams[:,ch] - (wvl + self.xics_wf_dlam_A)))
                    
                    xics_lam_weights[wvls_idx0:wvls_idx1,ch] *= self.xics_lam_wf

        # cache for later use in sig2diffs
        self.xics_lam_weights = xics_lam_weights


    def sig2diffs(self, sig):
        """Computes the individual diagnostic differences corresponding to the given signals.
        
        Parameters
        ----------
        sig : list of arrays of float
            The diagnostic signals. There should be one entry for each element
            of :py:attr:`self.signals` or :py:attr:`self.ar_signals`. Each entry should be an array
            of float with shape (`n_time`, `n_chords`).
        """
        sig_diff = []               

        # Apply wavelength weighting for XICS
        if self.signal_mask[0]:
            sig_diff.append(self.xics_lam_weights[:,None,:] *(sig[0] - (self.signals[0].y_norm if self.normalize else self.signals[0].y)))

        # no wavelength weights for VUV and SXR
        if self.signal_mask[1]:
            sig_diff.append(sig[1] - (self.signals[1].y_norm if self.normalize else self.signals[1].y))

        if self.signal_mask[2]:
            sig_diff.append(sig[2] - (self.signals[2].y_norm if self.normalize else self.signals[2].y))

        # Simpler loops with no weights for XICS wavelengths
        #for i, (s, ss) in enumerate(zip(sig, self.signals)):
        #    if ss is not None and self.signal_mask[i]:
        #        sig_diff.append(s - (ss.y_norm if self.normalize else ss.y))

        return sig_diff

    

    
    def diffs2ln_prob(
        self,
        sig_diff,
        params=None,
        sign=1.0,
        no_prior=False
        ):
        """Computes the log-posterior corresponding to the given differences.
        
        If there is a NaN in the differences, returns `-scipy.inf`.
        
        Here, the weighted differences :math:`\chi^2` are given as
        
        .. math::
            
            \chi^2 = \sum_i \left ( w_{i}\frac{b_{aurora, i} - b_{data, i}}{\sigma_i} \right )^2
         
        In effect, the weight factors :math:`w_i` (implemented as keywords
        `s_weight`, `v_weight` and `xtomo_weight`) let you scale the uncertainty
        for a given diagnostic up and down. A higher weight corresponds to a
        smaller uncertainty and hence a bigger role in the inference, and a
        lower weight corresponds to a larger uncertainty and hence a smaller
        role in the inference.
        
        The log-posterior itself is then computed as
        
        .. math::
            
            \ln p \propto -\chi^2 / 2 + \ln p(D, V)
        
        Here, :math:`\ln p(D, V)` is the log-prior.
        
        Parameters
        ----------
        sig_diff : list of arrays of float
            The diagnostic signal differences. There should be one entry for
            each element of :py:attr:`self.signals`. Each entry should
            be an array of float with shape (`n_time`, `n_chords`).
        params : array of float
            The parameters to use. If absent, :py:attr:`self.params` is used.
        sign : float, optional
            Sign (or other factor) applied to the final result. Set this to -1.0
            to use this function with a minimizer, for instance. Default is 1.0
            (return actual log-probability).
        no_prior: bool, optional
            Whether to include the prior or not. Note that this is included as an option to 
            distinguish between cases where this function should act as a log-likelihood and cases
            where it should be the full posterior. For use in pymultinest, no_prior should be set 
            to True, since the prior must be given separately to the pymultinest.run call. 
            Default is False (for use in MultiNest).
        """
        if params is not None:
            params = np.asarray(params, dtype=float)
            if len(params) == self.num_params:
                self.params = params
            else:
                self.free_params = params
                
        if self.free_d_weights:
            # get diagnostic weights from parameters array
            D_vals, V_vals, knots_D, knots_V, param_tshifts, d_weights, mix_rad_corr, D_gauss, V_gauss, sawtooth_width,\
                rcl, tau_divsol, n0x, zeta = self.split_params()
        else:
            if self.d_weights is None:
                d_weights = [1.0,] * self.nW   #len(sig_diff)     # changed 10/22/20
            else:
                d_weights=self.d_weights

        # Now VUV and SXR:
        signals_masked = [s for i,s in enumerate(self.signals) if self.signals[i] is not None and self.signal_mask[i]]

        lnlike = 0.0
        for ii, (w, s, ss) in enumerate(zip(d_weights, sig_diff, signals_masked)):

            sigmas = ss.std_y_norm if self.normalize else ss.std_y

            # approximate (but fast) accounting of correlations (see full calculation above)
            dnorm2 = (s / sigmas )**2.0 / (1.0 + 2.0 * self.gauss_corr_coeff) # up and down correlation

            # convert to loglike value, depending on Bayesian diagnostic combination method
            lnlike += self.chi2_to_loglike(dnorm2, sigmas, w, ind=ii)

        # This indicates that the brightness differences were all NaN:
        if lnlike == 0.0:            
            return -sign * scipy.inf

        elif self.check_particle_conservation and\
             (self.particle_penetration_ratio<self.min_penetration_fraction or\
              self.particle_loss_ratio>self.max_p_conserv_loss):  #check particle penetration AND conservation
            
            return -sign * 1e30 #scipy.inf              # use very large number to avoid internal multinest warning

        else:
            if no_prior:  #use for MultiNest, return log-likelihood
                lp = sign *  lnlike
            else:  # use for algorithms maximizing log-posterior
                priors = self.get_prior()
                lp = sign * ( lnlike + np.sum([priors[ii].logpdf(self.params[ii]) for ii in np.arange(len(self.params))]))
            return lp
    

    def chi2_to_loglike(self, dnorm2, sigmas, w=1.0, ind=0):
        ''' compute loglikelihood value from array of chi^2 values for a certain diagnostic. 

        Parameters
        ------------------
        dnorm2: chi^2 vector
        sigmas: vector of uncertainties for current signal
        w: scalar weight to be applied (leave to default of 1 if not applicable to diagnostic combination method. 
        ind: index of diagnostic in order of looping. THIS IS CURRENTLY NOT USED but it is kept as an
            argument because it could turn out to be useful later on.

        Returns
        -----------
        float
             Log-likelihood value
        '''

        if self.CDI==-1: # Cauchy likelihood
            # middle term gives +ve lnev (values of ~+3000)
            cauchy_lnlike = + 0.5 * np.log( w ) - np.log(np.pi * sigmas) - np.log( 1 +w * dnorm2 )
            lnlike_val = cauchy_lnlike[~scipy.isnan(cauchy_lnlike)].sum()  # log-likelihood

        elif self.CDI==0:
            # standard Gaussian likelihood, allowing for variable diagnostic weights

            # include Gaussian normalization, accounting for weight factor modification
            gaussian_lnlike = + 0.5 * np.log( w ) - np.log( np.sqrt(2 * np.pi) * sigmas ) - 0.5 * w * dnorm2     
            lnlike_val = gaussian_lnlike[~scipy.isnan(gaussian_lnlike)].sum()

        else:

            # Combined dataset inference with Gaussian error modeling (does not use explicit weights)
            mask = ~scipy.isnan(dnorm2)

            # 4/12/21: allow for simple multiplication by arbitrary weight
            dnorm2 *= w

            # get total number of (good) data points for current diagnostic:
            nk = 1 
            for dim in np.shape(dnorm2[mask]): nk *= dim

            if self.CDI==1:  # exponential prior on alpha_k (mean=1)
                lnlike_val =  - np.log( np.sqrt(2 * np.pi) * sigmas[mask]).sum() \
                              + scipy.special.gammaln(nk/2. +1) \
                              - 0.5 * (nk+2) * np.log( dnorm2[mask].sum() / 2. + 1)

            elif self.CDI==2 or self.CDI==3 or self.CDI==4:  #gamma prior (CDI=3 and 4 have CDI_a=1/CDI_b)

                lnlike_val =  - np.log( np.sqrt(2 * np.pi) * sigmas[mask]).sum() \
                            - scipy.special.gammaln(self.CDI_a) \
                            - self.CDI_a * np.log(self.CDI_b) \
                            - (nk/2. + self.CDI_a) * np.log( dnorm2[mask].sum() / 2. + 1./self.CDI_b) \
                            + scipy.special.gammaln(nk/2. +self.CDI_a) 

            else:
                raise ValueError('CDI not recognized! It was neither -1, 0, 1, 2, 3 or 4...')
                
            # 4/12/21: add constant factor for normalized probability
            lnlike_val += 0.5 * np.log( w )

        return lnlike_val



    def dlines2sigsplines(self, dlines, time, signals):
        """Convert the given diagnostic lines (from aurora calculation) to splines which can be used to interpolate onto the diagnostic timebase.
        """
        # shift aurora time base s.t. simulation time domain begins at 0 s:
        time = time - self.time_1   

        # use Akima1D splines, which are smooth, have small (good) overshoot and are fast
        return [
            [
                scipy.interpolate.Akima1DInterpolator( 
                    time,
                    dlines[s.atomdat_idx[i]-1, :, :].dot(s.weights[i, :])       
                    # possibly add argument ext=3 for constant extrapolation with IUS
                )
                for i in np.arange(s.y.shape[1])
            ]
            if s is not None else None
            for j, s in enumerate(signals)
        ]


    # Wrapper functions:
    
    def DV2dlines(self, params=None, 
                  explicit_D=None, explicit_D_roa=None, explicit_V=None, explicit_V_roa=None, 
                  debug_plots=False):
        """Computes the local emissivities corresponding to the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines`. See those functions for argument descriptions.
        
        Parameters
        ----------
     
        """
        cs_den = self.DV2cs_den(params=params,
                                explicit_D=explicit_D,
                                explicit_D_roa=explicit_D_roa,
                                explicit_V=explicit_V,
                                explicit_V_roa=explicit_V_roa,
                                debug_plots=debug_plots
                            )

        return self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots )

    
    def cs_den2sig(self, cs_den, params=None, debug_plots=False):
        """Computes the diagnostic signals corresponding to the given charge state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig`. See those functions for argument descriptions.
        """
        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)

        return self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)


    
    def dlines2diffs(self, dlines, dlines_xics, params=None, debug_plots=False, sigsplines=None):
        """Computes the diagnostic differences corresponding to the given local emissivities.
        
        This is simply a wrapper around the chain of :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs`. See those functions for argument descriptions.
        """
        sig = self.dlines2sig(dlines, dlines_xics, params=params,
                              debug_plots=debug_plots, sigsplines=sigsplines)

        return self.sig2diffs(sig)
    

    def sig2ln_prob(self, sig, params=None, sign=1.0):
        """Computes the log-posterior corresponding to the given diagnostic signals.
        
        This is simply a wrapper around the chain of :py:meth:`sig2diffs` ->
        :py:meth:`diffs2ln_prob`. See those functions for argument descriptions.
        """
        sig_diff = self.sig2diffs(sig)

        return self.diffs2ln_prob(sig_diff, params=params, sign=sign)
    


    def DV2sig(self, params=None, 
               explicit_D=None, explicit_D_roa=None, explicit_V=None, explicit_V_roa=None,
               debug_plots=False):
        """Predicts the diagnostic signals that would arise from the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig`. See those functions
        for argument descriptions.
        """

        cs_den = self.DV2cs_den(params=params,
                             explicit_D=explicit_D, explicit_D_roa=explicit_D_roa, 
                             explicit_V=explicit_V, explicit_V_roa=explicit_V_roa, 
                             debug_plots=debug_plots )


        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)

        return self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)

    
    def cs_den2diffs(self, cs_den, params=None, debug_plots=False):
        """Computes the diagnostic differences corresponding to the given charge state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig` -> :py:meth:`sig2diffs`. See those functions for
        argument descriptions.
        """
        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)

        sig = self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)

        return self.sig2diffs(sig)
    

    def dlines2ln_prob(self, dlines, dlines_xics, params=None, debug_plots=False, sign=1.0,
                       no_prior=False, sigsplines=None):
        """Computes the log-posterior corresponding to the given local emissivities.
        
        This is simply a wrapper around the chain of :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`. See those functions
        for argument descriptions.
        """
        sig = self.dlines2sig(dlines, dlines_xics, params=params, 
                              debug_plots=debug_plots, sigsplines=sigsplines)

        sig_diff = self.sig2diffs(sig)

        return self.diffs2ln_prob(sig_diff, params=params, sign=sign, no_prior=no_prior)

    
    def DV2diffs(self, params=None, 
                 explicit_D=None, explicit_D_roa=None, explicit_V=None, explicit_V_roa=None,
                 debug_plots=False):
        """Computes the diagnostic differences corresponding to the given parameters.
        
        The is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig` -> :py:meth:`sig2diffs`.
        See those functions for argument descriptions.
        """
        
        cs_den = self.DV2cs_den(params=params,
                             explicit_D=explicit_D, explicit_D_roa=explicit_D_roa, 
                             explicit_V=explicit_V, explicit_V_roa=explicit_V_roa,
                             debug_plots=debug_plots)

        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)

        sig = self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)

        return self.sig2diffs(sig)
    

    def cs_den2ln_prob(self, cs_den, params=None, debug_plots=False, sign=1.0):
        """Computes the log-posterior corresponding to the given charge-state densities.
        
        This is simply a wrapper around the chain of :py:meth:`cs_den2dlines` ->
        :py:meth:`dlines2sig` -> :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`.
        See those functions for argument descriptions.
        """
        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)

        sig = self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)

        sig_diff = self.sig2diffs(sig)

        return self.diffs2ln_prob(sig_diff, params=params, sign=sign)
    
    
    def DV2ln_prob(self, params=None, sign=1.0, 
                   explicit_D=None, explicit_D_roa=None, explicit_V=None, explicit_V_roa=None,
                   debug_plots=False, no_prior=True):
        """Computes the log-posterior corresponding to the given parameters.
        
        This is simply a wrapper around the chain of :py:meth:`DV2cs_den` ->
        :py:meth:`cs_den2dlines` -> :py:meth:`dlines2sig` ->
        :py:meth:`sig2diffs` -> :py:meth:`diffs2ln_prob`. See those functions
        for argument descriptions. This is designed to work as a log-posterior
        function for various MCMC samplers, etc.
        
        Parameters
        ----------

        """
        cs_den = self.DV2cs_den(params=params, 
                                explicit_D=explicit_D, explicit_D_roa=explicit_D_roa,
                                explicit_V=explicit_V, explicit_V_roa=explicit_V_roa,
                                debug_plots=debug_plots)
    
        dlines, dlines_xics = self.cs_den2dlines(cs_den, params=params, debug_plots=debug_plots)
        # dlines.shape --> (time,n_lines, space) 
        
        sig = self.dlines2sig(dlines, dlines_xics, params=params, debug_plots=debug_plots)
        # sig[0].shape--> (Hirex-Sr times, n_chords)

        sig_diff = self.sig2diffs(sig)
        # sig_diff[0].shape --> same as sig[0].shape

        return self.diffs2ln_prob(sig_diff, params=params, sign=sign, no_prior=no_prior)
    


    # =====================================
    # 
    #
    #
    # ====================================
    # Routines for multinest:

    def set_copula(self, _cube, Sig_copula):
        ''' Apply Gaussian copula to the free parameters in the unit hypercube. 
        
        Parameters
        ----------
        cube : array of float
            The variables in the unit hypercube for EITHER D or V
        Sig_copula : numpy.ndarray
            Square root of covariance matrix used for the gaussian copula. 
            This can be obtained via a Cholesky or eigenvalue decomposition.
        '''
        # transform to Gaussian IID samples
        zs = [self.norm.ppf(xx) for xx in _cube]
        
        # correlated Gaussian samples
        ws_csky = Sig_copula.dot(zs)

        # correlated uniform samples (Gaussian copula):
        return [self.norm.cdf(val) for val in ws_csky]


    def remap_coeff_by_gradient(self, pp, knotgrid, max_grad, lb, ub):
        '''Apply linear re-mapping of transport coefficients in the pp array to prevent gradients 
        larger than `max_grad`. 
        '''
        # Delta(r/a) values
        spacings = np.diff(knotgrid)  
        
        # maximum acceptable gradients in transport coefficients given by max_grad
        Delta = spacings * max_grad
        
        # now get correlated parameters in sequence (leave inner-most unvaried)    
        for i in np.arange(1, len(pp)):
            # near-axis np.arange
            a1 = max([lb[i-1], pp[i-1] - Delta[i]])   #need Delta[i], not i-1!
            a2 = min([pp[i-1] + Delta[i], ub[i-1]])
            pp[i] = a1 + ((a2 - a1)/(ub[i-1] - lb[i-1])) * (pp[i] - lb[i-1])  

        return pp



    def remap_coeff_by_gradient_old(self, pp, knotgrid, max_grad, 
                                axis_lb, axis_ub, 
                                lb_midradius, ub_midradius,
                                lb_outer, edge_ub,
                                num_axis_coeffs, num_mid_coeffs, num_edge_coeffs,
                                correlated_ped=False):
        '''
        Apply linear re-mapping of transport coefficients in the pp array to prevent gradients 
        larger than `max_grad`. 

        Note that correlated_ped should only be applied if bounds for the given 
        transport coefficients become larger as one goes to larger radii -- otherwise, one can 
        run into problems with "disconnected regions" and this technique fails. 
        '''
        # Delta(r/a) values
        spacings = np.diff(knotgrid)  
        
        # maximum acceptable gradients in transport coefficients given by max_grad
        Delta = spacings * max_grad
        
        # now get correlated parameters in sequence (leave inner-most unvaried)    
        for i in np.arange(1, num_axis_coeffs):
            # near-axis np.arange
            a1 = max([axis_lb, pp[i-1] - Delta[i]])   #need Delta[i], not i-1!
            a2 = min([pp[i-1] + Delta[i], axis_ub])
            pp[i] = a1 + ((a2 - a1)/(axis_ub - axis_lb)) * (pp[i] - axis_lb)  
            
        for i in np.arange(num_axis_coeffs, num_axis_coeffs + num_mid_coeffs):
            # central np.arange
            a1 = max([lb_midradius, pp[i-1] - Delta[i]]) 
            a2 = min([pp[i-1] + Delta[i], ub_midradius])
            pp[i] = a1 + ((a2 - a1)/(ub_midradius - lb_midradius)) * (pp[i] - lb_midradius)   

        # check if correlations can be applied in the pedetal:
        if edge_ub<ub_midradius:
            correlated_ped = False

        if correlated_ped:
            # For this to work, bound in the pedestal must be the same as at midradius
            ped_start_idx = num_axis_coeffs + num_mid_coeffs
            
            for i in np.arange(ped_start_idx, ped_start_idx+ num_edge_coeffs):
                a1 = max([lb_outer, pp[i-1] - Delta[i]])
                a2 = min([pp[i-1] + Delta[i], edge_ub])
                pp[i] = a1 + ((a2 - a1)/(edge_ub - lb_outer)) * (pp[i] - lb_outer) 

        return pp


    def multinest_prior(self, cube):
        """Prior distribution function for :py:mod:`pymultinest`.
        
        Maps the (free) parameters in `cube` from [0, 1] to real space.
        
        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.

        To test, use
        from numpy.random import uniform
        s.multinest_prior(uniform(size=len(s.free_params)))
        """
        # Set new knotgrids first, so that appropriate bounds can be set on D and V at different locations
        # when sampling from prior
 
        free_D_vals = ~self.fixed_params[:self.nD]   # booleans
        free_V_vals = ~self.fixed_params[self.nD:self.nD+self.nV]   # booleans
        
        num_D = np.sum(free_D_vals)  # number of free D values
        num_V = np.sum(free_V_vals)  # number of free V values

        if self.force_identifiability:
            # force identifiability of free knots by sampling them in uniform hyper-triangle
            # ignoring the simple uniform sampling done within self.get_prior() and imposing a min distance
            # Extension of Ref.  "PolyChord: next-generation nested sampling" (Handley et al. 2015). - Appendix A

            if self.free_D_knots and self.nkD>0: # D knots
                valmax = self.outermost_knot - self.min_knots_dist * self.nkD  # reduce 
                valmin = self.innermost_knot

                D_cube = cube[num_D+num_V: num_D+num_V+self.nkD]
                _D_knots = np.ones(self.nkD+1)*valmin  # extra dummy knot at 0
                
                for n,cubeval in enumerate(D_cube):
                    _D_knots[n+1] = valmax - pow(cubeval, 1.0 / (self.nkD - n))*(valmax - _D_knots[n])

                # eliminate first dummy knot and create new grid with imposed min distance
                D_knots = _D_knots[1:] + self.min_knots_dist*np.linspace(0, self.nkD, self.nkD)
                                                                            
                # set knotgrid_D --> allows updating of self.num_axis_D_coeffs, etc.
                self.knotgrid_D = np.concatenate(([self.innermost_knot,], D_knots, [self.outermost_knot,]))  

            if self.free_V_knots and self.nkV>0: # V knots  
                valmax = self.outermost_knot - self.min_knots_dist * self.nkV
                valmin = self.innermost_knot

                V_cube = cube[num_D+num_V+self.nkD:num_D+num_V+self.nkD+self.nkV]
                _V_knots = np.ones(self.nkDV+1)*valmin  # extra dummy knot at 0
                
                for n,cubeval in enumerate(V_cube):
                    _V_knots[n+1] = valmax - pow(cubeval, 1.0 / (self.nkV - n))*(valmax - _V_knots[n])

                # eliminate first dummy knot and create new grid with imposed min distance
                V_knots = _V_knots[1:] + self.min_knots_dist*np.linspace(0, self.nkV, self.nkV)

                # set knotgrid_D--> allows updating of self.num_axis_V_coeffs, etc.
                self.knotgrid_V = np.concatenate(([self.innermost_knot,], V_knots, [self.outermost_knot,]))   

        elif self.nkD>0 or self.nkV>0: # free knots, but not with forced-identifiability method
            
            # must sample knots first, so that self.num_axis_V_coeffs, etc. can be set for current iteration
            prior_knots, bounds_knots = self.get_DV_spline_knots_prior()   # likely slow
            u_knots = prior_knots().elementwise_cdf(self.params[self.nD+self.nV:self.nD+self.nV+self.nkD+self.nkV])

            if self.free_D_knots:
                u_knots[:self.nkD] = cube[num_D+num_V:num_D+num_V+self.nkD] #values of u are in [0,1]

            if self.free_V_knots:
                u_knots[self.nkD:] = cube[num_D+num_V+self.nkD:num_D+num_V+self.nkD+self.nkV]  

            # get uncorrelated sampled parameters (including those that we do not set correlations on):
            #p = prior_knots.sample_u(u_knots)   # evaluate inverse CDF
            p = np.array([prior_knots[ii].ppf(u_knots[ii]) for ii in np.arange(len(u_knots))])

            if self.nkD>0:
                D_knots = p[:self.nkD]
                self.knotgrid_D = np.concatenate(([self.innermost_knot,], D_knots, [self.outermost_knot,]))   

            if self.nkV>0:
                V_knots = p[self.nkD:self.nkD+self.nkV]
                self.knotgrid_V = np.concatenate(([self.innermost_knot,], V_knots, [self.outermost_knot,]))  
        else:
            pass

        ######
        if self.copula_corr:
            # set Gaussian copula correlations between spline D values and V values (separately)
            cube[:num_D] = self.set_copula(cube[:num_D], self.Sig_copula_D)
            cube[num_D:num_D+num_V] = self.set_copula(cube[num_D:num_D+num_V], self.Sig_copula_V)


        ######## Sampling of most parameters #######
        # Need to use self.params so that we can handle fixed params:
        priors = self.get_prior()
        u = np.array([priors[ii].cdf(self.params[ii]) for ii in np.arange(len(self.params))])
        u[~self.fixed_params] = cube  #values of u are in [0,1]
        
        # get uncorrelated sampled parameters (including those that we do not set correlations on):
        p = np.array([priors[ii].ppf(u[ii]) for ii in np.arange(len(u))])

        ####################################

        # Now substitute knots as sampled above, either from forced-identifiability non-separable uniform prior
        # or from simple uniform prior in self.get_prior() -- in the latter case, the substitution shouldn't make
        # any difference, but the knots sampling MUST happen first so that self.num_axis_D_coeffs and similar
        # variables are set before D and V sampling occurs.
        if self.free_D_knots and self.nkD>0:
            p[self.nD+self.nV: self.nD+self.nV+self.nkD] = D_knots
        if self.free_V_knots and self.nkV>0: # V knots     
            p[self.nD+self.nV+self.nkD: self.nD+self.nV+self.nkD+self.nkV] = V_knots


        #####
        # deactivated; this shouldn't be used now that we have copulas and non_separable_DV_priors
        # max-grad setting; set here fixed parameters in place so that maximum gradients can be set appropriately with any fixed D and V
        # p[self.fixed_params] = copy.deepcopy(self.params[self.fixed_params])
        
        # if self.set_max_grad_D:
        #     p[:self.nD] = self.remap_coeff_by_gradient(copy.deepcopy(p[:self.nD]), 
        #                                                self.knotgrid_D, self.max_D_grad, 
        #                                                self.D_lb, self.D_axis_ub,
        #                                                self.D_lb, self.D_mid_ub,
        #                                                self.D_lb, self.D_edge_ub,
        #                                                self.num_axis_D_coeffs, self.num_mid_D_coeffs, self.num_edge_D_coeffs,
        #                                                correlated_ped = self.correlated_D_ped)
            
        # if self.set_max_grad_V:
        #     p[self.nD:self.nD+self.nV] = self.remap_coeff_by_gradient(copy.deepcopy(p[self.nD:self.nD+self.nV]), 
        #                                                               self.knotgrid_V, self.max_V_grad, 
        #                                                               self.V_axis_lb, self.V_axis_ub
        #                                                               self.V_mid_lb, self.V_mid_ub,
        #                                                               self.V_edge_lb, self.V_edge_ub,
        #                                                               self.num_axis_V_coeffs, self.num_mid_V_coeffs, self.num_edge_V_coeffs,
        #                                                               correlated_ped = self.correlated_V_ped)
            
   
        # apply priors on D, V and V/D
        if self.non_separable_DV_priors and self.learn_VoD and np.allclose(self.knotgrid_V, self.knotgrid_D):

            # REQUIRE sampling of D and V/D (not V)
            # REQUIRE identical knots between D and V/D
            assert num_D == num_V

            # define scale factor between D and V from their gaussian std ratio
            chi = np.array(
                [self.D_axis_std/self.V_axis_std, ] * self.num_axis_D_coeffs +
                [self.D_mid_std/self.V_mid_std, ] * self.num_mid_D_coeffs +
                [self.D_edge_std/self.V_edge_std, ] * self.num_edge_D_coeffs
            )

            # re-interpret D and V/D sampled values for polar re-mapping
            r_vals = p[:self.nD] 
            VoD_vals = p[self.nD:self.nD+self.nV]
            
            thetas = np.array([math.atan(val) for val in chi *VoD_vals])
            
            # project onto D,V coordinates
            D_out = r_vals * np.array([math.cos(theta) for theta in thetas])
            V_out = r_vals * np.array([math.sin(theta) for theta in thetas]) / chi

            # store new V/D sample
            p[self.nD:self.nD+self.nV]  = V_out / D_out

            D_out[D_out < self.D_lb] = self.D_lb
            p[:self.nD] = D_out       
            

        return p[~self.fixed_params]


    def multinest_prior_simple(self, cube):
        """Prior distribution function for :py:mod:`pymultinest`.
        
        Maps the (free) parameters in `cube` from [0, 1] to real space.
        
        Parameters
        ----------
        cube : array of float, (`num_free_params`,)
            The variables in the unit hypercube.
        """
        if self.copula_corr:
            # set Gaussian copula correlations between spline D values and V values (separately)
            cube[:self.nD] = self.set_copula(cube[:self.nD], self.Sig_copula_D)
            cube[self.nD:self.nD+self.nV] = self.set_copula(cube[self.nD:self.nD+self.nV], self.Sig_copula_V)

        # Need to use self.params so that we can handle fixed params:
        priors = self.get_prior()
        u = np.array([priors[ii].cdf(self.params[ii]) for ii in np.arange(len(self.params))])
        u[~self.fixed_params] = cube  #values of u are in [0,1]
        p = np.array([priors[ii].ppf(u[ii]) for ii in np.arange(len(u))])  # sampled parameters

        return p[~self.fixed_params]
    

    def multinest_ll(self, params):
        """Log-likelihood function for py:mod:`pymultinest`. 
        
        Parameters
        ----------
        params : array of float, (`num_free_params`,)
            The free parameters.
        """
        ll = -scipy.inf #-1e99 #-scipy.inf   # use extremely small number to avoid MultiNest internal warning

        try:
            ll = self.DV2ln_prob(params=params, no_prior=True)
        except:
            warnings.warn("Something went wrong within the forward model!")   

        return ll

    
    def run_multinest(self, basename=None, n_live_points=400, **kwargs):
        """Run the multinest sampler.
        """
        if basename is None:
            basename = os.path.abspath('../chains_%d_%d/c-' % (self.shot, self.version))
        chains_dir = os.path.dirname(basename)
        if chains_dir and not os.path.exists(chains_dir):
            os.mkdir(chains_dir)
            
        n_dims = (~self.fixed_params).sum()
        kwargs['outputfiles_basename'] = basename
        kwargs['n_live_points'] = n_live_points

        pymultinest.solve(
            self.multinest_ll,
            self.multinest_prior, 
            n_dims,
            **kwargs
        )

    

    ############# Nonlinear optimizer solutions #################
    def find_MAP_estimate(self, random_starts=None, num_proc=None,  
                          theta0=None, thresh=None):
        """Find the most likely parameters given the data.
        
        Parameters
        ----------
        random_starts : int, optional
            The number of times to start the optimizer at a random point in
            order to find the global optimum. If set to None (the default), a
            number equal to twice the number of processors will be used.
        num_proc : int, optional
            The number of cores to use. If set to None (the default), half of
            the available cores will be used.
        theta0 : array of float, optional
            The initial guess(es) to use. If not specified, random draws from
            the prior will be used. This overrides `random_starts` if present.
            This array have shape (`num_starts`, `ndim`) or (`ndim`,).
        thresh : float, optional
            The threshold to continue with the optimization.
        """
        if num_proc is None:
            num_proc = multiprocessing.cpu_count() // 2

        # create parallelization pool
        pool = InterruptiblePool(processes=num_proc)

        if random_starts is None:
            random_starts = 2 * num_proc

        if theta0 is None:
            priors = self.get_prior()
            # this might need to be corrected...
            param_samples = np.array([priors[ii].rsv(size=random_starts) for ii in np.arange(len(priors))])[~self.fixed_params,:].T
        else:
            param_samples = scipy.atleast_2d(theta0)

        t_start = time_.time()
        if pool is not None:
            res = pool.map(
                _OptimizeEval(self, thresh=thresh),
                param_samples
            )
        else:
            res = map(
                _OptimizeEval(self, thresh=thresh),
                param_samples
            )
        t_elapsed = time_.time() - t_start
        print("All done, wrapping up!")
        res_min = max(res, key=lambda r: r[1])

        print("MAP estimate complete. Elapsed time is %.2fs. Got %d completed." % (t_elapsed, len(res)))

        # Estimate the AIC and BIC at the best result (should technically use MLE, not MAP...)
        print("Estimating AIC, BIC...")
        ll_hat = self.DV2ln_prob(params=res_min[0], no_prior=True)
        num_params = (~self.fixed_params).sum()
        num_data = 0
        ss = self.signals
        for s in ss:
            if s is not None:
                num_data += (~scipy.isnan(s.y)).sum()
        AIC = 2.0 * (num_params - ll_hat)
        BIC = -2.0 * ll_hat + num_params * scipy.log(num_data)
        
        out = {
            'all_results': res,
            'best_result': res_min,
            # 'best_covariance': cov_min,
            'best_AIC': AIC,
            'best_BIC': BIC
        }
        
        return out

    ################# Others ###################
    
    def get_prior_samples(self, nsamp=1000, plot=False):
        """Plot samples from the prior distribution.
        
        Parameters
        ----------
        nsamp : int
            The number of samples to plot.
        """
        # Make sure to apply the same correlations used for inference:
        cubes =scipy.stats.uniform.rvs(size=(nsamp,self.num_free_params))
        draws = self.multinest_prior(cubes[0,:])
        for cube in cubes[1:]:
            draws = np.vstack([draws, self.multinest_prior(cube)])

        if plot:
            f = plt.figure()
            aD = f.add_subplot(3, 1, 1)
            aV = f.add_subplot(3, 1, 2)
            aVoD = f.add_subplot(3, 1, 3)
            
            for d in draws:
                # draw samples including pedestal gaussians and other non-spline features
                D_z, V_z, times_DV = self.eval_DV(imp='Ca',params=d)
                # assume time independent and plot only last charge state
                D = D_z[:,0,-1]; V = V_z[:,0,-1]

                aD.plot(self.roa_grid_DV, D, alpha=0.1)
                aV.plot(self.roa_grid_DV, V, alpha=0.1)
                aVoD.plot(self.roa_grid_DV, V/D, alpha=0.1)
                
            aD.set_ylabel('$D$ [m$^2$/s]')
            aV.set_ylabel('$v$ [m/s]')
            aVoD.set_ylabel('$v/D$ [m^{-1}]')
            aVoD.set_xlabel('$r/a$')

            plt.tight_layout()

        return draws



    
    def get_labels(self):
        """Get the labels for each of the variables included in the sampler.
        """
        # D,V spline values
        labels = (
            ['$C_{D,%d}$' % (n + 1,) for n in np.arange(0, self.nD)] +
            ['$C_{V,%d}$' % (n + 1,) for n in np.arange(0,self.nV)]
        )
        # knots
        labels += ['$t_{D,%d}$' % (n + 1,) for n in np.arange(0,self.nkD)]
        labels += ['$t_{V,%d}$' % (n + 1,) for n in np.arange(0,self.nkV)]

        # time base shifts
        labels += [r'$\Delta t$ %d' % (n,) for  n in np.arange(0,self.nDiag)]

        # explicit diagnostic weights 
        labels += [r'$w_{d,%d}$' % n for n in np.arange(0,self.nW)]

        # mixing radius correction
        labels += [r'$\Delta r_{mix}$'] #cm

        # amplitude, width and location of D pedestal
        labels += [r'$D_{ped}$']
        labels += [r'$\Delta D_{ped}$']
        labels += [r'$\langle r_{D,ped}\rangle$']

        # amplitude, width and location of V pedestal
        labels += [r'$V_{ped}$']
        labels += [r'$\Delta V_{ped}$']
        labels += [r'$\langle r_{V,ped}\rangle$']

        # width of sawtooth crash region
        labels += [r'$d_{saw}$']

        # recycling coefficient and time scale for divertor-SOL flows
        labels += [r'$rcl$']
        labels += [r'$\tau_{div-SOL}$']

        # neutral density multiplier
        labels += [r'$\eta$']

        # pedestal Z scaling
        labels += [r'$\zeta$']

        return labels
    
    
    def compute_marginalized_DV(self, sampler, burn=0, thin=1, chain_mask=None,
                                pool=None, weights=None, cutoff_weight=None, plot=False, compute_VD=False, 
                                compute_M=False, compute_robust_stats=False):
        """Computes and plots the marginal D, V profiles.

        Note that taking statistics from MCMC-like samples is equivalent to weighting all sampled
        parameters, including nuisance ones. In the case of MCMC methods there are no weights, 
        but the frequency of certain parameters (determined by the algorithm used, e.g. Metropolis-Hastings)
        is indicative of the relevance of each parameter. In MultiNest, weights are more efficiently
        computed, allowing for proper weighted statistics based on the relevance of each parameter value 
        within the set that were sampled. With either approach, this function gives the 'marginalized'
        D,V, meaning statistics for D,V in which the probability for each nuisance parameter was taken into
        consideration. 

        Finally, note that a MAP estimate, being a single-point estimate, does NOT marginalize over nuisance
        parameters, but only provides the single combination of sampled parameters that gave the best 
        posterior probability over the sampled set. 
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` instance or ndarray (pyMultiNest)
            The sampler to process the data from.
        burn : int, optional
            The number of samples to burn from the front of each walker. Default
            is zero.
        thin : int, optional
            The amount by which to thin the samples. Default is 1.
        chain_mask : mask array
            The chains to keep when computing the marginalized D, V profiles.
            Default is to use all chains.
        pool : object with `map` method, optional
            Multiprocessing pool to use. If None, `sampler.pool` will be used.
        weights : array of float, optional
            The weights to use (i.e., when post-processing MultiNest output).
        cutoff_weight : float, optional
            Throw away any points with weights lower than `cutoff_weight` times
            `weights.max()`. Default is to keep all points.
        plot : bool, optional
            If True, make a plot of D and V.
        compute_VD : bool, optional
            If True, compute and return V/D in addition to D and V.
        compute_M : bool, optional
            If True, compute the Mahalanobis distance between the marginalized
            D and V and `self.explicit_D`, `self.explicit_V`.
        get_robust_stats: bool, optional
            If True, return robust statistics (median & quantiles) as well as the mean and standard
            deviations. 
        """
        if pool is None:
            try:
                pool = sampler.pool
            except:
                pool = InterruptiblePool(multiprocessing.cpu_count()//2)
        
        try:
            k = sampler.flatchain.shape[-1]
        except AttributeError:
            # Assumes array input is only case where there is no "flatchain" attribute.
            k = sampler.shape[-1]
        
        if isinstance(sampler, emcee.EnsembleSampler):
            if chain_mask is None:
                chain_mask = np.ones(sampler.chain.shape[0], dtype=bool)
            flat_trace = sampler.chain[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
        elif hasattr(emcee, 'PTSampler') and isinstance(sampler, emcee.PTSampler): # PTSampler excluded from recent versions of emcee
            if chain_mask is None:
                chain_mask = np.ones(sampler.nwalkers, dtype=bool)
            flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
        elif isinstance(sampler, scipy.ndarray): #case of PyMultiNest
            if sampler.ndim == 4:
                if chain_mask is None:
                    chain_mask = np.ones(sampler.shape[1], dtype=bool)
                flat_trace = sampler[temp_idx, chain_mask, burn:, :]
                flat_trace = flat_trace.reshape((-1, k))
                if weights is not None:
                    weights = weights[temp_idx, chain_mask, burn:]
                    weights = weights.ravel()
            elif sampler.ndim == 3:
                if chain_mask is None:
                    chain_mask = np.ones(sampler.shape[0], dtype=bool)
                flat_trace = sampler[chain_mask, burn:, :]
                flat_trace = flat_trace.reshape((-1, k))
                if weights is not None:
                    weights = weights[chain_mask, burn:]
                    weights = weights.ravel()
            elif sampler.ndim == 2:  #standard pymultinest
                flat_trace = sampler[burn:, :]
                flat_trace = flat_trace.reshape((-1, k))
                if weights is not None:
                    weights = weights[burn:]
                    weights = weights.ravel()
            if cutoff_weight is not None and weights is not None:
                mask = weights >= cutoff_weight * weights.max()
                flat_trace = flat_trace[mask, :]
                weights = weights[mask]
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        
        DV_samp = np.asarray(
            pool.map(
                _ComputeProfileWrapper(self),
                flat_trace
            )
        )
        
        # close the pool, because otherwise processes seem to hang around
        pool.close()

        D_samp = DV_samp[:, 0, :,:]
        D_mean = profiletools.meanw(D_samp, axis=0, weights=weights)
        D_std = profiletools.stdw(D_samp, axis=0, ddof=1, weights=weights)
        
        V_samp = DV_samp[:, 1, :, :]
        V_mean = profiletools.meanw(V_samp, axis=0)
        V_std = profiletools.stdw(V_samp, axis=0, ddof=1, weights=weights)

        ############
        if compute_robust_stats:
            D_stats = np.array([get_robust_weighted_stats(D_samp[:,:,nn], 
                                                          weights=weights) for nn in np.arange(D_samp.shape[2])]).transpose(1,2,0)
            V_stats =  np.array([get_robust_weighted_stats(V_samp[:,:,nn], 
                                                           weights=weights) for nn in np.arange(V_samp.shape[2])]).transpose(1,2,0)

        ################

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # eval_DV returns (D,V) pairs even if learn_VoD=True
        out = [D_mean, D_std, V_mean, V_std]
        
        if compute_VD:
            VD_samp = DV_samp[:, 1, :, :] / DV_samp[:, 0, :, :]
            VD_mean = profiletools.meanw(VD_samp, axis=0, weights=weights)
            VD_std = profiletools.stdw(VD_samp, axis=0, ddof=1)
            
            out += [VD_mean, VD_std]
            
        if compute_robust_stats:
            out += [D_stats, V_stats] 
            
        if compute_VD and compute_robust_stats:
            #VD_stats =  get_robust_weighted_stats(V_samp, weights=weights)
            VD_stats = np.array([get_robust_weighted_stats(VD_samp[:,:,nn], 
                                                           weights=weights) for nn in np.arange(VD_samp.shape[2])]).transpose(1,2,0)
            out += [VD_stats]

        # ~~~~~~~~~~~~~~~~~~~~~~~

        if compute_M:
            # First, interpolate the true profiles onto the correct D, V grid:
            D_point = scipy.interpolate.InterpolatedUnivariateSpline(
                self.explicit_D_roa, self.explicit_D
            )(scipy.sqrt(self.rhop**2))
            V_point = scipy.interpolate.InterpolatedUnivariateSpline(
                self.explicit_V_roa, self.explicit_V
            )(scipy.sqrt(self.rhop**2))

            DV_point = scipy.hstack((D_point, V_point))
            DV = scipy.hstack((DV_samp[:, 0, :], DV_samp[:, 1, :]))
            mu_DV = scipy.hstack((D_mean[:,-1], V_mean[:,-1]))   # consider only last charge state
            DV_point = scipy.hstack((self.explicit_D, self.explicit_V))
            cov_DV = scipy.cov(DV, rowvar=False, aweights=weights)
            L = scipy.linalg.cholesky(cov_DV + 1000 * sys.float_info.epsilon * scipy.eye(*cov_DV.shape), lower=True)
            y = scipy.linalg.solve_triangular(L, DV_point - mu_DV, lower=True)
            M = y.T.dot(y)

            out += [M,]
        
        if plot:
            from matplotlib import gridspec
            gss= gridspec.GridSpec(2,1,height_ratios=[1,1])
            a_D=plt.subplot(gss[0])
            a_V=plt.subplot(gss[1],sharex = a_D)

            a_D.fill_between(self.roa_grid_DV, D_mean-D_std, D_mean+D_std, alpha=0.8)
            a_V.fill_between(self.roa_grid_DV, V_mean-V_std, V_mean+V_std, alpha=0.8)
            a_D.fill_between(self.roa_grid_DV, D_mean-2*D_std, D_mean+2*D_std, alpha=0.4)
            a_V.fill_between(self.roa_grid_DV, V_mean-2*V_std, V_mean+2*V_std, alpha=0.4)

            # Add vertical dashed lines at all positions where a knot is located
            for pnt in self.knotgrid_D:
                a_D.plot([pnt,pnt],[a_D.get_ylim()[0],a_D.get_ylim()[1]], color='red', linestyle='dashed')
            for pnt in self.knotgrid_V:
                a_V.plot([pnt,pnt],[a_V.get_ylim()[0],a_V.get_ylim()[1]], color='red', linestyle='dashed')

            a_D.set_ylabel('$D$ [m$^2$/s]',fontsize=18)
            plt.setp(a_D.get_xticklabels(),visible=False)
            #remove last tick label for the second subplot
            yticks=a_V.yaxis.get_major_ticks()
            yticks[-1].label1.set_visible(False)
            a_V.set_ylabel('$V$ [m/s]',fontsize=18)
            a_V.set_xlabel('$r/a$',fontsize=18)
            a_D.grid(); a_V.grid()
            a_D.set_xlim(self.roa_grid_DV[0], self.roa_grid_DV[-1])
            a_D.set_ylim(bottom=0)
            a_D.tick_params(axis='y',which='major',labelsize=16)
            a_V.tick_params(axis='both',which='major',labelsize=16)

            #remove vertical gap between subplots
            plt.subplots_adjust(hspace=.0)
    
        
        return tuple(out)
    


    def compute_IC(self, sampler, burn, chain_mask=None, debug_plots=False, lp=None, ll=None):
        """Compute the DIC and AIC information criteria.
        
        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler`
            The sampler to compute the criteria for.
        burn : int
            The number of samples to burn before computing the criteria.
        chain_mask : array, optional
            The chains to include in the computation.
        debug_plots : bool, optional
            If True, plots will be made of the conditions at the posterior mean
            and a histogram of the log-likelihood will be drawn.
        lp : array, optional
            The log-posterior. Only to be passed if `sampler` is an array.
        ll : array, optional
            The log-likelihood. Only to be passed if `sampler` is an array.
        """
        # Compute the DIC:
        if chain_mask is None:
            if isinstance(sampler, emcee.EnsembleSampler):
                chain_mask = np.ones(sampler.chain.shape[0], dtype=bool)
            elif hasattr(emcee, 'PTSampler') and isinstance(sampler, emcee.PTSampler):
                chain_mask = np.ones(sampler.chain.shape[1], dtype=bool)
            elif isinstance(sampler, scipy.ndarray):
                if sampler.ndim == 4:
                    chain_mask = np.ones(sampler.shape[1], dtype=bool)
                else:
                    chain_mask = np.ones(sampler.shape[0], dtype=bool)
            else:
                raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        if isinstance(sampler, emcee.EnsembleSampler):
            flat_trace = sampler.chain[chain_mask, burn:, :]
        elif hasattr(emcee, 'PTSampler') and isinstance(sampler, emcee.PTSampler):
            flat_trace = sampler.chain[0, chain_mask, burn:, :]
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                flat_trace = sampler[0, chain_mask, burn:, :]
            else:
                flat_trace = sampler[chain_mask, burn:, :]
        else:
            raise ValueError("Unknown sampler class: %s" % (type(sampler),))
        flat_trace = flat_trace.reshape((-1, flat_trace.shape[2]))
        
        theta_hat = flat_trace.mean(axis=0)
        
        lp_theta_hat, blob = self.compute_ln_prob(theta_hat, debug_plots=debug_plots, return_blob=True)
        ll_theta_hat = blob[0]
        
        if isinstance(sampler, emcee.EnsembleSampler):
            blobs = np.asarray(sampler.blobs, dtype=object)
            ll = np.asarray(blobs[burn:, chain_mask, 0], dtype=float)
        elif hasattr(emcee, 'PTSampler') and isinstance(sampler, emcee.PTSampler):
            ll = sampler.lnlikelihood[0, chain_mask, burn:]
        elif isinstance(sampler, scipy.ndarray):
            if sampler.ndim == 4:
                ll = ll[0, chain_mask, burn:]
            else:
                ll = ll[chain_mask, burn:]
        E_ll = ll.mean()
        
        if debug_plots:
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            a.hist(ll.ravel(), 50)
            a.axvline(ll_theta_hat, label=r'$LL(\hat{\theta})$', color='r', lw=3)
            a.axvline(E_ll, label=r'$E[LL]$', color='g', lw=3)
            a.legend(loc='best')
            a.set_xlabel('LL')
        
        pD_1 = 2 * (ll_theta_hat - E_ll)
        pD_2 = 2 * ll.var(ddof=1)
        
        DIC_1 = -2 * ll_theta_hat + 2 * pD_1
        DIC_2 = -2 * ll_theta_hat + 2 * pD_2
        
        # Compute AIC:
        try:
            p = sampler.dim
        except AttributeError:
            p = sampler.shape[-1]
        ll_max = ll.max()
        AIC = 2 * p - 2 * ll_max
        
        # Compute WAIC?
        
        # Compute log-evidence:
        try:
            lev, e_lev = sampler.thermodynamic_integration_log_evidence(fburnin=burn / sampler.chain.shape[2])
        except:
            lev = None
            e_lev = None
            warnings.warn("Thermodynamic integration failed!", RuntimeWarning)
        
        out = {
            'DIC_1': DIC_1,
            'DIC_2': DIC_2,
            'pD_1': pD_1,
            'pD_2': pD_2,
            'AIC': AIC,
            'p': p,
            'theta_hat': theta_hat,
            'log_evidence': lev,
            'err_log_evidence': e_lev
        }
        return out





    def compute_centrifugal_asym(self, rhop_grid):
        
        ip = omfit_gapy.OMFITinputprofiles(f'{self.shot}/input.profiles.{self.shot}')
        omega0 = ip['omega0']
        z_eff = ip['z_eff']
        rhop_ip = np.sqrt((ip['polflux'])/ip['polflux'][-1])
        omega = interp1d(rhop_ip[2:], omega0[2:], bounds_error=False, fill_value='extrapolate')(rhop_grid)
        Zeff =  interp1d(rhop_ip[1:-1], z_eff[1:-1], bounds_error=False, fill_value='extrapolate')(rhop_grid)

        # find average Z at every radius from fractional abundances in ionization equilibrium (approx)
        atom_data = aurora.get_atom_data(self.asim.namelist['imp'],['scd','acd','ccd'])

        ne = interp2d(self.rhop, self.time, self.ne_cm3)(rhop_grid, self.time_spec)
        Te =  interp2d(self.rhop, self.time, self.Te_eV)(rhop_grid, self.time_spec) 

        # if len(rhop_grid)==len(self.rhop):
        #     # use ne,Te on complete Aurora grid
        #     ne = self.ne_cm3
        #     Te = self.Te_eV
        # else:
        #     # use ne,Te on reduced grids used for XICS spectral modeling
        #     ne = self.ne_cm3_spec
        #     Te = self.Te_eV_spec

        if self.include_neutrals_cxr:
            n0_by_ne = interp1d(self.rhop_n0, self.n0_cm3)(rhop_grid)/(ne[-1,:])
        else:
            n0_by_ne = 1e-5 # not used if include_cx=False in get_frac_abundances, but must be float

        # get fractional abundances from Aurora
        logTe, fz = aurora.get_frac_abundances(atom_data, ne[-1,:], Te[-1,:], 
                                                      n0_by_ne=n0_by_ne if self.include_neutrals_cxr else 0.0,
                                                      rho=rhop_grid, plot=False)

        Z_ave_vec = np.mean(self.asim.Z_imp * fz * np.arange(self.asim.Z_imp+1)[None,:],axis=1)  # Z_ave

        # centrifugal asymmetry exponential factor:
        #self.CF_lambda = self.asim.centrifugal_asym(omega, Zeff).mean(0) 
        if len(rhop_grid)==len(self.rhop):
            Rlfs = self.asim.Rlfs
        else:
            Rlfs = np.interp(rhop_grid, self.asim.rhop_grid, self.asim.Rlfs)
        
        CF_lambda = aurora.synth_diags.centrifugal_asym(rhop_grid, Rlfs, omega, Zeff, self.asim.A_imp, Z_ave_vec, 
                                                Te, Te, main_ion_A=self.asim.main_ion_A,   # Ti=Te
                                                plot=False, nz=None, geqdsk=None).mean(0)

        return CF_lambda


    def compute_diag_rad_weights(self, beam, rhop_grid, 
                                 apply_CF_corr=True, use_TRIPPy_inv=False, ds = 1e-5, #5e-5, 
                                 tok=None):
        '''Compute diagnostic weights as a function of radial coordinate.

        Args:
            beam : TRIPPy beam instance
            apply_CF_corr : if True, apply centrifugal (CF) asymmetry correction to diagnostic weights.
            use_TRIPPy_inv : if True, use TRIPPy to calculate diagnostic radial weights. This currently does not allow
                inclusion of CF asymmetry corrections. 
            ds : spatial separation [m] for beam path discretization.
            tok : instance of the TRIPPy.plasma.Tokamak class, only needed if use_TRIPPy_inv=True
        '''
        if not use_TRIPPy_inv:
            # get R along ray path
            temp = beam(scipy.mgrid[beam.norm.s[-2]:beam.norm.s[-1]:ds])
            R_path = temp.r0()
            Z_path = temp.x2()
            Psi_path = self.efit_tree.rz2psinorm(R_path, Z_path,(self.time_1 + self.time_2) / 2.0)

            #if not hasattr(self, 'CF_lambda'):
            CF_lambda = self.compute_centrifugal_asym(rhop_grid)

            p0 = np.array([temp.x0()[0], temp.x1()[0], temp.x2()[0]]) # outer-most point

            # length of line integration  (cartesian coordinates)
            pathL_m = np.linalg.norm(np.array([temp.x0(), temp.x1(), temp.x2()]) - p0[:,None],axis=0)

            # compute weights by summing over beam/ray path
            weights = aurora.synth_diags.line_int_weights(R_path,Z_path,np.sqrt(Psi_path), pathL_m, 
                                                          self.asim.Raxis_cm/100, rhop_out = rhop_grid, 
                                                          CF_lam=CF_lambda if apply_CF_corr else None)

        else:
            if tok is None:
                tokamak = TRIPPy.plasma.Tokamak(self.efit_tree)

            weights = TRIPPy.invert.fluxFourierSens(
                beam,
                self.efit_tree.rz2psinorm,
                tok.center,
                (self.time_1 + self.time_2) / 2.0,
                rhop_grid**2.0,
                ds=ds
            )[0]

        return weights

    def get_lineint_rays(self):
        '''Obtain description of line integration paths for each diagnostic using TRIPPy.
        '''
        tokamak = TRIPPy.plasma.Tokamak(self.efit_tree)
        
        xics_rays = []
        vuv_rays = []
        sxr_rays = []
        
        # Handle HiReX-SR:
        if self.signals[0] is not None:
            print("Obtaining Hirex-Sr beam paths info....")
            xics_rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in self.signals[0].pos]
            
        # Handle VUV diagnostics:
        if self.signals[1] is not None:
            print("Obtaining VUV beam paths info....")
            vuv_rays.append( TRIPPy.beam.pos2Ray(XEUS_POS, tokamak) )
            vuv_rays.append( TRIPPy.beam.pos2Ray(LOWEUS_POS, tokamak) )
            
        # Handle HiReX-SR argon:
        # ar_rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in self.ar_signals.pos]

        # Get xtomo beams:
        if self.signals[2] is not None:
            print("Obtaining XTOMO beam paths info....")
            sxr_rays.append( TRIPPy.XTOMO.XTOMO1beam(tokamak) )
            sxr_rays.append( TRIPPy.XTOMO.XTOMO3beam(tokamak) )
            sxr_rays.append( TRIPPy.XTOMO.XTOMO5beam(tokamak) )

        return xics_rays, vuv_rays, sxr_rays

    def compute_view_data(self,  debug_plots=False, contour_axis=None):
        """Compute the quadrature weights to line-integrate the emission profiles.
        
        Puts the results in the corresponding entries in signal.
        
        Parameters
        ----------
        debug_plots : bool, optional
            If True, plots of the weights and chords will be produced. Default
            is False (do not make plots).
        contour_axis : axis instance, optional
            If provided, plot the chords on this axis. All systems will be put
            on the same axis! Default is to produce a new figure for each system.
 
        """
        # obtain description of line integration paths
        xics_rays, vuv_rays, sxr_rays = self.get_lineint_rays()

        # fluxFourierSens returns shape (n_time, n_chord, n_quad), we just have one time element.
        # Handle HiReX-SR:
        print("Start getting diagnostic inversion weights: ")
        if self.signals[0] is not None:

            self.signals[0].weights = np.zeros((len(xics_rays), len(self.rhop_spec)))
            for bb,beam in enumerate(xics_rays):
                self.signals[0].weights[bb,:] = self.compute_diag_rad_weights(beam, self.rhop_spec,
                                                                              apply_CF_corr=self.apply_CF_corr)
            self.signals[0].rhop = self.rhop_spec
            print("Saved Hirex-Sr w-line weights!")

        # Handle XEUS and LoWEUS:
        if self.signals[1] is not None:

            XEUS_weights = self.compute_diag_rad_weights(vuv_rays[0], self.rhop, 
                                                         apply_CF_corr=self.apply_CF_corr)
            LoWEUS_weights = self.compute_diag_rad_weights(vuv_rays[1], self.rhop, 
                                                           apply_CF_corr=self.apply_CF_corr)
        
            # make sure that single-chord diagnostics have 2D arrays (second index identified diagnostic)
            self.signals[1].weights = np.zeros( (np.atleast_2d(self.signals[1].y).shape[1], len(self.rhop)) )

            for i, n in enumerate(self.signals[1].name):
                if n == 'XEUS':
                    self.signals[1].weights[i, :] = XEUS_weights
                    print("Saved XEUS weights for line group "+str(i)+"!")
                else:
                    self.signals[1].weights[i, :] = LoWEUS_weights
                    print("Saved LoWEUS weights for line group "+str(i)+"!")

            self.signals[1].rhop = self.rhop


        # Handle XTOMO:
        if self.signals[2] is not None:
            xtomo_weights = self.get_xtomo_weights(sxr_rays)
            
            # group all XTOMO weights
            self.signals[2].weights = np.zeros( (self.signals[2].y.shape[1], len(self.rhop)) )

            for i, b in enumerate(self.signals[2].blocks):
                self.signals[2].weights[i, :] = xtomo_weights[b][self.signals[2].weight_idxs[i], :]
            self.signals[2].rhop = self.rhop

            print('Setting XTOMO weights to 0 for rhop>1.02')
            self.signals[2].weights[:,self.signals[2].rhop>1.02] = 0.0

            print("Saved XTOMO weights!")

        # ----------------------------------
        if debug_plots:
            self.plot_line_integration_weights(contour_axis=None, xtomo_weights=xtomo_weights if self.signals[2] else None)

        print("Done finding view data!")

    def get_xtomo_weights(self, sxr_rays):
        '''Cache weights for XTOMO (SXR) chords.
        '''
        xtomo_weights = {}

        xtomo_weights[1] = np.zeros((len(sxr_rays[0]), len(self.rhop)))
        for bb,beam in enumerate(sxr_rays[0]):
             xtomo_weights[1][bb,:] = self.compute_diag_rad_weights(beam, self.rhop, 
                                                                    apply_CF_corr=self.apply_CF_corr)

        xtomo_weights[3] = np.zeros((len(sxr_rays[1]), len(self.rhop)))
        for bb,beam in enumerate(sxr_rays[1]):
             xtomo_weights[3][bb,:] = self.compute_diag_rad_weights(beam, self.rhop, 
                                                                    apply_CF_corr=self.apply_CF_corr)

        xtomo_weights[5] = np.zeros((len(sxr_rays[2]), len(self.rhop)))
        for bb,beam in enumerate(sxr_rays[2]):
             xtomo_weights[5][bb,:] = self.compute_diag_rad_weights(beam, self.rhop, 
                                                                    apply_CF_corr=self.apply_CF_corr)

        return xtomo_weights

    def plot_line_integration_weights(self, contour_axis=None, xtomo_weights=None):
        '''Get a series of figures describing line integration for XICS, VUV and XTOMO diagnostics.

        Parameters
        -----------------
        contour_axis : axis instance, optional
            If provided, plot the chords on this axis. All systems will be put
            on the same axis! Default is to produce a new figure for each system.
        xtomo_weights : dict, optional
            Weights for all SXR chords, including those not selected from experimental data.
            If self.signals[2] is not None and this argument is not passed, xtomo_weights are computed internally.
            
        '''
        
        # obtain description of line integration paths
        xics_rays, vuv_rays, sxr_rays = self.get_lineint_rays()

        # get time index of magnetic reconstruction of interest
        i_flux = profiletools.get_nearest_idx(
            (self.time_1 + self.time_2) / 2.0,
            self.efit_tree.getTimeBase()
        )

        ls_cycle = aurora.get_ls_cycle()

        if self.signals[0] is not None:
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            iii=0
            for w in self.signals[0].weights:
                a.plot(self.rhop_spec, w, next(ls_cycle), label=fr'chord \#{iii+1:d}')
                iii+=1
            a.set_xlabel(r"$\rho_p$")
            a.set_ylabel("quadrature weights")
            a.set_title("HiReX-SR - Ca")
            a.legend(fontsize=10).set_draggable(True) #labels=[str(i) for i in np.arange(0, self.signals[0].y.shape[1])])

        if self.signals[1] is not None:
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in self.signals[1].weights:
                a.plot(self.rhop, w, next(ls_cycle))
            a.set_xlabel(r'$\rho_\psi$')
            a.set_ylabel('quadrature weights')
            a.set_title('VUV')

        vuv_cycle = itertools.cycle(['b', 'g'])


        tokamak = TRIPPy.plasma.Tokamak(self.efit_tree)
        from TRIPPy.plot.pyplot import plotTokamak, plotLine
        if contour_axis is None:
            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            # Only plot the tokamak if an axis was not provided:
            plotTokamak(tokamak)
        else:
            a = contour_axis
            plt.sca(a)

        # Plot VUV in different color:
        if self.signals[0] is not None:
            for r in xics_rays:
                plotLine(r, pargs='r')#ls_cycle.next())
        if self.signals[1] is not None:
            plotLine(vuv_rays[0], pargs=next(vuv_cycle), lw=3)
            plotLine(vuv_rays[1], pargs=next(vuv_cycle), lw=3)

        if contour_axis is None:
            a.contour(
                self.efit_tree.getRGrid(),
                self.efit_tree.getZGrid(),
                self.efit_tree.getFluxGrid()[i_flux, :, :],
                80
            )
        #a.set_title("HiReX-SR, VUV")


        if self.signals[2] is not None:
            # And for XTOMO 1:
            ls_cycle = aurora.get_ls_cycle()

            if xtomo_weights is None:
                # get weights here
                xtomo_weights = self.get_xtomo_weights(sxr_rays)

            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in xtomo_weights[1]:
                a.plot(self.rhop, w, next(ls_cycle))
            a.set_xlabel(r"$\rho_p$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 1")

            ls_cycle = aurora.get_ls_cycle()

            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in sxr_rays[0]:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 1")

            # And for XTOMO 3:
            ls_cycle = aurora.get_ls_cycle()

            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in xtomo_weights[3]:
                a.plot(self.rhop, w, next(ls_cycle))
            a.set_xlabel(r"$\rho_p$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 3")

            ls_cycle = aurora.get_ls_cycle()

            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in sxr_rays[1]:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 3")

            # And for XTOMO 5:
            ls_cycle = aurora.get_ls_cycle()

            f = plt.figure()
            a = f.add_subplot(1, 1, 1)
            for w in xtomo_weights[5]:
                a.plot(self.rhop, w, next(ls_cycle))
            a.set_xlabel(r"$\rho_p$")
            a.set_ylabel("quadrature weights")
            a.set_title("XTOMO 5")

            ls_cycle = aurora.get_ls_cycle()

            if contour_axis is None:
                f = plt.figure()
                a = f.add_subplot(1, 1, 1)
                # Only plot the tokamak if an axis was not provided:
                plotTokamak(tokamak)
            else:
                a = contour_axis
                plt.sca(a)
            for r in sxr_rays[2]:
                plotLine(r, pargs='r')#ls_cycle.next())
            if contour_axis is None:
                a.contour(
                    self.efit_tree.getRGrid(),
                    self.efit_tree.getZGrid(),
                    self.efit_tree.getFluxGrid()[i_flux, :, :],
                    50
                )
            a.set_title("XTOMO 5")

    


#####################################################################
#
#
#
#
#
###############################################################
#
#

class _ComputeLnProbWrapper:
    """Wrapper to support parallel execution of aurora runs.
    
    This is needed since instance methods are not pickleable.
    
    Parameters
    ----------
    run : :py:class:`Run` instance
        The :py:class:`Run` to wrap.
    make_dir : bool, optional
        If True, a new aurora directory is acquired and released for each call.
        Default is False (run in current directory).
    for_min : bool, optional
        If True, the function is wrapped in the way it needs to be for a
        minimization: only -1 times the log-posterior is returned, independent
        of the value of `return_blob`.
    denormalize : bool, optional
        If True, a normalization from [lb, ub] to [0, 1] is removed. Default is
        False (don't adjust parameters).
    """
    def __init__(self, run, make_dir=False, for_min=False, denormalize=False):
        self.run = run
        self.make_dir = make_dir
        self.for_min = for_min
        self.denormalize = denormalize
    
    def __call__(self, params, **kwargs):
        if self.denormalize:
            bounds = np.asarray(self.run.lnprior().bounds[:], dtype=float)
            lb = bounds[:, 0]
            ub = bounds[:, 1]
            params = [x * (u - l) + l for x, u, l in zip(params, ub, lb)]
        try:
            if self.make_dir:
                acquire_working_dir()
            out = self.run.DV2ln_prob(
                params=params,
                sign=(-1.0 if self.for_min else 1.0),
                **kwargs
            )
        except:
            warnings.warn(
                "Unhandled exception. Error is: %s: %s. "
                "Params are: %s" % (
                    sys.exc_info()[0],
                    sys.exc_info()[1],
                    params
                )
            )
            if self.for_min:
                out = scipy.inf
            else:
                # if kwargs.get('return_blob', False):
                #     if kwargs.get('light_blob', False):
                #         out = (-scipy.inf)
                #     else:
                #         out = (-scipy.inf, (-scipy.inf, None, None, None, ''))
                # else:
                out = -scipy.inf
        finally:
            if self.make_dir:
                release_working_dir()
        return out

class _UGradEval(object):
    """Wrapper object for evaluating :py:meth:`Run.u2ln_prob` in parallel.
    """
    def __init__(self, run, sign, kwargs):
        self.run = run
        self.sign = sign
        self.kwargs = kwargs
    
    def __call__(self, p):
        return self.run.u2ln_prob(p, sign=self.sign, **self.kwargs)


class _CSDenResult:
    """Helper object to hold the results of :py:class:`_ComputeCSDenEval`.
    """
    def __init__(self, DV):
        self.DV = DV

class _ComputeCSDenEval:
    """Wrapper class to allow parallel evaluation of charge state density profiles.
    
    Also computes the following:
    
    * Time at which each charge state peaks at each location. This will be
      of shape (`n_cs`, `n_space`).
    * The peak value of each charge state at each location. This will be of
      shape (`n_cs`, `n_space`).
    
    * Time at which each charge state reaches its highest local value
      (across all spatial points). This will be of shape (`n_cs`,).
    * The peak value of each charge state across all spatial points. This
      will be of shape (`n_cs`,).
    * The spatial point at which each charge state across all spatial points
      reaches its peak value. This will be of shape (`n_cs`,).
    
    * Time at which the total impurity density peaks at each location. This
      will be of shape (`n_space`,).
    * The peak value of the total impurity density at each location. This
      will be of shape (`n_space`,).
    
    * The time at which the total impurity density peaks (across all spatial
      points). This will be a single float.
    * The peak value of the total impurity density across all spatial
      points. This will be a single float.
    * The spatial point at which the total impurity density across all
      spatial points reaches its peak value. This will be a single float.
    
    * The time at which the total impurity content peaks. This will be a
      single float.
    * The peak number of impurity atoms in the plasma. This will be a single
      float.
    
    * The confinement time for each charge state and each location. This
      will be of shape (`n_cs`, `n_space`).
    * The confinement time for the total impurity density at each location.
      This will be of shape (`n_space`,).
    * The confinement time for the total impurity content. This will be a
      single float.
    """
    def __init__(self, run):
        self.run = run
    
    def __call__(self, DV):

        #cs_den = self.run.DV2cs_den(DV)
        
        res = _CSDenResult(DV)
        
        # For each charge state and location:
        i_peak_local = cs_den.argmax(axis=0)
        res.t_cs_den_peak_local = time[i_peak_local]
        res.cs_den_peak_local = cs_den.max(axis=0)
        
        # For each charge state across all locations:
        i_peak_global = res.cs_den_peak_local.argmax(axis=1)
        res.t_cs_den_peak_global = res.t_cs_den_peak_local[list(np.arange(cs_den.shape[1])), i_peak_global]
        res.cs_den_peak_global = res.cs_den_peak_local.max(axis=1)
        res.rhop_cs_den_peak_global = rhop[i_peak_global]
        
        # For total impurity density at each location:
        n = cs_den.sum(axis=1) # shape is (`n_time`, `n_space`)
        i_n_peak_local = n.argmax(axis=0)
        res.t_n_peak_local = time[i_n_peak_local]
        res.n_peak_local = n.max(axis=0)
        
        # For total impurity density across all locations:
        i_n_peak_global = res.n_peak_local.argmax()
        res.t_n_peak_global = res.t_n_peak_local[i_n_peak_global]
        res.n_peak_global = res.n_peak_local[i_n_peak_global]
        res.rhop_n_peak_global = rhop[i_n_peak_global]
        
        # For total impurity content inside the LCFS:
        volnorm_grid = self.run.efit_tree.psinorm2volnorm(
            rhop**2.0,
            (self.run.time_1 + self.run.time_2) / 2.0
        )
        V = self.run.efit_tree.psinorm2v(1.0, (self.run.time_1 + self.run.time_2) / 2.0)
        mask = ~scipy.isnan(volnorm_grid)
        volnorm_grid = volnorm_grid[mask]
        nn = n[:, mask]
        # Use the trapezoid rule:
        N = V * 0.5 * ((volnorm_grid[1:] - volnorm_grid[:-1]) * (nn[:, 1:] + nn[:, :-1])).sum(axis=1)
        i_N_peak = N.argmax()
        res.t_N_peak = time[i_N_peak]
        res.N_peak = N[i_N_peak]
        
        # # Confinement time for each charge state and each location:
        # res.tau_cs_den_local = np.zeros(cs_den.shape[1:3])
        # for s_idx in np.arange(0, cs_den.shape[2]):
        #     for cs_idx in np.arange(0, cs_den.shape[1]):
        #         t_mask = (self.run.truth_data.time > res.t_cs_den_peak_local[cs_idx, s_idx] + 0.01) & (cs_den[:, cs_idx, s_idx] > 0.0) & (~scipy.isinf(cs_den[:, cs_idx, s_idx])) & (~scipy.isnan(cs_den[:, cs_idx, s_idx]))
        #         if t_mask.sum() < 2:
        #             res.tau_cs_den_local[cs_idx, s_idx] = 0.0
        #         else:
        #             X = scipy.hstack((np.ones((t_mask.sum(), 1)), scipy.atleast_2d(self.run.truth_data.time[t_mask]).T))
        #             theta, dum1, dum2, dum3 = scipy.linalg.lstsq(X.T.dot(X), X.T.dot(np.log(cs_den[t_mask, cs_idx, s_idx])))
        #             res.tau_cs_den_local[cs_idx, s_idx] = -1.0 / theta[1]
        #
        # # Confinement time for total impurity density at each location:
        # res.tau_n_local = np.zeros(cs_den.shape[2])
        # for s_idx in np.arange(0, n.shape[-1]):
        #     t_mask = (self.run.truth_data.time > res.t_n_peak_local[s_idx] + 0.01) & (n[:, s_idx] > 0.0) & (~scipy.isinf(n[:, s_idx])) & (~scipy.isnan(n[:, s_idx]))
        #     if t_mask.sum() < 2:
        #         res.tau_n_local[s_idx] = 0.0
        #     else:
        #         X = scipy.hstack((np.ones((t_mask.sum(), 1)), scipy.atleast_2d(self.run.truth_data.time[t_mask]).T))
        #         theta, dum1, dum2, dum3 = scipy.linalg.lstsq(X.T.dot(X), X.T.dot(np.log(n[t_mask, s_idx])))
        #         res.tau_n_local[s_idx] = -1.0 / theta[1]
        
        # Confinement time of total impurity content and shape factor:
        t_mask = (time > res.t_N_peak + 0.01) & (N > 0.0) & (~scipy.isinf(N)) & (~scipy.isnan(N))
        if t_mask.sum() < 2:
            res.tau_N = 0.0
            res.n075n0 = scipy.median(n[:, 62] / n[:, 0])
            res.prof = scipy.nanmedian(n / n[:, 0][:, None], axis=0)
        else:
            X = scipy.hstack((np.ones((t_mask.sum(), 1)), scipy.atleast_2d(time[t_mask]).T))
            theta, dum1, dum2, dum3 = scipy.linalg.lstsq(X.T.dot(X), X.T.dot(np.log(N[t_mask])))
            res.tau_N = -1.0 / theta[1]
            
            first_t_idx = scipy.where(t_mask)[0][0]
            # Take the median just in case I didn't wait until far enough after the peak:
            res.n075n0 = scipy.median(n[first_t_idx:, 62] / n[first_t_idx:, 0])
            res.prof = scipy.nanmedian(n[first_t_idx:, :] / n[first_t_idx:, 0][:, None], axis=0)
        
        return res

####################################################################
#
#
#
#
#
#
###################################################################

################################################################
class _ComputeProfileWrapper:
    """Wrapper to enable evaluation of D, V profiles in parallel.
    """
    def __init__(self, run, time_idx=-1, cs_idx=-1):
        self.run = run
        self.time_idx = time_idx
        self.cs_idx = cs_idx
    
    def __call__(self, params):
        D_z,V_z,times_DV = self.run.eval_DV(params=params, imp='Ca')  # fixed to "Ca"
        
        # select time slice and charge state
        #return D_z[:,self.time_idx,self.cs_idx], V_z[:,self.time_idx,self.cs_idx]
        return D_z[:,self.time_idx,:], V_z[:,self.time_idx,:]



#############################################################
def source_function(t, t_start, t_rise, n_rise, t_fall, n_fall, t_cluster=0.0, h_cluster=0.0):
    """Defines a model form to approximate the shape of the source function.
    
    Consists of an exponential rise, followed by an exponential decay and,
    optionally, a constant tail to approximate clusters.
    
    The cluster period is optional, so you can either treat this as a
    5-parameter function or a 7-parameter function.
    
    The function is set to have a peak value of 1.0.
    
    Parameters
    ----------
    t : array of float
        The time values to evaluate the source at.
    t_start : float
        The time the source starts at.
    t_rise : float
        The length of the rise portion.
    n_rise : float
        The number of e-folding times to put in the rise portion.
    t_fall : float
        The length of the fall portion.
    n_fall : float
        The number of e-folding times to put in the fall portion.
    t_cluster : float, optional
        The length of the constant period. Default is 0.0.
    h_cluster : float, optional
        The height of the constant period. Default is 0.0.
    """
    s = scipy.atleast_1d(np.zeros_like(t))
    tau_rise = t_rise / n_rise
    tau_fall = t_fall / n_fall
    
    rise_idx = (t >= t_start) & (t < t_start + t_rise)
    s[rise_idx] = 1.0 - scipy.exp(-(t[rise_idx] - t_start) / tau_rise)
    
    fall_idx = (t >= t_start + t_rise) & (t < t_start + t_rise + t_fall)
    s[fall_idx] = scipy.exp(-(t[fall_idx] - t_start - t_rise) / tau_fall)
    
    s[(t >= t_start + t_rise + t_fall) & (t < t_start + t_rise + t_fall + t_cluster)] = h_cluster
    
    return s

#################################################################




def compute_emiss(log10pec_dict, cw, hw, ne_cm3, Te_eV, nZ_ioniz, nZ_exc, nZ_rec, no_ne=False):
    """Compute the emission summed over all lines in a given window.
    
    Parameters
    ------------------
    log10pec_dict : dictionary
        The log-10 photon emission coefficient dictionary as returned by the Aurora :py:func:`read_adf15` 
        for the desired charge state.
    cw : array of float
        The center wavelengths of the bins to use, in angstroms.
    hw : array of float
        The half-widths of the bins to use, in angstroms.
    ne_cm3 : array of float  (time,space)
        The electron density on the grid, in cm^3.
    Te_eV : array of float  (time,space)
        The electron temperature on the grid, in eV.
    nZ_ioniz : array of float  (time,space)
        The density of the charge state that produces the atomic line of interest via ionization. 
        If left to None, this is not considered; otherwise, provide this array on Aurora grids, in cm^3.
    nZ_exc : array of float  (time,space)
        The density of the charge state that produces the atomic line of interest via excitation. 
        This must be given on the Aurora grids, in cm^3.
    nZ_rec : array of float  (time,space)
        The density of the charge state that produces the atomic line of interest via recombination
        (radiative or dielectronic).
        If left to None, this is not considered; otherwise, provide this array on Aurora grids, in cm^3.
    no_ne : bool, optional
        If True, the PEC is taken to not depend on density. Default is False.
    """
    lb = cw - hw
    ub = cw + hw
    wl = np.asarray(list(log10pec_dict.keys()))
    included = wl[(wl >= lb) & (wl <= ub)]
    emiss = np.zeros_like(ne_cm3)
    
    #lam_j_Ca_exc = 3.2090
    #lam_j_Ca_drsat = 3.2102  ####
    #lam_k_Ca_exc = 3.2058
    #lam_k_Ca_drsat = 3.2064

    if len(included)==1 and included[0]==3.2064:  # lam_k_Ca_drsat
        # include k and j drsat and exc lines individually, all from PEC[17]
        included = [3.2064, 3.2058, 3.2102, 3.2090]
        #included = [lam_k_Ca_drsat, lam_k_Ca_exc, lam_j_Ca_drsat, lam_j_Ca_exc]

    for lam in included:

        if 'ioniz' in log10pec_dict[lam]:
            # add ionization component to emissivity
            if no_ne:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_ioniz * 10**log10pec_dict[lam]['ioniz'](np.log10(Te_eV))
            else:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_ioniz * 10**log10pec_dict[lam]['ioniz'].ev(
                    np.log10(ne_cm3),
                    np.log10(Te_eV)
                )
        # excitation component
        if 'excit' in log10pec_dict[lam]:
            if no_ne:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_exc * 10**log10pec_dict[lam]['excit'](np.log10(Te_eV))
            else:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_exc * 10**log10pec_dict[lam]['excit'].ev(
                    np.log10(ne_cm3),
                    np.log10(Te_eV)
                )
            
        if 'recom' in log10pec_dict[lam]:
            # add radiative recombination component to emissivity
            if no_ne:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_rec * 10**log10pec_dict[lam]['recom'](np.log10(Te_eV))
            else:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_rec * 10**log10pec_dict[lam]['recom'].ev(
                    np.log10(ne_cm3),
                    np.log10(Te_eV)
                )

        if 'drsat' in log10pec_dict[lam]:
            # add dielectronic recombination component to emissivity
            if no_ne:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_rec * 10**log10pec_dict[lam]['drsat'](np.log10(Te_eV))
            else:
                emiss += constants.h*constants.c/(lam*1e-10) * ne_cm3 * nZ_rec * 10**log10pec_dict[lam]['drsat'].ev(
                    np.log10(ne_cm3),
                    np.log10(Te_eV)
                )
                
    # Sometimes there are interpolation issues with the PECs:
    emiss[emiss < 0] = 0.0
    
    return emiss




def lambda2eV_conv(lam):
    '''lambda in A units'''
    return 6.64*10**(-34)*3*10**8/(lam*10**(-10)*1.6*10**(-19))




def get_robust_weighted_stats(samples, weights=None, quantiles=None):
    '''
    Obtain robust weighted statistics for a set of samples and respective weights.
    This function gives a generalization of the meanw and stdw functions in profiletools.

    Parameters:
    ------------------------
    samples : array-like
        The vector to find robust weighted statistics for. Statistics will be taken along axis=0.
    weights : array-like, optional
        Weights to use for obtain robust statistics. If left to None, these are set to be
        uniform, thus giving a non-weighted set of statistics.
    quantiles : list or array-like, optional
        Quantiles to compute. If left to None, a standard set is used.
    '''

    if weights is None:
        # untested, but should work (even non-normalized should work, I think...)
        weights = np.ones_like(samples[:,1])/len(samples[:,1])

    if quantiles is None:
        quantiles = [.01, .1, .25, .5, .75, .9, .99]

    nrho = samples.shape[1]
    sind = np.argsort(samples, axis=0)

    sorted_samples = samples[sind, np.arange(nrho)].T
    sorted_weights = weights[sind].T
    sorted_cumweights = np.cumsum(sorted_weights,axis=1)

    qsamples = np.zeros((len(quantiles), nrho))

    for ir in range(nrho):
        qsamples[:,ir] = np.interp(quantiles,sorted_cumweights[ir], sorted_samples[ir] )

    return qsamples




# ===============

def set_xics_normalization(signals, bsfc_lines=['w','z'], rel_calibration=True, indices=[0,1]):
    '''
    Reset normalization and relative calibration of XICS signals, regardless of how they were initially normalized.
    
    `signals` is assumed to be an instance of the `Signals` class used in BITE. 
    
    `bsfc_lines` should give a list/array with the names of each line, e.g. ['w','x','y','z']. 
    
    rel_calibration=True indicates that XICS signals from different lines should be normalized together, 
    i.e. with a single norm shared across all measured lines. If False, each XICS signal is normalized independently. 
    
    indices : list
        indices at which XICS signals are located in the signals object. These are expected to be in the same order as
        bsfc_lines. 

    This should NOT be applied by default because there is important information in the relative calibration of 
    different XICS lines!
    '''
    hsignals =np.asarray( [signals[0]] )
    for ind in indices[1:]:
        hsignals = np.concatenate((hsignals, [ signals[ind] ] ))

    maxx_list = []; std_maxx_list = []
    for ll in np.arange(len(bsfc_lines)):
        # find indices of maxima across every dimension
        idxs = np.unravel_index(np.nanargmax(hsignals[ll].y), hsignals[ll].y.shape)

        maxx_list.append(hsignals[ll].y[idxs[0],idxs[1]])
        std_maxx_list.append(hsignals[ll].std_y[idxs[0],idxs[1]])

    # now that we've collected the maxima and std's for all signals:
    for ll in np.arange(len(bsfc_lines)):

        if rel_calibration:
            # find maximum across all XICS measured lines
            maxx = maxx_list[np.nanargmax(maxx_list)]   
            std_maxx = std_maxx_list[np.nanargmax(maxx_list)]    
        else:
            maxx = maxx_list[ll]
            std_maxx = std_maxx_list[ll]

        hsignals[ll].y_norm = hsignals[ll].y / maxx
        hsignals[ll].std_y_norm =  np.sqrt(
            (hsignals[ll].std_y / maxx) **2.0 + ((hsignals[ll].y / maxx) * (std_maxx / maxx)) **2.0
        )

    # store updated signal
    signals[0] = copy.deepcopy(hsignals[0])
    for ll, ind in enumerate(indices[1:]):
        signals[ind] = copy.deepcopy(hsignals[1+ll])

    return signals




def set_vuv_normalization(signals, rel_calibration=True):
    ''' 
    Reset normalization and relative calibration of VUV signals, regardless of how they were initially normalized. 

    `signals` is assumed to be an instance of the `Signals` class used in BITS. 
    rel_calibration=True indicates that VUV signals from the same spectrometer should be normalized together, 
    i.e. with a single norm shared across all measured lines on a detector. If False, each VUV signal within the detector 
    is normalized independently. 
  
    NB: only signals on the same detector are ever normalized jointly. XEUS and LoWEUS sensitivities are different, 
    so their signals cannot be normalized to the same value. 
    '''
    vuv_sig = signals[1]
    
    if rel_calibration:

        # find absolute across all lines on a single detector
        max_xeus = 0.0; std_max_xeus = 0.0
        max_loweus = 0.0; std_max_loweus=0.0
        for i_group, name in enumerate(vuv_sig.name):
            if name.lower()=='xeus':
                chord_max = np.nanmax(vuv_sig.y[:,i_group])
                if chord_max > max_xeus:
                    max_xeus = chord_max
                    std_max_xeus = vuv_sig.std_y[ np.nanargmax(vuv_sig.y[:,i_group]), i_group]
                if name.lower()=='loweus':
                    chord_max = np.nanmax(vuv_sig.y[:,i_group])
                    if chord_max > max_loweus:
                        max_loweus = chord_max
                        std_max_loweus = vuv_sig.std_y[ np.nanargmax(vuv_sig.y[:,i_group]), i_group]

        # normalize signals on the same detector with common norm
        for i_group, name in enumerate(vuv_sig.name):
            if name.lower()=='xeus':
                maxx = max_xeus
                std_maxx = std_max_xeus
            if name.lower()=='loweus':
                maxx = max_loweus
                std_maxx = std_max_loweus

            vuv_sig.y_norm[:,i_group] = vuv_sig.y[:,i_group]/maxx
            vuv_sig.std_y_norm[:,i_group] = np.sqrt(
                (vuv_sig.std_y[:, i_group] / maxx) **2.0 + ((vuv_sig.y[:, i_group] / maxx) * (std_maxx / maxx)) **2.0
            )

    else:
        
        # find absolute across each VUV signal
        for i_group, name in enumerate(vuv_sig.name):
            maxx = np.nanmax(vuv_sig.y[:,i_group])
            std_maxx = vuv_sig.std_y[ np.nanargmax(vuv_sig.y[:,i_group]), i_group]

            vuv_sig.y_norm[:,i_group] = vuv_sig.y[:,i_group] / maxx
            vuv_sig.std_y_norm[:,i_group] = np.sqrt(
                (vuv_sig.std_y[:, i_group] / maxx) **2.0 + ((vuv_sig.y[:, i_group] / maxx) * (std_maxx / maxx)) **2.0
            )
        
    # modify blocks
    if rel_calibration:
        # all vuv signals from the same detector should be counted as one (they share the same diagnostic weights and rescaling factors)
        vuv_sig.blocks =  np.asarray([0,]* vuv_sig.y.shape[1])    
    else:
        vuv_sig.blocks = np.arange(vuv_sig.y.shape[1])    
        
    # save new signal normalization
    signals[1] = copy.deepcopy(vuv_sig)

    return signals



def get_D_interpolation(method, knotgrid_D, D_params, roa_grid_out):
    '''Collection of interpolation methods for D spline parameters. 
    '''
    if method == 'linterp':
        D = scipy.interpolate.InterpolatedUnivariateSpline(
            knotgrid_D,
            scipy.insert(D_params, 0, D_params[0]),
            k=1
        )(roa_grid_out[roa_grid_out<=np.max(knotgrid_D)])  # only within knotgrid_D

    elif method == 'pchip':
        # this method preserves monotonicity in the interpolation data and does not overshoot if the data is not smooth.
        # The first derivatives are guaranteed to be continuous, but the second derivatives may jump (cit. scipy)

        # use log since D is always sampled >0
        lnD_tmp = np.log(scipy.insert(D_params, 0, D_params[0]))

        # reflect profile and enforce even function
        fitgrid_D = np.r_[-knotgrid_D[1:][::-1],knotgrid_D]
        lnD_tmp_ref = np.r_[ lnD_tmp[1:][::-1],lnD_tmp]

        # Interpolate
        lnD = scipy.interpolate.PchipInterpolator(fitgrid_D,lnD_tmp_ref)(
            roa_grid_out[roa_grid_out<=np.max(knotgrid_D)])  # only within knotgrid_D
        D = np.exp(lnD)

        # ensure grad(D)=0 at center
        D[0] = D[1]

    else:
        raise ValueError(f"Unknown method '{method}'!")

    # set flat D into pedestal using last spline knot value
    D = np.concatenate((D, D[-1]*np.ones_like(roa_grid_out[roa_grid_out>np.max(knotgrid_D)])))
    
    return D


def get_V_interpolation(method, knotgrid_V, _V_params, roa_grid_out):
    '''Collection of interpolation methods for V or V/D spline parameters. 
    Interpolation must be done based on the parameters that are freely sampled, i.e. either V or V/D,
    rather than on transformed parameters, or else spurious structure may be added to a profile.     
    '''
    if method == 'linterp':

        _V = scipy.interpolate.InterpolatedUnivariateSpline(
            knotgrid_V,
            scipy.insert(_V_params, 0, 0.0),
            k=1
        )(roa_grid_out[roa_grid_out<=np.max(knotgrid_V)])

    elif method == 'pchip':
        # this method preserves monotonicity in the interpolation data and does not overshoot if the data is not smooth.
        # The first derivatives are guaranteed to be continuous, but the second derivatives may jump (cit. scipy)

        # V does not need to be +ve, but make it even
        _V_tmp = scipy.insert(_V_params, 0, 0.0)
        fitgrid_V = np.r_[-knotgrid_V[1:][::-1],knotgrid_V]
        _V_tmp_ref = np.r_[ _V_tmp[1:][::-1],_V_tmp]

        # Interpolate
        _V = scipy.interpolate.PchipInterpolator(fitgrid_V,_V_tmp_ref)(
            roa_grid_out[roa_grid_out<=np.max(knotgrid_V)])  # only within knotgrid_V

    else:
        raise ValueError(f"Unknown method '{method}'!")

    # set flat V into pedestal using last spline knot value
    _V = np.concatenate((_V, _V[-1]*np.ones_like(roa_grid_out[roa_grid_out>np.max(knotgrid_V)])))
    
    return  _V


class _OptimizeEval(object):
    """Wrapper class to allow parallel execution of random starts when optimizing the parameters.
    
    Parameters
    ----------
    run : :py:class:`Run`
        The :py:class:`Run` instance to wrap.
    thresh : float, optional
        If True, a test run of the starting 
    """
    def __init__(self, run, thresh=None):
        self.run = run
        self.thresh = thresh
    
    def __call__(self, params):
        """Run the optimizer starting at the given params.
        
        All exceptions are caught and reported.
        
        Returns a tuple of (`u_opt`, `f_opt`, `return_code`).
        If it fails, returns a tuple of (None, None, `sys.exc_info()`).
        """        
        try:
            if self.thresh is not None:
                l = self.run.DV2ln_prob(params, sign=-1)
                if scipy.isinf(l) or scipy.isnan(l) or l > self.thresh:
                    warnings.warn("Bad start, skipping! lp=%.3g" % (l,))
                    return None
                else:
                    print("Good start: lp=%.3g" % (l,))

            # run a global optimizer:
            p0 = self.run.params.copy()
            p0[~self.run.fixed_params] = params
            # uopt = opt.optimize(self.run.lnprior().elementwise_cdf(p0)[~self.run.fixed_params])
            
            # Then polish the minimum:
            opt = nlopt.opt(nlopt.LN_SBPLX, len(params))
            opt.set_max_objective(self.run.u2ln_prob)
            opt.set_lower_bounds([0.0,] * opt.get_dimension())
            opt.set_upper_bounds([1.0,] * opt.get_dimension())
            opt.set_ftol_rel(1e-8)
            opt.set_maxtime(3600 * 12)
            # uopt = opt.optimize(uopt)
            uopt = opt.optimize(self.run.lnprior().elementwise_cdf(p0)[~self.run.fixed_params])
            
            # Convert uopt back to params:
            u_full = 0.5 * scipy.ones_like(self.run.params, dtype=float)
            u_full[~self.run.fixed_params] = uopt
            p_opt = self.run.lnprior().sample_u(u_full)[~self.run.fixed_params]
            
            out = (p_opt, opt.last_optimum_value(), opt.last_optimize_result())

            #print("Done. Made %d calls to STRAHL." % (NUM_STRAHL_CALLS,))
            return out
        except:
            warnings.warn(
                "Minimizer failed, skipping sample. Error is: %s: %s."
                % (
                    sys.exc_info()[0],
                    sys.exc_info()[1]
                )
            )
        return (None, None, sys.exc_info())





### SAVED code  -- setting forced-identifiability knots with minimum distance by tricking hyper-triangle sampling
            # if self.free_D_knots and self.nkD>0: # D knots
            #     _D_knots = np.zeros(self.nkD+1)  # extra dummy knot at 0
                
            #     for n,cubeval in enumerate(cube[num_D+num_V: num_D+num_V+self.nkD]):
            #         previous = np.minimum(_D_knots[n]+self.min_knots_dist, self.outermost_knot)
            #         _D_knots[n+1] = self.outermost_knot - pow(cubeval, 1.0 / (self.nkD - n))*\
            #                         (self.outermost_knot - np.maximum(previous, self.innermost_knot))

            #     # eliminate first dummy knot and create new grid
            #     D_knots = _D_knots[1:]
            #     # set knotgrid_D --> allows updating of self.num_axis_D_coeffs, etc.
            #     self.knotgrid_D = np.concatenate(([self.innermost_knot,], D_knots, [self.outermost_knot,]))  

            # if self.free_V_knots and self.nkV>0: # V knots              
            #     self.outermost_knot = np.max(self.knotgrid_V)
            #     self.innermost_knot = np.min(self.knotgrid_V) #0.0
            #     _V_knots = np.zeros(self.nkV+1)  # extra dummy knot at 0
                
            #     for n,cubeval in enumerate(cube[num_D+num_V+self.nkD:num_D+num_V+self.nkD+self.nkV]):
            #         previous = np.minimum(_V_knots[n]+self.min_knots_dist, self.outermost_knot)
            #         _V_knots[n+1] = self.outermost_knot - pow(cubeval, 1.0 / (self.nkV - n))*\
            #                         (self.outermost_knot - np.maximum(previous, self.innermost_knot))

            #     V_knots = _V_knots[1:]
            #     # set knotgrid_D--> allows updating of self.num_axis_V_coeffs, etc.
            #     self.knotgrid_V = np.concatenate(([self.innermost_knot,], V_knots, [self.outermost_knot,]))  




class DV_approximation(Run):
    '''Class used to find approximations of arbitrary D and V with the parameters chosen for a given BITS run.
    This class inherits from bits.Run, but is separate to facilitate re-loading/modifications. 

    MWE:
    import bits_helper
    from scipy.interpolate import interp1d
    roa_grid_out, explicit_D, explicit_V = bits_helper.get_merged_transport_models(mit.res.shot, rho_choice='r/a', plot=False)
    D_chosen = interp1d(roa_grid_out, explicit_D, bounds_error=False, fill_value=(explicit_D[0],explicit_D[-1]))(mit.res.roa_grid_DV)
    V_chosen = interp1d(roa_grid_out, explicit_V, bounds_error=False, fill_value=(explicit_V[0],explicit_V[-1]))(mit.res.roa_grid_DV)
    ex = DV_approximation(mit.res)
    ex.find_closest_representation(D_chosen, V_chosen)
    '''
    def __init__(self, res):
        self.__dict__ = res.__dict__.copy()

    def find_closest_representation(self, D_chosen, V_chosen, guess=None):
        """Find the closest representation of the given D, V profiles with the current basis functions (splines only!),
        i.e. given some D and V on self.roa_grid_DV, find the spline parameters that most closely represent 
        them (values and knots). 
        This can be useful to find how well a certain representation (e.g. 3 radial coeffs) can truly represent
        certain profiles on the detailed radial grid. 

        Parameters
        ----------
        D_chosen : array of float
            The values of D. Must be given on the same internal roa_grid_DV as
            the current run instance.
        V_chosen : array of float
            The values of V. Must be given on the same internal roa_grid_DV as
            the current run instance.
        guess : array of float, optional
            The initial guess to use for the parameters when running the
            optimizer. If not present, a random draw from the prior is used.
        """

        #num_proc=4
        #pool = InterruptiblePool(processes=num_proc)
        #random_starts=8

        # TODO: This needs random starts!
        bounds = [list(v) for v in np.array(self.bounds)[~self.fixed_params]]

        for v in bounds:
            if v[0] is not None and np.isinf(v[0]):
                v[0] = None
            if v[1] is not None and np.isinf(v[1]):
                v[1] = None

        if guess is None:
            #guess = self.get_lnprior(size=random_starts).random_draw()[~self.fixed_params,:].T
            guess = self.get_lnprior().random_draw()[~self.fixed_params,:]

        opt_res = scipy.optimize.minimize(
            self.objective_func,
            guess,
            args=(D_chosen, V_chosen),
            method='L-BFGS-B',
            # method='SLSQP',
            bounds=bounds
        )
        
        nD,nV,kD,kV,nkD,nkV,nS,nDiag,nW = self.get_indices()

        params_opt = copy.deepcopy(self.params)
        params_opt[~self.fixed_params] = opt_res.x
        D_z, V_z, times_DV = self.eval_DV(params_opt)

        fig,ax = plt.subplots(1,2, figsize=(9,4))
        ax[0].plot(self.rhop, D_z[:,0,1], 'r')
        ax[0].plot(self.rhop, D_chosen, 'b')
        ax[1].plot(self.rhop, V_z[:,0,1], 'r', label='Reconstruction')
        ax[1].plot(self.rhop, V_chosen, 'b', label='Input')
        ax[0].set_xlabel(r'$\rho_p$')
        ax[1].set_xlabel(r'$\rho_p$')
        ax[0].set_ylabel(r'$D$ [$m^2/s$]')
        ax[1].set_ylabel(r'$v$ [$m/s$]')
        ax[1].legend().set_draggable(True)
        plt.tight_layout()

        return opt_res

    def objective_func(self,pp, D_chosen, V_chosen):
        """Objective function for the minimizer in :py:meth:`find_closest_representation`.
        Assumes time and charge-state independent D and V.
        """
        #nD,nV,kD,kV,nkD,nkV,nS,nDiag,nW = self.get_indices()

        D_z, V_z, times_DV = self.eval_DV(pp)

        return scipy.sqrt((scipy.concatenate((D_z[:,0,1] - D_chosen, V_z[:,0,1] - V_chosen))**2).sum())

