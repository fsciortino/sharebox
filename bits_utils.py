'''
Utils for BITE framework. 

Sciortino, 7/3/2020
'''

import scipy, copy
import numpy as np
import matplotlib.pyplot as plt
import gptools
import matplotlib.gridspec as mplgs
import matplotlib.widgets as mplw
import itertools
from IPython import embed

def interp_max(x, y, err_y=None, s_guess=0.2, s_max=10.0, l_guess=0.005, fixed_l=False, debug_plots=False, method='GP'):
    """Compute the maximum value of the smoothed data.
    
    Estimates the uncertainty using Gaussian process regression and returns the
    mean and uncertainty.
    
    Parameters
    ----------
    x : array of float
        Abscissa of data to be interpolated.
    y : array of float
        Data to be interpolated.
    err_y : array of float, optional
        Uncertainty in `y`. If absent, the data are interpolated.
    s_guess : float, optional
        Initial guess for the signal variance. Default is 0.2.
    s_max : float, optional
        Maximum value for the signal variance. Default is 10.0
    l_guess : float, optional
        Initial guess for the covariance length scale. Default is 0.03.
    fixed_l : bool, optional
        Set to True to hold the covariance length scale fixed during the MAP
        estimate. This helps mitigate the effect of bad points. Default is True.
    debug_plots : bool, optional
        Set to True to plot the data, the smoothed curve (with uncertainty) and
        the location of the peak value.
    method : {'GP', 'spline'}, optional
        Method to use when interpolating. Default is 'GP' (Gaussian process
        regression). Can also use a cubic spline.
    """
    grid = np.linspace(max(0, x.min()), min(0.08, x.max()), 1000)
    if method == 'GP':
        hp = (
            gptools.UniformJointPrior([(0, s_max),]) *
            gptools.GammaJointPriorAlt([l_guess,], [0.1,])
        )
        k = gptools.SquaredExponentialKernel(
            # param_bounds=[(0, s_max), (0, 2.0)],
            hyperprior=hp,
            initial_params=[s_guess, l_guess],
            fixed_params=[False, fixed_l]
        )
        gp = gptools.GaussianProcess(k, X=x, y=y, err_y=err_y)
        gp.optimize_hyperparameters(verbose=True, random_starts=100)
        m_gp, s_gp = gp.predict(grid)
        i = m_gp.argmax()
    elif method == 'spline':
        m_gp = scipy.interpolate.UnivariateSpline(
            x, y, w=1.0 / err_y, s=2*len(x)
        )(grid)
        if np.isnan(m_gp).any():
            print(x)
            print(y)
            print(err_y)
        i = m_gp.argmax()
    else:
        raise ValueError("Undefined method %s" % (method,))
    
    if debug_plots:
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        a.errorbar(x, y, yerr=err_y, fmt='.', color='b')
        a.plot(grid, m_gp, color='g')
        if method == 'GP':
            a.fill_between(grid, m_gp - s_gp, m_gp + s_gp, color='g', alpha=0.5)
        a.axvline(grid[i])
    
    if method == 'GP':
        return (m_gp[i], s_gp[i])
    else:
        return m_gp[i]



class TruthData:
    """Class to hold the truth values for synthetic data.
    """
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

##############################################################
#
#
#
#
#
################################################################
class Injection:
    """Class to store information on a given injection.
    """
    def __init__(self, t_inj, t_start, t_stop):
        self.t_inj = t_inj
        self.t_start = t_start
        self.t_stop = t_stop


class SpectrumSignal:
    def __init__(self, y, std_y, y_norm, std_y_norm, lams, t, name, atomdat_idx, pos=None, sqrtpsinorm=None, weights=None,
                 blocks=0):
        """Class to store the data from a given diagnostic.
        
        This is different from the :py:class:`Signal` class because it includes a wavelength dependence
        for signals. Normalization of signals is not included, since the spectral response of the detector
        is assumed to be already accounted for and the absolute amplitude of the spectrum is assumed to
        contain all the information needed.
        
        Parameters
        ----------
        y : array, (`n_lam`, `n_time`, `n_ch`)
            The XICS signals as a function of wavelength, time and space. 
            If `pos` is not None, "space" refers to the chords. Bad points should be set to nan.
        std_y : array, (`n_lam`, `n_time`, `n_ch`)
            The uncertainty in the unnormalized, baseline-subtracted data as a
            function of wavelength, time and space.
        y_norm : array, (`n_lam`, `n_time`, `n_ch`)
            The normalized XICS signals as a function of wavelength, time and space. 
            If `pos` is not None, "space" refers to the chords. Bad points should be set to nan.
        std_y_norm : array, (`n_lam`, `n_time`, `n_ch`)
            The uncertainty in the normalized and baseline-subtracted data as a
            function of wavelength, time and space.
        lams : array, (`n_lam`, `n_ch`)
             The wavelength vector of the data, assumed to be time independent, but always a function of
             spatial channel to allow for XICS curvature of wavelength vectors across the detector.
        t : array, (`n_time`,)
            The time vector of the data.
        name : str
            The name of the signal.
        atomdat_idx : int or array of int, (`n_ch`,)
            The index or indices of the signals in the atomdat file. If a single value is given, 
            it is used for all of the signals. If a 1d array is provided, these are the indices for each 
            of the signals in `y`. If `atomdat_idx` (or one of its entries) is -1, it will be treated as
            an SXR measurement.
        pos : array, (4,) or (`n_ch`, 4), optional
            The POS vector(s) for line-integrated data. If not present, the data are assumed to be local 
            measurements at the locations in `sqrtpsinorm`. If a 1d array is provided, it is used for all of the
            chords in `y`. Otherwise, there must be one pos vector for each of the chords in `y`.
        sqrtpsinorm : array, (`n_ch`,), optional
            The square root of poloidal flux grid the (local) measurements are given on. 
            If line-integrated measurements with the standard Aurora grid for their quadrature points 
            are to be used this should be left as None.
        weights : array, (`n_ch`, `n_quadrature`), optional
            The quadrature weights to use. This can be left as None for a local measurement or can be set later.
        blocks : int or array of int, (`n`), optional
            A set of flags indicating which channels in the :py:class:`Signal`should be treated together as a 
            block when normalizing. If a single int is given, all of the channels will be taken together. Otherwise,
            any channels sharing the same block number will be taken together.
        """
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 3:
            raise ValueError("y must have 3 dimensions!")
        self.std_y = np.asarray(std_y, dtype=float)
        if self.y.shape != self.std_y.shape:
            raise ValueError("The shapes of y and std_y must match!")

        self.y_norm = np.asarray(y_norm, dtype=float)
        if self.y_norm.ndim != 3:
            raise ValueError("y_norm must have 3 dimensions!")
        self.std_y_norm = np.asarray(std_y_norm, dtype=float)
        if self.y_norm.shape != self.std_y_norm.shape:
            raise ValueError("The shapes of y_norm and std_y_norm must match!")

        ####
        self.lams = np.asarray(lams, dtype=float)
        if self.lams.ndim!=1 and self.lams.ndim!=2:
            raise ValueError("lams must have one or two dimensions!")
        if self.lams.ndim == 1:
            self.lams = np.tile(self.lams, (self.y.shape[2], 1)).T # set to have number of channels as second dimension
        if len(self.lams) != self.y.shape[0]:
            raise ValueError("The length of lams must equal the length of the leading dimension of y!")

        self.t = np.asarray(t, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("t must have one dimension!")
        if len(self.t) != self.y.shape[1]:
            raise ValueError("The length of t must equal the length of the second dimension of y!")

        if isinstance(name, str):
            name = [name,] * self.y.shape[2]
        self.name = name

        try:
            iter(atomdat_idx)
        except TypeError:
            self.atomdat_idx = atomdat_idx * np.ones(self.y.shape[2], dtype=int)
        else:
            self.atomdat_idx = np.asarray(atomdat_idx, dtype=int)
            if self.atomdat_idx.ndim != 1:
                raise ValueError("atomdat_idx must have at most one dimension!")
            if len(self.atomdat_idx) != self.y.shape[2]:
                raise ValueError("1d atomdat_idx must have the same number of elements as the second dimension of y!")

        if pos is not None:
            pos = np.asarray(pos, dtype=float)
            if pos.ndim not in (1, 2):
                raise ValueError("pos must have one or two dimensions!")
            if pos.ndim == 1 and len(pos) != 4:
                raise ValueError("pos must have 4 elements!")
            if pos.ndim == 2 and (pos.shape[0] != self.y.shape[2] or pos.shape[1] != 4):
                raise ValueError("pos must have shape (n_ch, 4)!")
        
        try:
            iter(blocks)
        except TypeError:
            self.blocks = blocks * np.ones(self.y.shape[2], dtype=int)
        else:
            self.blocks = np.asarray(blocks, dtype=int)
            if self.blocks.ndim != 1:
                raise ValueError("blocks must have at most one dimension!")
            if len(self.blocks) != self.y.shape[2]:
                raise ValueError("1d blocks must have the same number of elements as the second dimension of y!")
                
        self.pos = pos
        self.sqrtpsinorm = sqrtpsinorm
        self.weights = weights

    def sort_t(self):
        """Sort the time axis.
        """
        srt = self.t.argsort()
        self.t = self.t[srt]
        self.y = self.y[:, srt, :]
        self.std_y = self.std_y[:, srt, :]

    
    def plot_data(self, y_synth=None, ch=0, norm=True, fig=None, **kwargs):
        """Make a big plot with all of the data.
        
        Parameters
        -----------------
        y_synth : array (`n_lam`, `n_time`, `n_ch`)
            Synthetic signal (at the same wavelengths, times and channels as self.y). If given, 
            these will be plotted together with the experimental signals.
        ch : int
            Channel to plot. Since this class (unlike :py:class:`Signal`) includes spectral resolution for each
            chord and time, this method allows selection of one chord at a time to plot. 
        norm : bool
            If True, plot normalized signals (y_norm and std_y_norm) rather than the unnormalized ones
            (y and sd_y). Default is True.
        fig : :py:class:`Figure`, optional
            The figure instance to make the subplots in. If not provided, a
            figure will be created.
        share_y : bool, optional
            If True, the y axes of all of the subplots will have the same scale.
            Default is False (each y axis is automatically scaled individually).
        """
        if y_synth is not None and self.y.shape != y_synth.shape:
            raise ValueError('Shape of input y_synth array should be the same as the one of experimental signals!')
        self.y_synth = y_synth
        self.ch = ch
        self.norm = norm
        self.fig = fig
        
        if fig is None:
            self.fig = plt.figure() #num=f'XICS channel {self.ch}')

        # initialize slider spectrum:
        self.init_plot(**kwargs)

        # create slider
        self.slider = mplw.Slider(
            self.a_slider,
            'time [s]',
            0,
            self.y.shape[1]-1,
            valinit=0,
            valfmt='%d'
        )

        # initialize
        self.slider.on_changed(lambda dum: self.update(dum,**kwargs))
        self.update(0, **kwargs)

        self.fig.canvas.mpl_connect(
            'key_press_event',
            lambda evt: self.arrow_respond(self.slider, evt)
        )


    def arrow_respond(self,slider, event):
        ''' Method to respond to slider within the function scope '''
        if event.key == 'right':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key == 'left':
            slider.set_val(max(slider.val - 1, slider.valmin))
        elif event.key == 'up':
            if self.ch<self.y.shape[2]: self.ch += 1
        elif event.key == 'down':
            if self.ch>0: self.ch -= 1            


    def init_plot(self, **kwargs):
        ''' Create spectrum plot and slider '''

        self.fig.set_size_inches(10,7, forward=True)
        self.a_labels = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=self.fig) 
        self.a_plot = plt.subplot2grid((10,1),(1,0),rowspan = 7, colspan = 1, fig=self.fig, sharex=self.a_labels) 
        self.a_slider = plt.subplot2grid((10,1),(9,0),rowspan = 1, colspan = 1, fig=self.fig) 
        self.a_labels.axis('off')
        self.a_plot.set_xlabel(r'$\lambda$ [$\AA$]')
        self.a_plot.set_ylabel(r'b [A.U.]')
        
        self.a_plot.set_xlim([3.17, 3.215]) # A, He-lke Ca spectrum

        with open('/home/sciortino/usr/python3modules/bsfc/data/hirexsr_wavelengths.csv', 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()]
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]])
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]])
            lineName = np.array([ld[3] for ld in lineData[2:]])

        # select only lines from Ca
        xics_lams = lineLam[lineZ==20]
        xics_names = lineName[lineZ==20]

        self.xics_names = xics_names[(xics_lams>3.17)&(xics_lams<3.215)]
        self.xics_lams = xics_lams[(xics_lams>3.17)&(xics_lams<3.215)]

        for ii,_line in enumerate(self.xics_lams):
            self.a_plot.axvline(_line, c='r', ls='--')
            self.a_labels.axvline(_line, c='r', ls='--')
            self.a_labels.text(_line, 0.5, self.xics_names[ii], rotation=90, fontdict={'fontsize':14})

        self.a_plot.errorbar(self.lams[:,self.ch], 
                             self.y_norm[:,0,self.ch] if self.norm else self.y[:,0,self.ch], 
                             self.std_y_norm[:,0,self.ch] if self.norm else self.std_y[:,0,self.ch], 
                             fmt='.', **kwargs)

        if self.y_synth is not None:
            # plot synthetic data
            self.a_plot.plot(self.lams[:,self.ch], self.y_synth[:,0,self.ch])

        self.a_plot.grid('on', which='both')
        self.title = self.fig.suptitle('')

    def update(self, dum, **kwargs):
        
        self.a_plot.cla()
        self.a_labels.cla()
        self.a_labels.axis('off')
        
        i = int(self.slider.val)
        self.a_plot.errorbar(self.lams[:,self.ch], 
                             self.y_norm[:,i,self.ch] if self.norm else self.y[:,i,self.ch], 
                             self.std_y_norm[:,i,self.ch] if self.norm else self.std_y[:,i,self.ch], 
                             fmt='.', **kwargs)

        if self.y_synth is not None:
            # plot synthetic data
            self.a_plot.plot(self.lams[:,self.ch], self.y_synth[:,i,self.ch])
            
        for ii,_line in enumerate(self.xics_lams):
            self.a_plot.axvline(_line, c='r', ls='--')
            self.a_labels.axvline(_line, c='r', ls='--')
            self.a_labels.text(_line, 0.5, self.xics_names[ii], rotation=90, fontdict={'fontsize':14})
            
        self.a_plot.set_xlabel(r'$\lambda$ [\AA]')
        self.a_plot.set_ylabel(r'b [A.U.]')

        self.a_plot.relim()
        self.a_plot.autoscale()
        
        self.title.set_text(f'time = {self.t[i]:.5f}, channel = {self.ch}')

        self.fig.canvas.draw()
    







    
class Signal:
    def __init__(self, y, std_y, y_norm, std_y_norm, t, name, atomdat_idx, pos=None, sqrtpsinorm=None, 
                 weights=None, blocks=0,m=None,s=None ):
        """Class to store the data from a given diagnostic.
        
        In the parameter descriptions, `n` is the number of signals (both
        spatial and temporal) contained in the instance.
        
        Parameters
        ----------
        y : array, (`n_time`, `n`)
            The unnormalized, baseline-subtracted data as a function of time and
            space. If `pos` is not None, "space" refers to the chords. Wherever
            there is a bad point, it should be set to NaN.
        std_y : array, (`n_time`, `n`)
            The uncertainty in the unnormalized, baseline-subtracted data as a
            function of time and space.
        y_norm : array, (`n_time`, `n`)
            The normalized, baseline-subtracted data.
        std_y_norm : array, (`n_time`, `n`)
            The uncertainty in the normalized, baseline-subtracted data.
        t : array, (`n_time`,)
            The time vector of the data.
        name : str
            The name of the signal.
        atomdat_idx : int or array of int, (`n`,)
            The index or indices of the signals in the atomdat file. If a single
            value is given, it is used for all of the signals. If a 1d array is
            provided, these are the indices for each of the signals in `y`. If
            `atomdat_idx` (or one of its entries) is -1, it will be treated as
            an SXR measurement.
        pos : array, (4,) or (`n`, 4), optional
            The POS vector(s) for line-integrated data. If not present, the data
            are assumed to be local measurements at the locations in
            `sqrtpsinorm`. If a 1d array is provided, it is used for all of the
            chords in `y`. Otherwise, there must be one pos vector for each of
            the chords in `y`.
        sqrtpsinorm : array, (`n`,), optional
            The square root of poloidal flux grid the (local) measurements are
            given on. If line-integrated measurements with the standard Aurora
            grid for their quadrature points are to be used this should be left
            as None.
        weights : array, (`n`, `n_quadrature`), optional
            The quadrature weights to use. This can be left as None for a local
            measurement or can be set later.
        blocks : int or array of int, (`n`), optional
            A set of flags indicating which channels in the :py:class:`Signal`
            should be treated together as a block when normalizing. If a single
            int is given, all of the channels will be taken together. Otherwise,
            any channels sharing the same block number will be taken together.
        m : float
            maximum signal recorded across any chords and any time for this diagnostic.
            This value is used for normalization of the signals. 
        s : float
            uncertainty in m (see above)
        """
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim != 2:
            raise ValueError("y must have two dimensions!")
        self.std_y = np.asarray(std_y, dtype=float)
        if self.y.shape != self.std_y.shape:
            raise ValueError("The shapes of y and std_y must match!")
        self.y_norm = np.asarray(y_norm, dtype=float)
        if self.y.shape != self.y_norm.shape:
            raise ValueError("The shapes of y and y_norm must match!")
        self.std_y_norm = np.asarray(std_y_norm, dtype=float)
        if self.std_y_norm.shape != self.y.shape:
            raise ValueError("The shapes of y and std_y_norm must match!")
        self.t = np.asarray(t, dtype=float)
        if self.t.ndim != 1:
            raise ValueError("t must have one dimension!")
        if len(self.t) != self.y.shape[0]:
            raise ValueError("The length of t must equal the length of the leading dimension of y!")
        if isinstance(name, str):
            name = [name,] * self.y.shape[1]
        self.name = name
        try:
            iter(atomdat_idx)
        except TypeError:
            self.atomdat_idx = atomdat_idx * np.ones(self.y.shape[1], dtype=int)
        else:
            self.atomdat_idx = np.asarray(atomdat_idx, dtype=int)
            if self.atomdat_idx.ndim != 1:
                raise ValueError("atomdat_idx must have at most one dimension!")
            if len(self.atomdat_idx) != self.y.shape[1]:
                raise ValueError("1d atomdat_idx must have the same number of elements as the second dimension of y!")
        if pos is not None:
            pos = np.asarray(pos, dtype=float)
            if pos.ndim not in (1, 2):
                raise ValueError("pos must have one or two dimensions!")
            if pos.ndim == 1 and len(pos) != 4:
                raise ValueError("pos must have 4 elements!")
            if pos.ndim == 2 and (pos.shape[0] != self.y.shape[1] or pos.shape[1] != 4):
                raise ValueError("pos must have shape (n, 4)!")
        
        self.pos = pos
        self.sqrtpsinorm = sqrtpsinorm
        
        self.weights = weights
        
        try:
            iter(blocks)
        except TypeError:
            self.blocks = blocks * np.ones(self.y.shape[1], dtype=int)
        else:
            self.blocks = np.asarray(blocks, dtype=int)
            if self.blocks.ndim != 1:
                raise ValueError("blocks must have at most one dimension!")
            if len(self.blocks) != self.y.shape[1]:
                raise ValueError("1d blocks must have the same number of elements as the second dimension of y!")
        
        if isinstance(m,(float)):
            self.m=m
        elif m==None: 
            pass
        else:
            raise ValueError("maximum signal m must be a float!")
        if isinstance(s,(float)):
            self.s=s
        elif s==None:
            pass
        else: 
            raise ValueError("maximum signal m must be a float!")

    def sort_t(self):
        """Sort the time axis.
        """
        srt = self.t.argsort()
        self.t = self.t[srt]
        self.y = self.y[srt, :]
        self.std_y = self.std_y[srt, :]
        self.y_norm = self.y_norm[srt, :]
        self.std_y_norm = self.std_y_norm[srt, :]
    
    def plot_data(self, norm=False, f=None, share_y=False, y_label='$b$ [AU]',
                  max_ticks=None, rot_label=False, fast=False, ncol=6):
        """Make a big plot with all of the data.
        
        Parameters
        ----------
        norm : bool, optional
            If True, plot the normalized data. Default is False (plot
            unnormalized data).
        f : :py:class:`Figure`, optional
            The figure instance to make the subplots in. If not provided, a
            figure will be created.
        share_y : bool, optional
            If True, the y axes of all of the subplots will have the same scale.
            Default is False (each y axis is automatically scaled individually).
        y_label : str, optional
            The label to use for the y axes. Default is '$b$ [AU]'.
        max_ticks : int, optional
            The maximum number of ticks on the x and y axes. Default is no limit.
        rot_label : bool, optional
            If True, the x axis labels will be rotated 90 degrees. Default is
            False (do not rotate).
        fast : bool, optional
            If True, errorbars will not be drawn in order to make the plotting
            faster. Default is False
        ncol : int, optional
            The number of columns to use. Default is 6.
        """
        if norm:
            y = self.y_norm
            std_y = self.std_y_norm
        else:
            y = self.y
            std_y = self.std_y
        
        if f is None:
            f = plt.figure()
        
        ncol = int(min(ncol, self.y.shape[1]))
        nrow = int(np.ceil(1.0 * self.y.shape[1] / ncol))
        gs = mplgs.GridSpec(nrow, ncol)
        
        a = []
        i_col = 0
        i_row = 0
        
        for k in range(0, self.y.shape[1]):
            a.append(
                f.add_subplot(
                    gs[i_row, i_col],
                    sharex=a[0] if len(a) >= 1 else None,
                    sharey=a[0] if len(a) >= 1 and share_y else None
                )
            )

            '''
            if i_col > 0: # and share_y:
                plt.setp(a[-1].get_yticklabels(), visible=False)
            else:
                a[-1].set_ylabel(y_label)
            '''
            if i_col==0:
                a[-1].set_ylabel(y_label)

            if i_row < nrow - 2 or (i_row == nrow - 2 and i_col < self.y.shape[1] % (nrow - 1)):
                plt.setp(a[-1].get_xticklabels(), visible=False)
            else:
                a[-1].set_xlabel('$t$ [s]') 
                if rot_label:
                    plt.setp(a[-1].xaxis.get_majorticklabels(), rotation=90)
            i_col += 1
            if i_col >= ncol:
                i_col = 0
                i_row += 1

                
            #if not name:
            #    name = self.name[k] 
            #name='XICS-w'  ### temporary
            a[-1].text(0.7,a[-1].get_ylim()[1]*0.9,'%s, %d' % (
                self.name[k], k),horizontalalignment='center',verticalalignment='center', transform=a[-1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.5), fontsize=16
                   )
            a[-1].grid(which='major'); a[-1].grid(which='minor'); 
            
            good = ~np.isnan(self.y[:, k])
            if fast:
                a[-1].plot(self.t[good], y[good, k], '.')
            else:
                a[-1].errorbar(self.t[good], y[good, k], yerr=std_y[good, k], fmt='.')
            if max_ticks is not None:
                a[-1].xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
                a[-1].yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks - 1))
        
        if share_y:
            a[0].set_ylim(bottom=0.0)
            a[0].set_xlim(self.t.min(), self.t.max())
        
        f.canvas.draw()
        
        return (f, a)
