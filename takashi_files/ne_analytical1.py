from numpy import float64, zeros, arange, exp, ones, copy, append, dot, mean, std, sum, asarray, max, sqrt, pi, correlate, meshgrid, abs, linalg,\conjugate, fliplr, log10, floor, power, angle, conjugate, linspace, hamming, hanning, add, diff, where, power, real, random, inf, isfinite, eye,\log, diag, diff, histogram, percentile, maximum, minimum, histogram2d, vectorize, int32, int64, tanh, concatenate, cumsum, fill_diagonal, triu, tril, tilefrom scipy.interpolate import interp1dwith open("func_ne.py") as f:     code = compile(f.read(), "func_c_o_ne_long_cx.py", 'exec') #defines the input profiles    exec(code)    def plot_comp(F, labelsize=13, nz=3, log_plot=1, thin=2, rho=1):    color_arr=["r", "b", "g", "y", "c", "m", "k", (0.3, 0.6, 0.1), (0.8, 0.2, 0.1), (0.2, 0.3, 0.1) ]    fig = plt.figure( figsize=(5*1.618, 6*1.0) )    ax1=plt.subplot(111)    ax1.grid(True)    for i in range(nz):        if log_plot!=0:            if rho==0:                ax1.semilogy(F.r_btw[::thin], F.nz_sol[-1-i,::thin].T, 'o' , lw=3, alpha=0.6, color=color_arr[i], fillstyle="none", markersize=10)                ax1.semilogy(F.r_btw, F.nz_ans[-i,:].T*F.nz_sol[-1-1,0], '-' , lw=3, label='Ne'+' {0}'.format(F.n_state_all-i)+'+', color=color_arr[i])                    ax1.set_xlabel('r (m)', size=labelsize)            else:                ax1.semilogy(F.rho_btw[::thin], F.nz_sol[-1-i,::thin].T, 'o' , lw=3, alpha=0.6, color=color_arr[i], fillstyle="none", markersize=10)                ax1.semilogy(F.rho_btw, F.nz_ans[-1-i,:].T*F.nz_sol[-1,0], '-' , lw=3, label='Ne'+' {0}'.format(F.n_state_all-i)+'+', color=color_arr[i])                                ax1.set_xlabel(r'$\rho_{pol}$'+' (m)', size=labelsize)                                else:            if rho==0:                            ax1.plot(F.r_btw, F.nz_ans[-1-i,:].T*F.nz_sol[-1,0], 'o' , lw=3, alpha=0.6 , color=color_arr[i])                ax1.plot(F.r_btw, F.nz_sol[-1-i,:].T, '-' , lw=3, label='Ne'+' {0}'.format(F.n_state_all-i)+'+', color=color_arr[i])                ax1.set_xlabel('r (m)', size=labelsize)                            else:                            ax1.plot(F.rho_btw, F.nz_ans[-1-i,:].T*F.nz_sol[-1,0], 'o' , lw=3, alpha=0.6 , color=color_arr[i])                ax1.plot(F.rho_btw, F.nz_sol[-1-i,:].T, '-' , lw=3, label='Ne'+' {0}'.format(F.n_state_all-i)+'+', color=color_arr[i])                                ax1.set_xlabel(r'$\rho_{pol}$'+' (m)', size=labelsize)            ax1.set_ylabel('density (m'+r'$^{-3}$'+')' , size=labelsize+2)    ax1.set_title('Circles are calculated by solving the time evolution.\n Solid lines are analytical\n n_state={0}'.format(F.n_state) , size=labelsize+2)        ax1.tick_params(axis='both', which='major', labelsize=labelsize+2)    plt.legend()        return     class Matrix_ne(object):    u"""        Calculate the profiles of all Ne charge states for given D, v, and other background plasma parameters.    """    def __init__(self, n_state=None,  M = 0.1, L_para=25.0, edge_given=1):                u"""        parameters        --------            n_state: # of charge states used in the calculation. If n_state=None, all but neutral state will be included.                    If n_state< atomic number, lower charge states will be trancated.            M: Moch number used in the parallel low term            L_para: connection length in the parallel low term (m)            Details of the parallel low term can be found in R. Dux, et.al., Nucl Fusion (2011)            doi:10.1088/0029-5515/51/5/053002            edge_given: If not zero, use the input impurity profiles for the charge state balance at the wall.                        If zero, calculate the charge state balance at the wall by neglecting the transport        """            self.acd_arr_ne = acd_arr_ne  #recombination rate [charge state, radial postion]        self.scd_arr_ne = scd_arr_ne  #ionization rate [charge state, radial postion]        self.cdd_arr_ne = ccd_arr_ne  #charge exchange rate [charge state, radial postion]                if n_state==None:            n_state=len( acd_arr_ne[:,0] )        self.n_state=n_state # see argument n_state at __init__        self.n_state_all=len( acd_arr_ne[:,0] ) # number of charge states from 1 to Z, e.g. 10 for neon                self.del_r=diff(r) #dr_vol        self.r=r           #v_vol        self.r_btw=(self.r[1:]+self.r[:-1])*0.5  #v_vol btn the grids         self.rho_btw=( rho_pol_input[1:] + rho_pol_input[:-1])*0.5 #rho_poloidal btw the grids          self.del_r_btw=diff( self.r_btw ) # dr_vol for self.r_btw        self.N=len(r) # total number of spatial grids                 self.ti=ti_input        #ion temperature profile        self.ne=ne_input # input ne profile         self.te=te_input # input te profile                 self.rho=rho_pol_input #rho_poloidal array                        self.M = M             #Mach number for the parallel low term        self.L_para=L_para      #connection length for the parallel low term        self.vel = sqrt( (3*ti_input+te_input )/(2*940e+6) )*3e+8 #parallel flow velocity, 2*940e+6 is the deutrium mass, 3e+8 is the speed of light        par_los_rate_dum = 2/self.L_para*self.M*sqrt( (3*ti_input+te_input )/(2*940e+6) )*3e+8 #parallel low rate        b=1>0.5*( self.rho[1:] + self.rho[:-1] ) #indecies for the confined region        self.par_los_rate =   0.5*(par_los_rate_dum[1:]+par_los_rate_dum[:-1] ) #parallel low rate btw the spatial grid         self.par_los_rate[b] = 0                                              #No parallel loss in the confined region                                self.D=D_input # input D profile         self.D_btw= (self.D[1:]+self.D[:-1])*0.5 # input D profile btw the spatial grids                 self.v=v_input # input v profile         self.v_btw= (self.v[1:]+self.v[:-1])*0.5 # input v profile btw the spatial grids                 self.A_re_ne=acd_arr_ne*ne_input+ccd_arr_ne*n0_input # (recombination rates)*density for each charge state (2D array [charge state, radial position] )        self.B_io_ne=scd_arr_ne*ne_input                     # (ionization rates)*density for each charge state (2D array [charge state, radial position] )            #-----------Defines matrices to be filled later. They are defined btw the grids. Thus (N-1)*(N-1), not N*N        self.a_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )        self.b_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )        self.c_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )                self.A_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )        self.B_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )        self.C_z = zeros( [ self.n_state, self.N-1, self.N-1 ] )                self.exp_Dr = zeros( [  self.N-1, self.N-1 ] )        self.exp_diag = zeros( [  self.N-1, self.N-1 ] )        self.del_r_matrix = zeros( [  self.N-1, self.N-1 ] )                #-----------calculate the charge state balance at the wall by neglecting the transport                self.edge_ratio_ne=ones( len(self.A_re_ne[:,0] ) )        for i in range( 1, len(self.edge_ratio_ne ) ):            self.edge_ratio_ne[i] = self.B_io_ne[i,-1]/self.A_re_ne[i,-1]*copy( self.edge_ratio_ne[i-1] )                                      #-----------Filling the matrices, see T. Nishizawa et.al. (2022), to be submitted.                for i in range(self.n_state):            fill_diagonal( self.a_z[i,:,:], 0.5*(self.A_re_ne[i+self.n_state_all-self.n_state,1:]+self.A_re_ne[i+self.n_state_all-self.n_state,:-1] )*self.r_btw  )            fill_diagonal( self.b_z[i,:,:], 0.5*(self.B_io_ne[i+self.n_state_all-self.n_state,1:]+self.B_io_ne[i+self.n_state_all-self.n_state,:-1] )*self.r_btw  )            fill_diagonal( self.c_z[i,:,:], self.par_los_rate*self.r_btw  )                            for i in range( len(self.del_r) ):            self.del_r_matrix[i,:]= self.del_r        self.del_r_up=triu( copy( self.del_r_matrix ) )        self.del_r_down=tril( copy( self.del_r_matrix ) )                u_vec_dum=append( dot( self.del_r_up, self.v_btw/self.D_btw ), [0.0] )        self.u_vec = 0.5*( u_vec_dum[1:] + u_vec_dum[:-1] )                fill_diagonal( self.exp_Dr,  exp( self.u_vec)/(self.D_btw*self.r_btw ) )        fill_diagonal( self.exp_diag, exp(-self.u_vec) )                        for i in range(self.n_state):            am1=concatenate(  [ [zeros(self.N-1) ], dot( self.del_r_down, self.a_z[i,:,:] ) ] )            am2=0.5*( am1[1:,:]+ am1[:-1,:] )            am3=dot(self.exp_Dr, am2 )            am4=concatenate( [ dot(self.del_r_up, am3), [ zeros(self.N-1) ] ] )            am5=0.5*( am4[1:]+ am4[:-1])            self.A_z[i,:,:]= dot(self.exp_diag, am5 )                        bm1=concatenate(  [ [zeros(self.N-1) ], dot( self.del_r_down, self.b_z[i,:,:] ) ] )            bm2=0.5*( bm1[1:,:]+ bm1[:-1,:] )            bm3=dot(self.exp_Dr, bm2 )            bm4=concatenate( [ dot(self.del_r_up, bm3), [ zeros(self.N-1) ] ] )            bm5=0.5*( bm4[1:]+ bm4[:-1])            self.B_z[i,:,:]= dot(self.exp_diag, bm5 )                        cm1=concatenate(  [ [zeros(self.N-1) ], dot( self.del_r_down, self.c_z[i,:,:] ) ] )            cm2=0.5*( cm1[1:,:]+ cm1[:-1,:] )            cm3=dot(self.exp_Dr, cm2 )            cm4=concatenate( [ dot(self.del_r_up, cm3), [ zeros(self.N-1) ] ] )            cm5=0.5*( cm4[1:]+ cm4[:-1])            self.C_z[i,:,:]= dot(self.exp_diag, cm5 )                                    self.nz_0=asarray([])        for i in range(self.n_state):            if edge_given==0:                self.nz_0=append(   copy(self.nz_0) , exp(-self.u_vec)*self.edge_ratio_ne[i+self.n_state_all-self.n_state] )                        else:                self.read_solution()                self.nz_0=append(   copy(self.nz_0) , exp(-self.u_vec)*self.nz_sol[i, -1] )         self.source_mtx=zeros( [ len(self.nz_0), len(self.nz_0) ])                for i in range(self.n_state):            if i==0:                self.source_mtx[i*( self.N-1):(i+1)*(self.N-1), : ] = concatenate( [ -self.B_z[i+1, :, :]-self.C_z[i, :, :], +self.A_z[i+1, :, :], zeros( [ self.N-1, (self.n_state-2)*(self.N-1) ] ) ], axis=1 )            elif i!=n_state-1:                self.source_mtx[i*( self.N-1):(i+1)*(self.N-1), : ] = concatenate( [ zeros( [ self.N-1, (i-1)*(self.N-1) ] ), self.B_z[i, :, :], -self.A_z[i, :, :]-self.B_z[i+1, :, :]-self.C_z[i, :, :],  self.A_z[i+1, :, :] , zeros( [ self.N-1, (self.n_state-i-2)*(self.N-1) ] ) ], axis=1 )            else:                self.source_mtx[i*( self.N-1):(i+1)*(self.N-1), : ] = concatenate( [  zeros( [ self.N-1, (self.n_state-2)*(self.N-1) ] ) ,  self.B_z[i, :, :], -self.A_z[i, :, :]-self.C_z[i, :, :] ] , axis=1 )                        self.mtx_solv=eye(  len(self.nz_0)  ) - self.source_mtx         nz_ans_dum = ( linalg.solve( self.mtx_solv, self.nz_0 ) ).reshape(-1, self.N-1)        self.nz_ans = nz_ans_dum/nz_ans_dum[-1,0] # impurity profiles [charge state, radial position], when self.n_state<self.n_state_all, lower states are truncated                     def read_solution(self, title='Ne_with_flow.ncdf'):        u"""        Read the impurity profile caluculated by solving the time evolution.  M = 0.1, L_para=25.0 for Ne_with_flow.ncdf        """        wout = Dataset(title,more = 'r')                self.nz_sol=asarray( wout.variables['nz_tr'][:] )        self.nz_sol=( 0.5*(  copy(self.nz_sol[:,1:]) + copy(self.nz_sol[:,:-1]) ) )[::-1,:]def save( name ):    plt.savefig( name, transparent=True  )    print( name+" saved" )    return 0        