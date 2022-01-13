
hfactor='h89'

fn = FigureNotebook(1, name='neg triang')
fig2, ax2 = fn.subplots(label='Prad/Pin vs. '+hfactor.upper())
fig22, ax22 = fn.subplots(label='Prad vs. '+hfactor.upper())
fig5, ax5 = fn.subplots(label='Psol/Pin vs '+hfactor.upper())
fig55, ax55 = fn.subplots(label='Psol vs '+hfactor.upper())
fig6, ax6 = fn.subplots(label='neutrons')

ax2.scatter([],[], marker='s', s=80, label='H-mode')
ax2.scatter([],[], marker='o', s=80, label='L-mode')
ax2.legend(frameon=False)
ax22.scatter([],[], marker='s', s=80, label='H-mode')
ax22.scatter([],[], marker='o', s=80, label='L-mode')
ax22.legend(frameon=False)


ax5.scatter([],[], marker='s', s=80, label='H-mode')
ax5.scatter([],[], marker='o', s=80, label='L-mode')
ax5.legend(frameon=False)
ax55.scatter([],[], marker='s', s=80, label='H-mode')
ax55.scatter([],[], marker='o', s=80, label='L-mode')
ax55.legend(frameon=False)

options = {}
#options.update({ 180519: ['o','b', 2300, 4500],
#            180520: ['s','r', 2300, 4500],
#            180523: ['o','b', 2300, 4500],
#            180524: ['o','b', 2300, 4500],
#            180526: ['o','b', 2300, 4500],
#            180527: ['o','b', 2300, 4500],
#            180528: ['o','b', 2300, 4500],
#            180529: ['o','b', 2300, 4200],
#            180530: ['o','b', 2300, 4300],
#            180533: ['o','b', 2300, 4500],
#})
options.update({
            186473: ['o','g', 1500, 3600],
            186478: ['s','m', 1500, 4800],
})

for ii,shot in enumerate(options.keys()):

    marker = options[shot][0]
    col = options[shot][1]
    tmin = options[shot][2]
    tmax = options[shot][3]

    # plot every 50 ms in chosen interval
    time = np.arange(tmin,tmax,50)

    # get average C concentration inside rho=0.8 and Zeff from QUICKFIT
    ne = 
    nC = np.

    # Total ohmic + smoothed NBI Power + ECH + ICH - WDOT on EFIT01 timebase
    Ptot_mds = OMFITmdsValue(server='DIII-D', shot=shot, treename='TRANSPORT', TDI='.GLOBAL.TIMES.INPUTS.PTOT')
    Ptot = interpolate.interp1d(Ptot_mds.dim_of(0), Ptot_mds.data())(time)

    # Radiated power in confined plasma
    Prad_core_mds = OMFITmdsValue(server='DIII-D', shot=shot, TDI='prad_core')
    Prad_core = interpolate.interp1d(Prad_core_mds.dim_of(0), Prad_core_mds.data())(time)

    H98_mds = OMFITmdsValue(server='DIII-D', shot=shot, TDI='h_thh98y2')
    H89_mds = OMFITmdsValue(server='DIII-D', shot=shot, TDI='h_l89')
    H98 = interpolate.interp1d(H98_mds.dim_of(0), H98_mds.data())(time)
    H89 = interpolate.interp1d(H89_mds.dim_of(0), H89_mds.data())(time)

    neutrons_mds = OMFITmdsValue(server='DIII-D', shot=shot, TDI='neutronsrate')
    neutrons = interpolate.interp1d(neutrons_mds.dim_of(0), neutrons_mds.data())(time)

    # power going into the SOL:
    Psol = Ptot - Prad_core
    
    sc = ax2.scatter(Prad_core/Ptot, H98 if hfactor=='h98' else H89, c=Ptot/1e6, 
                        s=80, marker=marker, label=str(shot)) 
    sc22 = ax22.scatter(Prad_core/1e6, H98 if hfactor=='h98' else H89, c=Ptot/1e6, 
                        s=80, marker=marker, label=str(shot)) 

    sc2 = ax5.scatter(Psol/Ptot, H98 if hfactor=='h98' else H89, c=Ptot/1e6, 
                        s=80, marker=marker, label=str(shot)) 

    sc5 = ax55.scatter(Psol/1e6, H98 if hfactor=='h98' else H89, c=Ptot/1e6, 
                        s=80, marker=marker, label=str(shot)) 

    sc3 = ax6.scatter(Psol/Ptot, H98 if hfactor=='h98' else H89, c=Ptot/1e6, 
                        s=80, marker=marker, label=str(shot)) 
    

ax2.grid(); ax5.grid(); ax22.grid(); ax55.grid()
ax2.axhline(y=1., ls='--')
ax22.axhline(y=1., ls='--')
ax5.axhline(y=1., ls='--')
ax55.axhline(y=1., ls='--')

ax2.set_xlabel(r'$P_{rad}/P_{in}$')
ax22.set_xlabel(r'$P_{rad}$ [MW]')
ax2.set_ylabel(r'$H_{98}$' if hfactor=='h98' else r'$H_{89}$')
ax22.set_ylabel(r'$H_{98}$' if hfactor=='h98' else r'$H_{89}$')
ax5.set_xlabel(r'$P_{sol}/P_{in}$')
ax5.set_ylabel(r'$H_{98}$' if hfactor=='h98' else r'$H_{89}$')
ax55.set_xlabel(r'$P_{sol}$ [MW]')
ax55.set_ylabel(r'$H_{98}$' if hfactor=='h98' else r'$H_{89}$')
cbar = fig2.colorbar(sc, ax=ax2)
cbar.set_label(r'$P_{in}$ [MW]')
cbar2 = fig5.colorbar(sc, ax=ax5)
cbar2.set_label(r'$P_{in}$ [MW]')
cbar22 = fig22.colorbar(sc22, ax=ax22)
cbar22.set_label(r'$P_{in}$ [MW]')
cbar55 = fig55.colorbar(sc5, ax=ax55)
cbar55.set_label(r'$P_{in}$ [MW]')


