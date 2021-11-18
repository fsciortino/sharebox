# Script used to test remote submission to cmodws84 and preparation of TGLF runs

import numpy as np
import sys
import os
import subprocess
import socket
import math

# ===== User settings ======
runid = 89743
server='alcdata-transp.psfc.mit.edu'
tree='transp'
device='CMOD'
time=1.250
avgtim = 0.050
nzones=None
# ====================

statefile = 'state{}_{}x{}'.format(runid, int(np.floor(time)), int(np.floor(time*1e3-np.floor(time)*1e3)))

trxpl_in = []

trxpl_in.append('M S %s' % server)  # Set the server
trxpl_in.append('T %s ' % tree)  # Set the tree
trxpl_in.append('D S %s ' % runid)  # Set the 'shot' as the runid, will convert A->01

# not sure....
Bt_CW = '0'
Ip_CW = '0'

trxpl_in.append('A')#accept
trxpl_in.append(str(time))  # Set time
trxpl_in.append(str(avgtim))  # Set avgerating time +/-
trxpl_in.append('151') # Set number of theta points for 2D splines (equil)
trxpl_in.append('101')  # Set number of R points for cartesian grid (equil)
trxpl_in.append('101')  # Set number of Z points for cartesian grid (equil)
trxpl_in.append(Bt_CW)  # Set direction of torodial field CCW=1, CW = -1, READ = 0
trxpl_in.append(Ip_CW)  # Set direction of plasma current CCW=1, CW = -1, READ = 0
trxpl_in.append('Y')  # Accept these grid/field settings
trxpl_in.append('X')  # Extract plasma state
if nzones is not None:  # If re-zoning
    trxpl_in.append('N %d' % int(nzones))
trxpl_in.append('H')  # Exract "heavy" with equilibrium
trxpl_in.append('W')  # Write file
trxpl_in.append(statefile)  # Give filename
trxpl_in.append('Q')  # Exit from extraction
trxpl_in.append('Q')  # Exit run options
trxpl_in.append('Q')  # Exit trxpl

# write to ascii file
trxpl_in_sep = np.concatenate([x.split() for x in trxpl_in])
open('trxpl.in', 'w').write('\n'.join(trxpl_in_sep))

# Inputs
inputs = ['trxpl.in']

# Outputs
outputs = ['{}.cdf'.format(statefile),'{}.geq'.format(statefile)]

#os.system(' /proc/sys/kernel/hostname | ssh sciortino@cmodws84.psfc.mit.edu cat ')
#os.system('cat trxpl.in | ssh sciortino@cmodws84.psfc.mit.edu trxpl ')


# ============
# create statefiles on cmodws84
ssh = subprocess.Popen(["ssh", "sciortino@cmodws84.psfc.mit.edu", 'cd ~/TGLF_scan; cat trxpl.in | trxpl'], 
                       shell=False, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)

result = ssh.stdout.readlines()

if result == []:
    error = ssh.stderr.readlines()
    print >>sys.stderr, "ERROR: %s"%error
else:
    print result


if socket.gethostname=='eofe7.cm.cluster':
    # if running on engaging, fetch back statefiles from cmodws84
    ssh = subprocess.Popen(["ssh", "sciortino@cmodws84.psfc.mit.edu", 
                            'cd ~/TGLF_scan; scp state* sciortino@eofe7.mit.edu:~/TGLF_scan'], 
                           shell=False, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)

    result = ssh.stdout.readlines()

    if result == []:
        error = ssh.stderr.readlines()
        print >>sys.stderr, "ERROR: %s"%error
    else:
        print result

    # Run profiles_gen without EQDSK file for the moment
    os.system("cd ~/TGLF_scan; profiles_gen -i {}.cdf -g {}.geq".format(statefile, statefile))

else:
    # if on MFEWS, read in gEQDSK file (EFIT output)
    #with open(statefile+'.geq','r') as f:
    #   EQDSK = f.read().splitlines()
    
    from omfit_eqdsk import OMFITgeqdsk
    gEQDSK = OMFITgeqdsk('./{}.geq'.format(statefile), forceFindSeparatrix=False)

    ipccw = int(math.copysign(1,gEQDSK['CURRENT']))
    btccw = int(math.copysign(1,gEQDSK['BCENTR']))

    # Run profiles_gen on mfews02 with EQDSK file
    command = 'cd ~/TGLF_scan; profiles_gen -i {}.cdf -g {}.geq -ipccw {} -btccw {}'.format(
        statefile, statefile,ipccw, btccw)

    ssh = subprocess.Popen(["ssh", "sciortino@mfews02.psfc.mit.edu",command ], 
                           shell=False, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
    
    result = ssh.stdout.readlines()
    
    if result == []:
        error = ssh.stderr.readlines()
        print >>sys.stderr, "ERROR: %s"%error
    else:
        print result

    # send statefiles
    os.system('scp state* sciortino@eofe7.mit.edu:~/TGLF_scan')
    
    # send input.profile files to engaging
    os.system('scp input.profiles* sciortino@eofe7.mit.edu:~/TGLF_scan')

    # send out.gato files
    os.system('scp out.gato* sciortino@eofe7.mit.edu:~/TGLF_scan')

    


# ===================
#
#
#
# ===================
## Create TGLF inputs at 10 radial locations through TGYRO
for i in range(10):
    if not os.path.exists('TGLF%d'%i):
        os.mkdir('TGLF%d'%i)
    #os.system('cp input.tglf TGLF%d/input.tglf'%i)

# now run TGYRO in these directories
os.system('tgyro -n 10 -e .')


'''
radial_grid = map(float,map(str,np.linspace(.01,1,100)))

TGYRO_inputs = {}

#TGYRO needs at least 2 radial locations to run
TGYRO_inputs['TGYRO_RMIN'] = min(radial_grid)
TGYRO_inputs['TGYRO_RMAX'] = max(radial_grid)
#do -1 iterations (i.e. no evolution, since OMFIT will run TGYRO in test mode)
TGYRO_inputs['TGYRO_RELAX_ITERATIONS'] = -1
TGYRO_inputs['TGYRO_ITERATION_METHOD'] = 1

#use original profiles
TGYRO_inputs['LOC_LOCK_PROFILE_FLAG'] = 1
#dump TGLF input file (this is what we are after)
TGYRO_inputs['TGYRO_TGLF_DUMP_FLAG'] = 1
TGYRO_inputs['TGYRO_TGLF_REVISION'] = 0
#use exact equilibrium
TGYRO_inputs['LOC_NUM_EQUIL_FLAG'] = 0
#need to evolve something even if only 0 steps are taken # This also fills in non-primary ion temperature
TGYRO_inputs['LOC_TE_FEEDBACK_FLAG'] = 1
TGYRO_inputs['LOC_TI_FEEDBACK_FLAG'] = 1
TGYRO_inputs['TGYRO_DEN_METHOD0'] = 1
TGYRO_inputs['LOC_ER_FEEDBACK_FLAG'] = 0

#set number of ions to 10, and TGYRO['SCRIPTS']['update_ion_data'] will correct it
#TGYRO['SETTINGS']['PHYSICS']['LOC_N_ION_USER'] = 10

#TGYRO['SETTINGS']['SETUP']['n_cpu_rad'] = 1
'''
