import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
import matplotlib.pylab as plt

# YACORA Hydrogen PEC
class YH_PEC:
    def __init__(self,filename,transition=""):
        self.filename = filename
        self.transition = transition
        self._read()

    def _read(self):
        self.T_e,self.n_e,self.n_n1,self.pop = np.loadtxt(self.filename,skiprows=27).T
        _,index = np.unique(self.pop,return_index=True)
        index = np.sort(index)
        self.T_e = self.T_e[index]
        self.n_e = self.n_e[index]
        self.pop = self.pop[index]
        for line in open(self.filename,"r").readlines():
            # Read n
            if "Filename" in line:
                self.n = int(line[36])
            # Read dimensions
            if "T_e:" in line:
                self.n_T = int(line.split(" ")[15])
            if "n_e:" in line:
                self.n_n = int(line.split(" ")[15])
        self.T_e = self.T_e.reshape((self.n_T,self.n_n))
        self.n_e = self.n_e.reshape((self.n_T,self.n_n))
        self.pop = self.pop.reshape((self.n_T,self.n_n))*self.A(self.n)

    def A(self,n):
        self.Ap = [4.4114e7, 2.2135e5, 1.2159e5, 7.1242e4, 4.3983e4, 2.8344e4,
              1.8927e4, 1.3032e4, 9.2102e3, 6.6583e3, 4.901e3, 3.6851e3,
              2.8903e3, 2.1719e3]
        return self.Ap[n-3]

f = plt.figure()
ax = f.add_subplot(111)
for d in os.listdir("./"):
    if d[0] == "H":
        pec = YH_PEC("%s/YacoraRun_PopCoeff_n3.dat"%d,transition=d)
        ax.plot(pec.T_e[:,0],pec.pop[:,0],label=pec.transition)
ax.legend()
plt.show()

pecHnH = YH_PEC("H-H/YacoraRun_PopCoeff_n3.dat")
pecHnH2 = YH_PEC("H-H2/YacoraRun_PopCoeff_n3.dat")
plt.figure()
plt.plot(pecHnH.T_e,pecHnH.pop,label="H-H")
plt.plot(pecHnH.T_e,pecHnH.pop,label="H-H2")
plt.legend()
plt.show()
