from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import matplotlib.pyplot as plt
plt.ion()

path = '/home/sciortinof/transfer/neg_triang/'
geqdsk20 = OMFITgeqdsk(path+'180520/geqdsk_180520_2500')
geqdsk26 = OMFITgeqdsk(path+'180526/geqdsk_180526_2750')
geqdsk30 = OMFITgeqdsk(path+'180530/geqdsk_180530_3800')

levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
geqdsk20.plot(only2D=True, color='r', levels=levels)
geqdsk26.plot(only2D=True, color='b', ax=plt.gca(), levels=levels)
geqdsk30.plot(only2D=True, color='g', ax=plt.gca(), levels=levels)

plt.gca().set_xlabel('R [m]')
plt.gca().set_ylabel('Z [m]')

plt.gca().plot([],[],'r-', label='180520')
plt.gca().plot([],[],'b-', label='180526')
plt.gca().plot([],[],'g-', label='180530')

plt.legend(loc='best').set_draggable(True)
