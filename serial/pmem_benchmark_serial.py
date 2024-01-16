import matplotlib.pyplot as plt
import sys
import math
import numpy as np


shift=[10,12,14,16,18,20,22,24,26,28]
nthread=[16, 32]

nshift=len(shift)
nthread_pattern=len(nthread)

def read_data(ishift, ithread, bw_D2D, bw_D2P_1, bw_D2P_2, bw_P2P_1, bw_P2P_2, data_size):
    filename = 'pmem_io_serial_'+str(shift[ishift])+'_'+str(nthread[ithread])+'.dat'

    print(filename)
    file =  open(filename,'r')

    nline = 0
    bw = 0.0
    for line in file:
        data = line.split()
        if data[0] != '#':
            bw = data[1]
            data_size[ishift] = data[0]
            nline += 1
        if nline==1:
            bw_D2D[ishift,ithread] = bw
        elif nline==2:
            bw_D2P_1[ishift,ithread] = bw
        elif nline==3:
            bw_D2P_2[ishift,ithread] = bw
        elif nline==4:
            bw_P2P_1[ishift,ithread] = bw
        elif nline==5:
            bw_P2P_2[ishift,ithread] = bw
            
    file.close()


data_size = np.zeros(nshift)
bw_D2D = np.zeros((nshift, nthread_pattern),dtype='float')
bw_D2P_1 = np.zeros((nshift, nthread_pattern))
bw_D2P_2 = np.zeros((nshift, nthread_pattern))
bw_P2P_1 = np.zeros((nshift, nthread_pattern))
bw_P2P_2 = np.zeros((nshift, nthread_pattern))


for ishift in range(nshift):
    for ithread in range(nthread_pattern):
        read_data(ishift, ithread, bw_D2D, bw_D2P_1, bw_D2P_2, bw_P2P_1, bw_P2P_2, data_size)

print(bw_D2D)
print(bw_D2P_1)
print(bw_D2P_2)
print(bw_P2P_1)
print(bw_P2P_2)
print(data_size)

fig = plt.figure(dpi=100)

label_D2D   = ['D2D (w/  memcpy)/n=16','D2D (w/  memcpy)/n=32']
label_D2P_1 = ['D2P (w/  memcpy)/n=16','D2P (w/  memcpy)/n=32']
label_D2P_2 = ['D2P (w/o memcpy)/n=16','D2P (w/o memcpy)/n=32']
label_P2P_1 = ['P2P (w/  memcpy)/n=16','P2P (w/  memcpy)/n=32']
label_P2P_2 = ['P2P (w/o memcpy)/n=16','P2P (w/o memcpy)/n=32']

ls = ['solid','dotted']

ax1 = fig.add_axes((0.1, 0.1, 0.9, 0.9))
ax2 = fig.add_axes((1.1, 0.1, 0.9, 0.9))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')
ax2.set_yscale('log')

ax1.set_ylim(3.0, 1100.0)
ax2.set_ylim(3.0, 1100.0)

ax1.set_xlabel('data size [GiB]')
ax1.set_ylabel('bandwidth [GiB/s]')

ax2.set_xlabel('data size [GiB]')
ax2.set_ylabel('bandwidth [GiB/s]')

for ithread in range(nthread_pattern):
    ax1.plot(data_size[2:], bw_D2D[2:,ithread],label=label_D2D[ithread], lw=0.5, color='red', ls=ls[ithread])
    ax1.plot(data_size[2:], bw_D2P_1[2:,ithread],label=label_D2P_1[ithread], color='blue', ls=ls[ithread])
    ax1.plot(data_size[2:], bw_D2P_2[2:,ithread],label=label_D2P_2[ithread], color='green', ls=ls[ithread])
    ax2.plot(data_size[2:], bw_D2D[2:,ithread],label=label_D2D[ithread], lw=0.5, color='red', ls=ls[ithread])
    ax2.plot(data_size[2:], bw_P2P_1[2:,ithread],label=label_P2P_1[ithread], color='blue', ls=ls[ithread])
    ax2.plot(data_size[2:], bw_P2P_2[2:,ithread],label=label_P2P_2[ithread], color='green', ls=ls[ithread])

ax1.legend()
ax2.legend()
fig.savefig('hoge.pdf',bbox_inches='tight')
