import matplotlib.pyplot as plt
import sys
import math
import numpy as np

nshift=10
nthread_pattern=6

shift=[10,12,14,16,18,20,22,24,26,28]
nthread=[1,2,4,8,16,32]

def read_data(ishift, ithread, bw_D2D, bw_D2P_1, bw_D2P_2, data_size):
    filename = 'pmem_io_single_'+str(shift[ishift])+'_'+str(nthread[ithread])+'.dat'

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

    file.close()


data_size = np.zeros(nshift)
bw_D2D = np.zeros((nshift, nthread_pattern),dtype='float')
bw_D2P_1 = np.zeros((nshift, nthread_pattern))
bw_D2P_2 = np.zeros((nshift, nthread_pattern))


for ishift in range(nshift):
    for ithread in range(nthread_pattern):
#        print(shift[ishift], nthread[ithread])
        read_data(ishift, ithread, bw_D2D, bw_D2P_1, bw_D2P_2, data_size)

print(bw_D2D)
print(bw_D2P_1)
print(bw_D2P_2)
print(data_size)

fig = plt.figure(dpi=300)

label = ['nthread=1','nthread=2','nthread=4','nthread=8','nthread=16','nthread=32']

ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
for ithread in range(nthread_pattern):
    ax.plot(data_size[7:], bw_D2P_1[7:,ithread],label=label[ithread])

ax.legend()
fig.savefig('hoge.pdf',bbox_inches='tight')
