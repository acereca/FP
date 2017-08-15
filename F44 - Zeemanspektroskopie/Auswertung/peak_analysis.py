import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.optimize as opt
import peakutils as pu
import numpy as np

plt.style.use('bmh')

fig, axarr = plt.subplots(3,3, sharey="row", figsize=(19.2,10.8))

for c,d in enumerate(['10A', '12A', '13A']):
    for i,f in enumerate(['', '_pi', '_sig']):

        data = pd.read_table("data/trans/"+d +"/" + d + f, header=None, names=['px','int','d'])
        img = plt.imread("data/trans/"+ d + "/" + d + f + "_c.png")
        img = img[:,:,0]

        peaks = pu.indexes(data['int'], min_dist=15, thres=0.02)


        offset = min(data['int'])

        axarr[i,c].plot(data['px'], data['int']-offset, label=d+f)

        #y0,y1 = axarr[i,c].get_ylim()
        y0,y1 = min(data['int']-offset), max(data['int']-offset)+2e6

        #select and plot peak pos
        cy = .75*y1
        m = - cy / max(data['px'])

        #tarr = abs(data['int'][peaks]-offset-(m*data['px'][peaks]+cy)) < (m*(data['px'][peaks])+cy)/1.5
        exclude_arr = [
            [ [0,4, 8, 12, 17, 19,-1], [0,7,11,14,17,19,-1], [0, 10, 13, 17, -1] ],
            [ [0,2,3,5,6,7,9,10,-1],   [0,1,3,4,6,7],        [0,1,3,4,5,7,8]     ],
            [ [0,2,4,6,9,-1],          [0,2,4,6,9,-1],       [-1,-4,0,1,3,5,8]   ]
        ]

        peaks_x = np.ma.array(data['px'][peaks].tolist(), mask=False)
        peaks_y = np.ma.array(data['int'][peaks].tolist(), mask=False)

        peaks_x.mask[exclude_arr[i][c]] = True
        peaks_y.mask[exclude_arr[i][c]] = True

        if i == 0:
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask][::3], peaks_y[~peaks_y.mask][::3]-offset+1e5, fmt='v', label=d+f +"-Peaks", color='green')
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask][1::3], peaks_y[~peaks_y.mask][1::3]-offset+1e5, fmt='v', label=d+f +"-Peaks", color='red')
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask][2::3], peaks_y[~peaks_y.mask][2::3]-offset+1e5, fmt='v', label=d+f +"-Peaks", color='yellow')

        elif i== 1:
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask], peaks_y[~peaks_y.mask]-offset+1e5, fmt='v', label=d+f +"-Peaks")

        elif i==2:
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask][::2], peaks_y[~peaks_y.mask][::2]-offset+1e5, fmt='v', label=d+f +"-Peaks", color='green')
            axarr[i,c].errorbar(peaks_x[~peaks_x.mask][1::2], peaks_y[~peaks_y.mask][1::2]-offset+1e5, fmt='v', label=d+f +"-Peaks", color='yellow')

        #rotate and colorcorrect image
        rotated = nd.rotate(img, 1.5)
        imgarr = axarr[i,c].imshow(rotated, extent=[min(data['px']),max(data['px']),y0,y1], aspect='auto')
        imgarr.set_cmap('binary')


        if i == 2:
            axarr[i,c].set_xlabel('Pixel')
        if c == 0:
            axarr[i,c].set_ylabel(['Intensität der Linien', r"Intensität der $\pi$-Linien", r"Intensität der $\sigma$-Linien"][i])
        if i == 0:
            axarr[i,c].set_title(d)
        #axarr[i,c].legend()
plt.savefig('peaks.png')
