import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import peakutils as pu
import numpy as np

plt.style.use('bmh')

fig, axarr = plt.subplots(3,3, sharey="row")

for c,d in enumerate(['10A', '12A', '13A']):
    for i,f in enumerate(['', '_pi', '_sig']):


        data = pd.read_table("data/trans/"+ d +"/" + d + f, header=None, names=['px','int','d'])
        img = plt.imread("data/trans/"+ d + "/" + d + f + "_c.png")
        img = img[:,:,0]

        peaks = pu.indexes(data['int'], min_dist=15, thres=0.02)

        offset = min(data['int'])

        axarr[i,c].plot(data['px'], data['int']-offset, label=d+f)
        axarr[i,c].errorbar(data['px'][peaks], data['int'][peaks]-offset+1e5, fmt='v', label=d+f +"-Peaks")

        #x0,x1 = axarr[i,c].get_xlim()
        y0,y1 = axarr[i,c].get_ylim()

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
plt.show()
