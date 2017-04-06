
# coding: utf-8

# # Versuch E01 - Grundpraktikum Elektronik

# In[98]:

get_ipython().magic('matplotlib inline')

from uncertainties import unumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

#print(plt.style.available)


# In[86]:

def plotfreq(x, y, xerr, yerr,             xlabel='x', ylabel='y', xscale='linear', yscale='linear',             title='graph', filename='tmp', style='bmh', yticks=[], xticks=[]):
    plt.figure(figsize=(15,10))
    plt.style.use(style)   
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.')
    
    plt.xscale(xscale)
    plt.yscale(yscale)
    if len(yticks) > 0:
        plt.yticks(yticks)
        plt.axes().get_yaxis().set_major_formatter(tck.ScalarFormatter())
    if len(xticks) > 0:
        plt.xticks(xticks)
        plt.axes().get_xaxis().set_major_formatter(tck.ScalarFormatter())
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename+'.png')


# ## V1 - Emitterschaltung

# In[87]:

data = np.genfromtxt('1-freq.dat', dtype=float, skip_header=0, delimiter=',')[2:]

# Inbetriebnahme und Vermessung
init = np.array([6.79,0.615,7.94,0.997,0.387,7.20])
init*.05/100+.001


# In[107]:

plotfreq(data[:,0], data[:,4], xerr=data[:,1], yerr=data[:,5],        xlabel='f / Hz', ylabel='$V_{pp,out}$ / V', xscale='log', yscale='log',        title='Frequenzgang der Ausgangsspannung $V_{pp,out}$',        filename='1-freqU')
plt.clf()


# In[108]:

vppin = unumpy.uarray(data[:,2],data[:,3])
vppout = unumpy.uarray(data[:,4],data[:,5])

verst = vppout/vppin*1000

plotfreq(data[:,0], [verst[i].n for i in range(len(verst))],         xerr=data[:,1], yerr=[verst[i].s for i in range(len(verst))],         xlabel='f / Hz', ylabel='$V_U$', xscale='log', yscale='log',         title='Frequenzgang der Verstärkung $V_U$', filename='1-freqV')
plt.clf()
#pd.DataFrame(verst)


# ## V2 - Kollektorschaltung

# In[109]:

init = np.array([7.69,0.64,6.78,8.31])
init*.05/100+.001


# In[110]:

data2 = np.genfromtxt("2-freq.dat", dtype=float, skip_header=0, delimiter=',')[2:]

plotfreq(data2[:,0], data2[:,4], xerr=data2[:,1], yerr=data2[:,5],         xscale="log", yscale="log", xlabel="f / Hz", ylabel="$V_{pp,out} / mV$",         title="Frequenzgang der Ausgangsspannung $V_{pp,out}$",         filename='2-freqU', yticks=[450, 500, 550, 600])
plt.clf()


# In[111]:

vppout = unumpy.uarray(data2[:,4],data2[:,5])
vppin = unumpy.uarray(data2[:,2],data2[:,3])

verst = vppout/vppin

plotfreq(data2[:,0], [verst[i].n for i in range(len(verst))],         yerr=[verst[i].s for i in range(len(verst))], xerr=data2[:,1],         xscale="log", yscale="log", yticks=[.95+i*.05 for i in range(6)],         xlabel="f / Hz", ylabel="$V_U$", title="Frequenzgang der Verstärkung $V_U$",         filename='2-freqV')
plt.clf()
#pd.DataFrame(verst)


# In[ ]:




# In[ ]:




# In[ ]:



