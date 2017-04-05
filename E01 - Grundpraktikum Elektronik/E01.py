
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

from uncertainties import unumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck


# ## V2

# In[24]:

data = np.genfromtxt('1-freq.dat', dtype=float, skip_header=0, delimiter=',')[2:]

# Inbetriebnahme und Vermessung
init = np.array([6.79,0.615,7.94,0.997,0.387,7.20])
init*.05/100+.001


# In[6]:

ax = plt.subplot(111)
ax.errorbar(data[:,0], data[:,4], xerr=data[:,1], yerr=data[:,5], fmt='.')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("f / Hz")
ax.set_ylabel("$V_{pp,out}$ / V")
#ax.set_label("Frequenzgang der Ausgangsspannung $V_{pp,out}$")
ax.set_title("Frequenzgang der Ausgangsspannung $V_{pp,out}$")
plt.show()


# In[25]:

vppin = unumpy.uarray(data[:,2],data[:,3])
vppout = unumpy.uarray(data[:,4],data[:,5])

verst = vppout/vppin*1000

ax2 = plt.subplot(111)
ax2.errorbar(data[:,0], [verst[i].n for i in range(len(verst))], xerr=data[:,1], yerr=[verst[i].s for i in range(len(verst))], fmt='.')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("f / Hz")
ax2.set_yticks([6,8,10,12,14,16,18,20])
ax2.get_yaxis().set_major_formatter(tck.ScalarFormatter())
ax2.set_ylabel("$V_U$")
ax2.set_title("Frequenzgang der Verstärkung $V_U$")

plt.show()
pd.DataFrame(verst)


# ## V2

# In[27]:

init = np.array([7.69,0.64,6.78,8.31])
init*.05/100+.001


# In[14]:

data2 = np.genfromtxt("2-freq.dat", dtype=float, skip_header=0, delimiter=',')[2:]
ax = plt.subplot(111)
ax.errorbar(data2[:,0], data2[:,4], xerr=data2[:,1], yerr=data2[:,5], fmt='.')
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("f / Hz")
ax.set_ylabel("$V_{pp,out}$ / V")
ax.set_title("Frequenzgang der Ausgangsspannung $V_{pp,out}$")
plt.show()


# In[26]:

vppout = unumpy.uarray(data2[:,4],data2[:,5])
vppin = unumpy.uarray(data2[:,2],data2[:,3])

verst = vppout/vppin

ax2 = plt.subplot(111)
ax2.errorbar(data2[:,0], [verst[i].n for i in range(len(verst))], yerr=[verst[i].s for i in range(len(verst))], xerr=data2[:,1], fmt='.')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_yticks([.95,1, 1.05, 1.1, 1.15])
ax2.set_xticklabels([])
ax2.get_yaxis().set_major_formatter(tck.ScalarFormatter())
ax2.set_xlabel("f / Hz")
ax2.set_ylabel("$V_U$")
ax2.set_title("Frequenzgang der Verstärkung $V_U$")

plt.show()
pd.DataFrame(verst)


# In[ ]:




# In[ ]:



