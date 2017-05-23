
# coding: utf-8

# # F77 - Computer und Datenverarbeitung
# 
# ## Versuchsteil A

# $$\nu_n$$

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
# import uncertainties as unc
# import uncertainties.unumpy as unp
from helper import *

plt.style.use('dark_background')

str_f = 'f / Hz'
str_A = 'A / V'
str_dA = 'dA / V'


def lorentz(omega, gamma, f_0, omega_0):    
    return f_0 / np.sqrt((omega_0**2-omega**2)**2+gamma**2*omega**2)


# In[2]:

df_a = pd.read_csv('data/a.dat', decimal=',', delimiter='\t')

df_a1 = df_a.ix[df_a[str_f] < 400]
df_a2 = df_a.ix[df_a[str_f] > 400]
df_a2 = df_a2.ix[df_a2[str_f] < 3e3]
df_a3 = df_a.ix[df_a[str_f] > 3e3]


# In[3]:

for it, e in enumerate([df_a1, df_a2, df_a3]):
    lmod = Model(lorentz)
    
    if it == 0:
        x = e[str_f][e[str_f] > 260]
        y = e[str_A][e[str_f] > 260]
        dy = e[str_dA][e[str_f] > 260]
    elif it == 1:
        x = e[str_f][e[str_f] > 1755]
        y = e[str_A][e[str_f] > 1755]
        dy = e[str_dA][e[str_f] > 1755]
    else:
        x = e[str_f]
        y = e[str_A]
        dy = e[str_dA]
    
    fitdata = lmod.fit(
        y, 
        omega=x, 
        gamma=2, 
        f_0=np.max(y), 
        omega_0=(np.max(x)+np.min(x))/2, 
        weights=1/dy
    )
    
    pd.DataFrame()
    
    otl = OutputTable()
    otl.add("\omega_0", fitdata.params['omega_0'].value, fitdata.params['omega_0'].stderr, "Hz", aftercomma=3)
    otl.add("f_0", fitdata.params['f_0'].value, fitdata.params['f_0'].stderr, "V", aftercomma=3)
    otl.add("\gamma", fitdata.params['gamma'].value, fitdata.params['gamma'].stderr)
    otl.add("\chi^2_{red}", fitdata.redchi, aftercomma=3)

    otl.print()
    
    otl.empty()

    plt.figure(figsize=(12.8,7.2))
    
    plt.errorbar(e[str_f], e[str_A]*1e6, yerr=e[str_dA]*1e6, fmt='.')
    plt.plot(x, fitdata.best_fit*1e6, 'r-')    
    
    
    plt.xlim((np.min(e[str_f])*.99, np.max(e[str_f])*1.01))
    plt.ylim((np.min(y*1e6), np.max(y*1e6)))
    
    plt.xlabel(str_f)
    plt.ylabel('A / $\mu$V')
    
    plt.plot()


# In[ ]:



