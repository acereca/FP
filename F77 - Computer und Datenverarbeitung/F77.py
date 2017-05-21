
# coding: utf-8

# # F77 - Computer und Datenverarbeitung
# 
# ## Versuchsteil A

# $$\nu_n$$

# In[17]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

get_ipython().magic('matplotlib inline')

str_f = 'f / Hz'
str_A = 'A / V'
str_dA = 'dA / V'


# In[18]:

df_a = pd.read_csv('data/a.dat', decimal=',', delimiter='\t')

df_a1 = df_a.ix[df_a[str_f] < 400]
df_a2 = df_a.ix[df_a[str_f] > 400]
df_a2 = df_a2.ix[df_a2[str_f] < 3e3]
df_a3 = df_a.ix[df_a[str_f] > 3e3]


# In[21]:

for e in [df_a1, df_a2, df_a3]:
    
    # TODO: fit

    plt.figure(figsize=(12.8,7.2))
    plt.errorbar(e[str_f], e[str_A]*1e6, yerr=e[str_dA], fmt='.')
    plt.xlim((np.min(e[str_f]), np.max(e[str_f])))
    plt.ylim((np.min(e[str_A]*1e6), np.max(e[str_A]*1e6)))
    
    plt.xlabel(str_f)
    plt.ylabel('A / $\mu$V')
    
    plt.plot()


# In[ ]:



