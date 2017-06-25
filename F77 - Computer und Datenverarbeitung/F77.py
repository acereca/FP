#! /usr/bin/python3
# coding=utf8

# # F77 - Computer und Datenverarbeitung
#
# ## Versuchsteil A

# setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from lmfit import Model
import uncertainties as unc
import uncertainties.unumpy as unp
from helper import *

plt.style.use('bmh')

str_f = 'f / Hz'
str_A = 'A / V'
str_dA = 'dA / V'

def lorentz(omega, gamma, f_0, omega_0):
    return f_0 / np.sqrt((omega_0**2-omega**2)**2+gamma**2*omega**2)


# data import
df_a = pd.read_csv('data/a.dat', decimal=',', delimiter='\t')

# korrektur für falschen output
for row in df_a.index:
    if row > 0 and row < max(df_a.index) and df_a[str_f][row] == df_a[str_f][row+1]:
        for i in range(5,1,-1):
            if row+i <= max(df_a.index) and df_a[str_f][row+i] == df_a[str_f][row]:
                df_a[str_f][row+i] += .2*i

# trennung in 3 frequenzbereiche
df_a1 = df_a.ix[df_a[str_f] < 400]
df_a2 = df_a.ix[df_a[str_f] > 400]
df_a2 = df_a2.ix[df_a2[str_f] < 3e3]
df_a3 = df_a.ix[df_a[str_f] > 3e3]

# least square linear model fit
lmod = Model(lorentz)
for it, e in enumerate([df_a1, df_a2, df_a3]):

    # reduziere zu fittende daten
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

    # generiere output
    otl = OutputTable("Teil A - lorentz Fit Nr." + str(it+1))
    otl.add("\omega_{0," + str(it+1) + "}", fitdata.params['omega_0'].value, fitdata.params['omega_0'].stderr, "Hz", aftercomma=3)
    otl.add("A_{0," + str(it+1) + "}", fitdata.params['f_0'].value, fitdata.params['f_0'].stderr, "V", aftercomma=3)
    otl.add("\gamma_" + str(it+1), fitdata.params['gamma'].value, fitdata.params['gamma'].stderr)
    otl.add("\chi^2_{" + str(it+1) + "}", fitdata.chisqr)
    otl.add("\chi^2_{red," + str(it+1) + "}", fitdata.redchi, aftercomma=3)

    otl.save('a_' + str(it+1) + '.tex')

    otl.empty()

    # plot gen 1
    plt.figure(figsize=(11.7,8.3))
    plt.errorbar(
        e[str_f],
        e[str_A] * 1e6,
        yerr=e[str_dA] * 1e6,
        fmt='.',
        c='#00000033',
        label='Messdaten'
    )
    plt.plot(x, fitdata.best_fit * 1e6, 'r-')
    d_f0h = np.sqrt((np.sqrt(fitdata.params['gamma'].value**4+8)-fitdata.params['gamma'].value**2)/2)
    w0 = np.abs(fitdata.params['omega_0'].value)
    f0 = fitdata.params['f_0'].value*(2*np.pi)**2
    print(w0-d_f0h, w0+d_f0h, 2*d_f0h, f0)

    plt.plot([w0- d_f0h, w0 + d_f0h],[f0, f0], 'b-')

    plt.xlim((np.min(e[str_f]), np.max(e[str_f])))
    plt.ylim((np.min(y * 1e6), np.max(y * 1e6)))

    plt.title(r'Frequenzgang der Schwingungsamplitude eines "Vibrating Reed" um $\nu_'
                       + str(it) + '$')
    plt.xlabel(str_f)

    plt.ylabel('A / $\mu$V')
    plt.tight_layout()
    plt.savefig('a_1' + str(it + 1) + '.png')

    # plot gen 2
    f, axarr = plt.subplots(2,1, sharex='col', figsize=(11.7,8.3))
    axarr[0].errorbar(
        e[str_f],
        e[str_A]*1e6,
        yerr=e[str_dA]*1e6,
        fmt='.',
        c='#00000033',
        label='Messdaten'
    )
    axarr[0].plot(x, fitdata.best_fit*1e6, 'r-')


    axarr[1].errorbar(
        e[str_f],
        e['phi / rad'],
        yerr=e['dphi/rad'],
        fmt='.',
        c='#00000033'
    )

    axarr[0].set_xlim((np.min(e[str_f]), np.max(e[str_f])))
    axarr[0].set_ylim((np.min(y*1e6), np.max(y*1e6)))
    axarr[1].set_ylim((-1,3))

    axarr[1].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    axarr[1].yaxis.set_major_locator(tck.MultipleLocator(base=1.0))

    axarr[0].set_ylabel('A / $\mu$V')
    axarr[1].set_ylabel('$\phi$')

    axarr[0].set_title(r'Frequenzgang der Schwingungsamplitude eines "Vibrating Reed" um $\nu_'
              + str(it) + '$')
    axarr[1].set_title(r'Frequenzgang der Schwingungsphase')

    f.subplots_adjust(wspace=.01)
    f.text(0.5, 0.04, 'f / Hz', ha='center')

    #plt.xlabel(str_f)

    #plt.ylabel('A / $\mu$V')
    plt.tight_layout(rect=(0,0.05,1,1))
    plt.savefig('a_2' + str(it+1) + '.png')

    # plot gen 3
    f, axarr = plt.subplots(2,1, sharex='col', figsize=(11.7,8.3))
    axarr[0].errorbar(
        e[str_f],
        e[str_A]*1e6,
        yerr=e[str_dA]*1e6,
        fmt='.',
        c='#00000033',
        label='Messdaten'
    )
    axarr[0].plot(x, fitdata.best_fit*1e6, 'r-')
    axarr[1].errorbar(
        e[str_f],
        (e[str_A]-fitdata.best_fit)/e[str_dA],
        fmt='.',
        c='#00000033'
    )
    sys_res = np.mean((e[str_A]-fitdata.best_fit)/e[str_dA])
    axarr[1].axhline(sys_res)
    axarr[1].annotate(
        '${:.2e}$'.format(sys_res),
        xy=(np.min(e[str_f])+1 ,sys_res),
        xycoords='data',
        xytext=(0, 0),
        textcoords='offset points',
        fontsize=14,
        bbox=dict(boxstyle="round",
        fc="1")
    )

    axarr[0].set_xlim((np.min(e[str_f]), np.max(e[str_f])))
    axarr[0].set_ylim((np.min(y*1e6), np.max(y*1e6)))
    axarr[0].set_ylabel('A / $\mu$V')
    axarr[1].set_ylabel('residual')

    axarr[0].set_title(r'Frequenzgang der Schwingungsamplitude eines "Vibrating Reed" um $\nu_'
              + str(it) + '$')
    axarr[1].set_title(r'Residuen')

    f.subplots_adjust(wspace=.01)
    f.text(0.5, 0.04, 'f / Hz', ha='center')

    #plt.xlabel(str_f)

    #plt.ylabel('A / $\mu$V')
    plt.tight_layout(rect=(0,0.05,1,1))
    plt.savefig('a_3' + str(it+1) + '.png')

    d_stat = np.mean(e[str_dA])
    #print(x.shape[0])
    s_sys = 1/(fitdata.chisqr/fitdata.redchi) * np.sum(((e[str_A]-fitdata.best_fit)*1e6)**2) - d_stat**2
    #print(np.sqrt(s_sys), d_stat)

## Versuchsteil B

str_f = 'f'
str_A = 'A'
str_dA = 'dA'

df_b = pd.read_csv('data/b.dat', decimal=',', delimiter='\t')


# trennung in temperaturbereiche
df_blist = []
#df_b.set_index(keys=['T'], drop=False,inplace=True)
temps=df_b['T'].unique().tolist()
temps[0], temps[1] = temps[1], temps[0]
for temp in temps:
    if temp < 62.0:
        df_blist.append(df_b[df_b['T'] == temp])


plt.cla()
f, axarr = plt.subplots(3,3, sharex='col', sharey='row', figsize=(11.7,8.3))

positions = []
dpositions = []
temps = []
dtemps = []

for it, part_df in enumerate(df_blist):

    temp = part_df['T'].unique().tolist()[0]
    dtemp = part_df['dT'].unique().tolist()[0]
    x  = np.array(part_df[str_f].tolist()[1:])
    y  = np.array(part_df[str_A].tolist()[1:])
    dy = np.array(part_df[str_dA].tolist()[1:])

    fitdata = lmod.fit(
        y,
        omega=x,
        gamma=2,
        f_0=np.max(y),
        omega_0=(np.max(x) + np.min(x)) / 2,
        weights=1 / dy
    )

    positions.append(fitdata.params['omega_0'].value)
    dpositions.append(fitdata.params['omega_0'].stderr)
    temps.append(temp+273.15)
    dtemps.append(dtemp)

    splt = axarr[int(it / 3), int(it % 3)]
    splt.errorbar(x, y*1e6, yerr=dy*1e6,fmt='.',c='#00000033')
    splt.plot(x, fitdata.best_fit*1e6, 'r-')
    splt.set_title(r"T = ${:.2fL}$ °C".format(ufloat(temp, dtemp)))
    splt.axvline(np.abs(fitdata.params['omega_0'].value))
    splt.set_xlim(275,285)

f.subplots_adjust(wspace=.01)
f.text(0.5, 0.04, 'f / Hz', ha='center')
f.text(0.04, 0.5, 'A / $\mu$V', va='center', rotation='vertical')
f.suptitle(r'Frequenzgänge um die Resonanzfrequenz $\nu_0$ für unterschiedliche Temperaturen T')
plt.savefig('b.png')

plt.clf()

lmod = Model(lambda x, m, c: m*x+c)

fitdata = lmod.fit(
        np.abs(positions),
        x=temps,
        m=-.1,
        c=300,
        weights=1/np.sqrt((np.array(positions)/np.array(dpositions))**2 + (np.array(temp)/np.array(dtemps))**2)/np.array(np.abs(positions))
    )

otl = OutputTable("Teil B - linearer Fit")
otl.add("m", fitdata.params['m'].value, fitdata.params['m'].stderr, r'\frac{Hz}{K}')
otl.add("c", fitdata.params['c'].value, fitdata.params['c'].stderr, 'Hz')
otl.add("\chi^2_{" + str(it+1) + "}", fitdata.chisqr)
otl.add("\chi^2_{red," + str(it+1) + "}", fitdata.redchi, aftercomma=3)

otl.save('b_2.tex')

plt.figure(figsize=(11.7,8.3))
plt.errorbar(temps, np.abs(positions), yerr=dpositions, xerr=dtemps, fmt='.')
plt.plot(temps, fitdata.best_fit, 'r-')
plt.title("Resonanzfrequenz in Abhängigkeit der Temperatur")
plt.xlabel('T / K')
plt.ylabel('$\omega_0$ / Hz')
plt.tight_layout()
plt.savefig('b_2.png')
