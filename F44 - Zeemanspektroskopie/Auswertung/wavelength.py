import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import scipy.ndimage as nd
import scipy.optimize as opt
import peakutils as pu
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from helper import mpl_annotate

plt.style.use('bmh')
plt.figure(figsize=(8, 4))

data_ne = pd.read_table(
    'data/part2/ne_2_colorcorr_evaluation',
    header=None,
    names=['px','int','d']
)
data_cd = pd.read_table(
    'data/part2/cd_2_colorcorr_evaluation',
    header=None,
    names=['px','int','d']
)

data_cd['int'] = data_cd['int'] - np.min(data_cd['int'])
filter_arr = (data_ne['px'] > 450) & (data_ne['px'] < 700)
filter_arr2 = (data_cd['px'] > 450) & (data_cd['px'] < 700)

plt.plot(
    data_ne['px'][filter_arr],
    data_ne['int'][filter_arr],
    label = 'Neon Lampen Spektrum'
)
plt.plot(
    data_cd['px'][filter_arr2],
    data_cd['int'][filter_arr2],
    label = 'Cadmium Lampen Spektrum'
)

plt.savefig('wavelength_analysis/wl.png')



# new plot

plt.clf()
plt.figure(figsize=(19.2,10.5))
peaks = [470, 505, 520, 600, 620, 670]
fitted_peaks = []

for it, peak in enumerate(peaks):
    xdata = np.arange(peak-8, peak+8)
    x_range = (data_ne['px'] >= xdata[0]) & (data_ne['px'] <= xdata[-1])
    #axarr[it].plot(xdata,data_ne['int'][x_range])

    fitfunc = lambda x, a, mu, sig: a*np.exp(-.5*(x-mu)**2/sig**2)
    sig0 = 4

    pfinal, pcov = opt.curve_fit(
        fitfunc,
        data_ne['px'][x_range],
        data_ne['int'][x_range],
        p0=[max(data_ne['int'][x_range]), peak, sig0],
        sigma=[.05 for i in data_ne['int'][x_range]]
    )

    xdata = np.arange(peak-8, peak+8, .1)
    #axarr[it].plot(xdata, fitfunc(xdata, *pfinal))

    fitted_peaks.append(unc.ufloat(pfinal[1], np.sqrt(pcov[1,1]**2 + pfinal[2]**2)))

#print(unp.std_devs(fitted_peaks))

given_wls = [633.443, 638.299, 640.225, 650.623, 653.288, 659.895]
plt.errorbar(
    unp.nominal_values(fitted_peaks),
    given_wls,
    xerr=unp.std_devs(fitted_peaks),
    fmt='.',
    label= 'beobachtete Ne-Spektrum Linien',
    lw=1
)

fitfunc = lambda x, m, c: x*m+c
pfinal, pcov = opt.curve_fit(
    fitfunc,
    unp.nominal_values(fitted_peaks),
    given_wls,
    p0=[.1, 560],
    sigma=unp.std_devs(fitted_peaks)
)

xdata = np.arange(
    min(unp.nominal_values(fitted_peaks))-5,
    max(unp.nominal_values(fitted_peaks))+5
)
plt.plot(xdata, fitfunc(xdata, *pfinal), label='Gefittete Gerade')

fitfunc2 = lambda x, a, mu, sig: a*np.exp(-.5*(x-mu)**2/sig**2)
pfinal2, pcov2 = opt.curve_fit(
    fitfunc2,
    data_cd['px'][filter_arr2],
    data_cd['int'][filter_arr2],
    p0=[max(data_cd['int'][filter_arr2]), 550, 10],
    sigma=[.05 for i in data_ne['int'][filter_arr2]]
)

cd_pos = unc.ufloat(pfinal2[1], pfinal2[2])
cd_wl = fitfunc(cd_pos, *pfinal)

print('lambda_cd = {:.3f} nm'.format(cd_wl))
print('pos_cd = {:.3f} px'.format(cd_pos))

plt.plot(
    [cd_pos.n, cd_pos.n],
    [min(fitfunc(xdata, *pfinal)), cd_wl.n],
    '--',
    color='#aaaaaa'
)

plt.plot(
    [min(xdata), cd_pos.n],
    [fitfunc(cd_pos.n, *pfinal), cd_wl.n],
    '--',
    color='#aaaaaa'
)

plt.errorbar(
    [cd_pos.n],
    [cd_wl.n],
    xerr=cd_pos.s,
    yerr=cd_wl.s,
    fmt='.',
    color='g',
    label='Cd-Linie',
    zorder = 10,
    lw=1
)

mpl_annotate(plt, '$x_{cd} = $' + '${:.1fL}$ px'.format(cd_pos), (543, 635))
mpl_annotate(plt, '$\lambda_{cd} = $' + '${:.1fL}$ nm'.format(cd_wl), (475, 643.5) )

plt.ylim([min(fitfunc(xdata, *pfinal)), max(fitfunc(xdata, *pfinal))])
plt.xlim([min(xdata), max(xdata)])
plt.xlabel('Position / px')
plt.ylabel('Wellenlänge / nm')
plt.title('Wellenlängen Kalibrierung')

filter_arr = (data_cd['px'] > 600) & (data_cd['px'] < 650)
fitfunc2 = lambda x, a, mu, sig: a*np.exp(-.5*(x-mu)**2/sig**2)
pfinal2, pcov2 = opt.curve_fit(
    fitfunc2,
    data_cd['px'][filter_arr],
    data_cd['int'][filter_arr],
    p0=[max(data_cd['int'][filter_arr]), 550, 10],
    sigma=[.05 for i in data_ne['int'][filter_arr]]
)

uk_pos = unc.ufloat(pfinal2[1], pfinal2[2])
uk_wl = fitfunc(uk_pos, *pfinal)

plt.errorbar(
    [uk_pos.n],
    [uk_wl.n],
    xerr = uk_pos.s,
    yerr = uk_wl.s,
    fmt='.',
    color='orange',
    label = 'unbekannte Linie',
    zorder = 10,
    lw=1
)

plt.plot(
    [uk_pos.n, uk_pos.n],
    [min(fitfunc(xdata, *pfinal)), uk_wl.n],
    '--',
    color='#aaaaaa'
)

plt.plot(
    [min(xdata), uk_pos.n],
    [fitfunc(uk_pos.n, *pfinal), uk_wl.n],
    '--',
    color='#aaaaaa'
)

mpl_annotate(plt, '$x_{uk} = $' + '${:.1fL}$ px'.format(uk_pos), (605, 645))
mpl_annotate(plt, '$\lambda_{uk} = $' + '${:.1fL}$ nm'.format(uk_wl), (550, 652))

print('uk: {:.3f}nm'.format(uk_wl))

print('dl: {:.3f} sig'.format(abs(uk_wl.n-656.281)/uk_wl.s))

plt.legend()

plt.savefig('wavelength_analysis/wl_ne_cal.png')
