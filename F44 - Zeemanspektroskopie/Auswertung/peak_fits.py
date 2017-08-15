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

deltatable = pd.DataFrame(columns=('I','delta_1/m','delta_2/m'))

for c,d in enumerate(['10A', '12A', '13A']):
    #i = 1
    #f = '_pi'
    i = 0
    f = ''

    data = pd.read_table("data/trans/"+d +"/" + d + f, header=None, names=['px','int','d'])

    peaks = pu.indexes(data['int'], min_dist=15, thres=0.02)
    offset = min(data['int'])


    y0,y1 = min(data['int']-offset), max(data['int']-offset)

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

    #print(len(peaks_x[~peaks_x.mask][1::3]))

    peak_pos_exp = []

    #create subplots-array axarr
    fig,axarr = plt.subplots(
        1,
        len(peaks_x[~peaks_x.mask][1::3]),
        sharey=True,
        figsize=(19.2,10.8)
    )
    for it, peak in enumerate(peaks_x[~peaks_x.mask][1::3].tolist()):

        fitfunc = lambda x, a, mu, sig: a*np.exp(-.5*(x-mu)**2/sig**2) # general gaussian bell-curve, non-normalized
        intervalsize = 20 # no of points surrounding peak-estimate

        tr = [int(peak-intervalsize/2), int(peak+intervalsize/2)]
        sig0 = np.sqrt(
            sum(data['int'][tr[0]:tr[1]]*(data['px'][tr[0]:tr[1]]-peak)**2) /
            sum(data['int'][tr[0]:tr[1]])
        )


        axarr[it].errorbar(data['px'][tr[0]:tr[1]], data['int'][tr[0]:tr[1]]-y0-min(data['int'][tr[0]:tr[1]]-y0), fmt='.')

        pfinal, pcov = opt.curve_fit(
            fitfunc,
            data['px'][tr[0]:tr[1]],
            data['int'][tr[0]:tr[1]]-y0-min(data['int'][tr[0]:tr[1]]-y0),
            p0=[max(data['int'][tr[0]:tr[1]]), peak, sig0],
            sigma=[.05 for i in data['int'][tr[0]:tr[1]]]
        )


        xdata = np.arange(min(data['px'][tr[0]:tr[1]]), max(data['px'][tr[0]:tr[1]]), .1)
        axarr[it].plot(xdata, fitfunc(xdata, *pfinal))
        axarr[it].errorbar(pfinal[1], fitfunc(pfinal[1], *pfinal), xerr=abs(np.sqrt(pfinal[2])))


        peak_pos_exp.append(unc.ufloat(
            pfinal[1],
            np.sqrt(pcov[1,1]**2+pfinal[2]**2)
        ))

    fig.savefig('gauss/peak_' +d+f+'.png')

    # from here: plot scatterorder by position
    plt.clf()
    plt.figure(figsize=(19.2,10.8))
    xdata = peak_pos_exp
    ydata = [i for i,item in enumerate(peak_pos_exp)]
    plt.errorbar(
        unp.nominal_values(xdata),
        ydata,
        xerr = unp.std_devs(xdata),
        fmt='.',
        label='gemessene Ordnung der $\pi$-Linie',
        markersize=10,
        color='r'
    )
    plt.errorbar(
        peaks_x[~peaks_x.mask][0::3],
        ydata,
        xerr=unp.std_devs(xdata),
        color='g',
        fmt='.',
        markersize=10,
        label='gemessene Ordnung der $\sigma_1$-Linie'
    )
    plt.errorbar(
        peaks_x[~peaks_x.mask][2::3],
        ydata,
        xerr=unp.std_devs(xdata),
        color='#eedd00',
        fmt='.',
        markersize=10,
        label='gemessene Ordnung der $\sigma_2$-Linie'
    )

    # fit and plot a polynomial fit
    fitpoly = lambda x, m1, m2, y: m1*x**2+m2*x+y

    pfinal, pcov = opt.curve_fit(
        fitpoly,
        unp.nominal_values(xdata),
        ydata,
        p0=[-1e-5,.02,0]
    )

    fillxdata = np.arange(-10,max(unp.nominal_values(xdata))+10, 10)
    plt.plot(
        fillxdata,
        fitpoly(fillxdata, *pfinal),
        label='Polynomfit 2.Grades, $M_1\cdot x^2+M_2\cdot x+M_3$'
    )

    mpl_annotate(
        plt,
        '$M_1 = {:.3eL}$\n$M_2 = {:.3eL}$\n$ M_3 = {:.3eL}$'.format(
            unc.ufloat(pfinal[0], np.sqrt(pcov[0,0])),
            unc.ufloat(pfinal[1], np.sqrt(pcov[0,0])),
            unc.ufloat(pfinal[2], np.sqrt(pcov[0,0]))
        ),
        data_pos=(max(unp.nominal_values(xdata))/2,1)
    )

    plt.xlabel('Position / px')
    plt.ylabel('Beugungsordnung')
    plt.title('Fit der Polynomfunktion an die beobachteten Werte, B(' + d + ')')
    plt.legend()

    plt.savefig('scatterorder/sco_' +d+'.png')

    # fit the difference in orders between sigma and pi lines
    plt.clf()

    plt.errorbar(
        ydata,
        fitpoly(peaks_x[~peaks_x.mask][0::3], *pfinal) - ydata,
        yerr=(fitpoly(peaks_x[~peaks_x.mask][0::3]+unp.std_devs(xdata), *pfinal) - fitpoly(peaks_x[~peaks_x.mask][0::3]-unp.std_devs(xdata), *pfinal))/2,
        color='g',
        fmt='.',
        markersize=10,
        label='Verschiebung der $\sigma_1$-Linie'
    )

    plt.errorbar(
        ydata,
        fitpoly(unp.nominal_values(xdata), *pfinal) - ydata,
        yerr=(fitpoly(unp.nominal_values(xdata)+unp.std_devs(xdata), *pfinal) - fitpoly(unp.nominal_values(xdata)-unp.std_devs(xdata), *pfinal))/2,
        fmt='.',
        color='r',
        label='Verschiebung der $\pi$-Linie'
    )
    plt.errorbar(
        ydata,
        fitpoly(peaks_x[~peaks_x.mask][2::3], *pfinal) - ydata,
        yerr=(fitpoly(peaks_x[~peaks_x.mask][2::3]+unp.std_devs(xdata), *pfinal) - fitpoly(peaks_x[~peaks_x.mask][2::3]-unp.std_devs(xdata), *pfinal))/2,
        color='#eedd00',
        fmt='.',
        markersize=10,
        label='Verschiebung der $\sigma_2$-Linie'
    )

    plt.ylabel('Verschiebung')
    plt.xlabel('Ordnung')
    plt.title('Verschiebung dr Beugungsordnungen')
    plt.ylim((-.4, .4))
    plt.legend()

    plt.savefig('scatterorder/diff_sco' + d + '.png')


    #print(unp.std_devs(peak_pos_exp))

    deltatable.loc[c] = (
        d,
        unc.ufloat(
            np.mean(fitpoly(peaks_x[~peaks_x.mask][0::3], *pfinal) - ydata),
            np.std(fitpoly(peaks_x[~peaks_x.mask][0::3], *pfinal) - ydata)
        ),
        unc.ufloat(
            np.mean(fitpoly(peaks_x[~peaks_x.mask][2::3], *pfinal) - ydata),
            np.std(fitpoly(peaks_x[~peaks_x.mask][2 ::3], *pfinal) - ydata)
        )
    )


n = 1.4567
d = 4.04e-3
hc = 1.986e-25 # J*m
ld = 643.847e-9 #theoretischer wert
# ld = unc.float(600e-9, .)
diff_wl = ld**2/(2*d*np.sqrt(n**2-1))

deltatable['delta_1/m'] = deltatable['delta_1/m']*diff_wl
deltatable['delta_2/m'] = deltatable['delta_2/m']*diff_wl

deltatable['DE_1/J'] = hc/ld - hc/(ld + deltatable['delta_1/m'])
deltatable['DE_2/J'] = hc/ld - hc/(ld + deltatable['delta_2/m'])

# fitted params of Hysteresis
B_c = unc.ufloat(130.765, 15.849) #in T/A
B_m = unc.ufloat(39.168, 1.552)   #in T
deltatable['B/T'] = [v*B_m+B_c for v in [10,12,13]]

deltatable['mu_B_1*T/J'] = abs(deltatable['DE_1/J'] / deltatable['B/T'])
deltatable['mu_B_2*T/J'] = abs(deltatable['DE_2/J'] / deltatable['B/T'])

print(deltatable)
print()
print('mu_B1 = {}'.format(np.sum(deltatable['mu_B_1*T/J']+deltatable['mu_B_2*T/J'])/6))

plt.clf()
mua = []
for line in range(1,3):

    fitfunc = lambda x, m, c: x*m+c

    pfinal, pcov = opt.curve_fit(
        fitfunc,
        unp.nominal_values(deltatable['B/T']),
        abs(unp.nominal_values(deltatable['DE_'+ str(line) +'/J'])),
        p0=[100/.7, 1-100/.7]
    )

    plt.plot(
        [490, 670],
        fitfunc(np.array([490, 670]), *pfinal),
        label='gefittete Gerade für $\mu_{}$'.format(line)
    )

    plt.errorbar(
        unp.nominal_values(deltatable['B/T']),
        abs(unp.nominal_values(deltatable['DE_'+ str(line) +'/J'])),
        xerr=unp.std_devs(deltatable['B/T']),
        yerr=unp.std_devs(deltatable['DE_'+ str(line) +'/J']),
        label='Energieverschiebung $\sigma$-Linie {}'.format(line),
        fmt='.',
        lw=1
    )
    mu = unc.ufloat(pfinal[0], np.sqrt(pcov[0,0]))

    mpl_annotate(
        plt,
        '$\mu_{}\ \!_B$'.format(line) +'$ = {:.3eL}$'.format(mu) + r'$\frac{T}{J}$',
        (547,fitfunc(560, *pfinal))
    )

    mua.append(mu)

mu = (mua[0]+mua[1])/2
mpl_annotate(
    plt,
    '$\mu_B = {:.3eL}$'.format(mu) + r'$\frac{T}{J}$',
    (620, 5.25e-24)
)

plt.legend()

plt.xlim([490,670])
plt.xlabel('Magnetfeld / T')
plt.ylabel('Energiedifferenz / J')
plt.title('2te Bestimmung des Bohrschen Magneton')
plt.savefig('mu_B2.png')
