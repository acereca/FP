#! /usr/bin/python3

import VisTools.plotting as vtp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
from scipy.interpolate import spline
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp

collected_data = pd.DataFrame()

values = {
    "co60":  {
        "name":             "$^{60}$Co",
        "marked_ch":        [76, 196, 708, 927, 1047],
        "marked_ch_err":    [5, 10, 30, 20, 10, 10],
        "energy_theo":      [1.17323, 1.33248],
        "fitting_interval": [[870, 990], [990, 1150]],
        "timeframe":        735
    },
    "cs137": {
        "name":             "$^{137}$Cs",
        "marked_ch":        [78, 171, 370, 543],
        "marked_ch_err":    [5, 10, 15, 10],
        "energy_theo":      [0.6616],
        "fitting_interval": [[480, 700]],
        "timeframe":        290
    },
    "mn54":  {
        "name":             "$^{54}$Mn",
        "marked_ch":        [14, 75, 178, 497, 678],
        "marked_ch_err":    [2, 5, 10, 15, 10],
        "energy_theo":      [0.8353],
        "fitting_interval": [[600, 800]],
        "timeframe":        298
    },
    "ba133": {
        "name":             "$^{133}$Ba",
        "marked_ch":        [78, 102, 144, 257, 308],
        "marked_ch_err":    [5, 10, 10, 10, 10],
        "energy_theo":      [0.356],
        "fitting_interval": [[275, 400]],
        "timeframe":        282
    },
    "na22":  {
        "name":             "$^{22}$Na",
        "marked_ch":        [76, 162, 420, 804, 1005],
        "marked_ch_err":    [5, 20, 10, 15, 15, 20],
        "energy_theo":      [1.2746],
        "fitting_interval": [[900, 1100]],
        "timeframe":        316
    }
}

distr = lambda x, m, gamma, intens: intens*(gamma**2/((x-m)**2+gamma**2))

###############################################################################
# plotting for intensity
for f in values.keys():

    depth = len(values[f]["fitting_interval"])

    plt.cla()
    data = pd.read_table("data/" + f + "_int", header=None, decimal=',').transpose()

    data.columns = [f + "_int"]

    # add read values into collected_data
    collected_data = pd.concat([collected_data, data], axis=1)

    values[f]["peak_params"] = []

    # calculating right xlimit
    for i, v in enumerate(reversed(data[f + "_int"][:2000])):
        if v > 100:
            values[f]["xlim_r"] = len(data[f + "_int"])-i-1
            #print(xlim2, v)
            break
    #plt.xlim([0, xlim2*1.1])

    # fitting the photopeaks
    for i, interv in enumerate(values[f]["fitting_interval"]):
        x_data = np.arange(interv[0],interv[1])

        p0 = [
            values[f]["marked_ch"][-depth+i],
            70,
            data[f + "_int"][values[f]["marked_ch"][-depth+i]]
        ]

        fparams = vtp.fit(
            x_data,
            data[f + "_int"][interv[0]:interv[1]],
            distr,
            p0,
            [10]*len(x_data)
        )

        # extrapolation by eye (bc of energy_resolution fit)
        if f == "co60":
            if i == 1:
                fparams[1]*=.87
            if i == 0:
                fparams[1]*=.79

        ## replace the old estimations with the fitted peak positions
        values[f]["marked_ch"][-depth+i]      = fparams[0].n
        values[f]["marked_ch_err"][-depth+i] += fparams[0].s
        values[f]["marked_ch_err"][-depth+i]  = values[f]["marked_ch_err"][-depth+i]/2


        values[f]["peak_params"].append([*fparams])
        #print(*unp.nominal_values([*fparams]))

# add underground to dataset
data = pd.read_table("data/" + f + "_int", header=None, decimal=',').transpose()
data.columns = ["underground_int"]

# add read values into collected_data
collected_data = pd.concat([collected_data, data], axis=1)

# fitting the calibration
list_en = []
list_ch = []
for e in values.keys():
    for i, v in enumerate(values[e]["energy_theo"]):
        depth = len(values[e]["energy_theo"])

        list_en.append(v)
        list_ch.append(values[e]["marked_ch"][-depth+i])


fit_en_m, fit_en_c = vtp.fit_linear(
    list_ch,
    list_en,
    [
        1/1000,
        0
    ],
    None
)


###############################################################################
# calibration plot
## figure setup
fig = plt.figure(figsize=(11.7,8.3))
plt.style.use('bmh')
plt.minorticks_on()
plt.grid(b=True, which="minor", color="#cccccc")

## plotting the fitted linear curve
calib_xdata = np.array([200, 1100])
plt.plot(
    calib_xdata,
    calib_xdata*fit_en_m.n+fit_en_c.n,
    label="linear fit",
    color="#c0c0c0"
)

## plotting the used points
for e in values.keys():
    for i, v in enumerate(values[e]["energy_theo"]):
        depth = len(values[e]["energy_theo"])
        plt.errorbar(
            values[e]["marked_ch"][-depth +i],
            v,
            xerr=values[e]["marked_ch_err"][-depth+i],
            label=(values[e]["name"] + " " + str(i+1)) if e == "co60" else values[e]["name"],
            zorder=2,
            elinewidth=1,
            fmt=".",
            ms=4
        )

## add the curve parameters to plot
plt.annotate(
    "Energy = Channel $\cdot$ m + c\n\nm = $({:.3fL})$ keV\nc  = $({:.3fL})$ keV"
        .format(fit_en_m*1000, fit_en_c*1000),
    xy=(800, 0.6),
    xycoords='data',
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=14,
    bbox=dict(
        boxstyle="round",
        fc="1"
    )
)

## plot meta
plt.legend()
plt.xlabel("channel")
plt.ylabel("Energy / MeV")
plt.title("Calibration Fit, Channel vs. Energy")
plt.savefig("energy_calibration.png")


###############################################################################
# underground plot

plt.clf()
plt.plot(
    np.array(collected_data.index.values.tolist()) * fit_en_m.n + fit_en_c.n,
    collected_data["underground_int"]
)

plt.xlim([0,1.5])
plt.ylim([-100,2500])
plt.xlabel("Energy / MeV")
plt.ylabel("Intensity")
plt.savefig("int_underground.png")


###############################################################################
# plot Energyresolution
plt.cla()
lims    = [.2, 1.5]
fitlims = [.3, 1.5]

plt.xlim(lims)

fitpointsx = np.array([])
fitpointsy = np.array([])

for e in values.keys():
    for i, v in enumerate(values[e]['energy_theo']):
        depth = len(values[e]["energy_theo"])
        plt.errorbar(
            (values[e]["marked_ch"][-depth +i]*fit_en_m+fit_en_c).n,
            (values[e]["peak_params"][i][1]*2*fit_en_m).n,
            xerr=(values[e]["marked_ch"][-depth +i]*fit_en_m+fit_en_c).s,
            yerr=(values[e]["peak_params"][i][1]*2*fit_en_m).s,
            label= (values[e]["name"] + " " + str(i+1)) if e == "co60" else values[e]["name"],
            zorder=2,
            elinewidth=1,
            fmt=".",
            ms=4
        )

        fitpointsx = np.append(fitpointsx,np.array([values[e]["marked_ch"][-depth +i]*fit_en_m.n+fit_en_c.n]))
        fitpointsy = np.append(fitpointsy,np.array([values[e]["peak_params"][i][1].n*2*fit_en_m.n]))
        #print(e, i, -depth+i, values[e]["peak_params"][-depth +i][1].n)

ef = lambda x, intensity, w, c: c + np.exp(x/w) * intensity

fparams = vtp.fit(
    fitpointsx,
    fitpointsy,
    ef,
    [
        1,
        1,
        0
    ],
    None
)

fx_data = np.arange(*fitlims, .1)

plt.plot(
    fx_data,
    ef(fx_data, *unp.nominal_values(fparams)),
    label="exp. Fit: $I\cdot e^{E/w} + c$",
    color="gray"
)

## add the curve parameters to plot

plt.annotate(
    "$c = ({c:.3fL})$ keV\n$I = ({I:.3fL})$ keV\n$w = ({w:.3fL})$ keV".format(
        c=fparams[2]*1000,
        I=fparams[0]*1000,
        w=fparams[1]*1000
    ),
    xy=(.9, 0.05),
    xycoords='data',
    xytext=(0, 0),
    textcoords='offset points',
    fontsize=14,
    bbox=dict(
        boxstyle="round",
        fc="1"
    )
)


plt.xlabel('Energy / MeV')
plt.ylabel('$\Delta$ Energy / MeV')
plt.legend()
plt.savefig("energy_resolution.png")



plt.cla()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

###############################################################################
# plot intensity with Energy
for e in values.keys():
    ax1.cla()

    # plot intensity measurements
    ax1.plot(
        collected_data[e + "_int"] -
            collected_data["underground_int"] /
            71304 * values[e]["timeframe"],
        collected_data.index.values,
        color="gray"
    )

    ymax = collected_data[e + "_int"][20:].max()*1.1

    for i, pos in enumerate(values[e]["marked_ch"]):
        if e == "ba133":
            offset = 5000
        else:
            offset = 0
        ann_ypos = np.linspace( ymax*.01+offset, ymax/4+offset, 5)
        ax1.vlines(
            pos,
            0,
            ymax,
            linestyles='dotted',
            colors='gray'
        )

        ax1.annotate(
                "ch = {:.0f}\nE  = ${:.3fL}$ MeV".format(
                    pos,
                    pos * fit_en_m+fit_en_c
                ),
                xy=(pos, ann_ypos[i]),
                xycoords='data',
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(
                    boxstyle="round",
                    fc="1"
                )
            )

    for i, interv in enumerate(values[e]["fitting_interval"]):
        fit_xdata = np.arange(interv[0], interv[1])


        ax1.plot(
            fit_xdata,
            distr(
                fit_xdata,
                values[e]["peak_params"][i][0].n,
                values[e]["peak_params"][i][1].n,
                values[e]["peak_params"][i][2].n
            )# + offset
        )

    ylimits = np.array([0, ymax])

    # dummy plot
    #ax2.plot(np.arange(0, 1, .1), np.ones(10))
    ax2.set_xlim([fit_en_c.n, values[e]["xlim_r"] * fit_en_m.n + fit_en_c.n])

    ax2color = "#00cc00"

    ax2.grid(b=True, which="major", color=ax2color)
    ax2.spines['top'].set_color(ax2color)
    ax2.tick_params(axis='x', colors=ax2color)

    ax2.set_xlabel("Energy / MeV")

    # plot meta
    ax1.set_xlim([20, values[e]["xlim_r"]])
    ax1.set_ylim(ylimits)

    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Counts")
    ax1.legend()

    elm = e[:2].capitalize()
    nc = e[2:]
    plt.title("Intensityspectrum of $^{{{}}}${}".format(nc, elm), y=1.08)

    plt.savefig("int_" + e + ".png")
    plt.cla()

###############################################################################
# plot coincidence spectra

coinc_list = {
    "co60": {
        "filename":         "co60_koinz",
        "fitting_interval": [[1.2, 1.325], [1.36, 1.49]],
        "name":             "$^{60}$Co"
    },
    "cs137_uncal":{
        "filename":         "cs137_koinz_02",
        "fitting_interval": [[0.68, 0.8]],
        "name":             "$^{137}$Cs uncalibrated"
    },
    "cs137_cal": {
        "filename":         "cs137_koinz_03",
        "fitting_interval": [[0.68, 0.8]],
        "name":             "$^{137}$Cs calibrated"
    }
}

for e in coinc_list.keys():
    data = pd.read_table(
        "data/" + coinc_list[e]["filename"],
        header=None, decimal=','
    ).transpose()
    data.columns = [e]
    xdata = np.array(data.index.values.tolist())*fit_en_m.n

    plt.clf()

    plt.plot(
        xdata,
        data[e],
        label = "measurement of {}".format(coinc_list[e]["name"])
    )

    # fit lorentzian(x, m, hwhm, int)
    for i, interv in enumerate(coinc_list[e]["fitting_interval"]):
        fitfilter = np.logical_and(xdata > interv[0], xdata < interv[1])

        p0 = [
            np.mean(interv),
            .05,
            20
        ]


        #print(np.logical_and(xdata > interv[0], xdata < interv[1]))
        fparams = vtp.fit(
            xdata[fitfilter],
            data[e][fitfilter],
            distr,
            p0,
            None
        )
        print(fparams)

        plt.plot(
            xdata[fitfilter],
            distr(xdata[fitfilter], *(unp.nominal_values(fparams))),
            label= "lorentzian fit" +
                (" no. {}".format(i+1) if e == "co60" else "")
        )

        # plt.errorbar(
        #     xdata[np.logical_not(fitfilter)],
        #     distr(
        #         xdata[np.logical_not(fitfilter)],
        #         *(unp.nominal_values(fparams))
        #     ),
        #     fmt = ".",
        #     markersize=1
        # )


    #plt.hlines()


    ilist = data[data[e] > 5].index.tolist()
    print([min(ilist)*.9, max(ilist)*1.1])
    plt.xlim(np.array([min(ilist)*.9, max(ilist)*1.1])*fit_en_m.n)

    plt.xlabel("Energy / MeV")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig("coinc_" + e + ".png")
