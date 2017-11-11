#! /usr/bin/python3

import VisTools.plotting as vtp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
from scipy.interpolate import spline
import numpy as np
import uncertainties.unumpy as unp

collected_data = pd.DataFrame()

values = {
    "co60":  {
        "marked_ch":        [76, 196, 708, 927, 1047],
        "marked_ch_err":    [5, 10, 30, 20, 10, 10],
        "energy_theo":      [1.17323, 1.33248],
        "fitting_interval": [[875, 990], [990, 1200]],
        "timeframe":        735
    },
    "cs137": {
        "marked_ch":        [78, 171, 370, 543],
        "marked_ch_err":    [5, 10, 15, 10],
        "energy_theo":      [0.6616],
        "fitting_interval": [[480, 700]],
        "timeframe":        290
    },
    "mn54":  {
        "marked_ch":        [14, 75, 178, 497, 678],
        "marked_ch_err":    [2, 5, 10, 15, 10],
        "energy_theo":      [0.8353],
        "fitting_interval": [[600, 800]],
        "timeframe":        298
    },
    "ba133": {
        "marked_ch":        [78, 102, 144, 257, 308],
        "marked_ch_err":    [5, 10, 10, 10, 10],
        "energy_theo":      [0.356],
        "fitting_interval": [[275, 400]],
        "timeframe":        282
    },
    "na22":  {
        "marked_ch":        [76, 162, 420, 804, 1005],
        "marked_ch_err":    [5, 20, 10, 15, 15, 20],
        "energy_theo":      [1.2746],
        "fitting_interval": [[900, 1100]],
        "timeframe":        316
    }
}

distr = lambda x, m, gamma, intens: intens*(gamma**2/((x-m)**2+gamma**2))

# plotting for intensity
for f in values.keys():

    depth = len(values[f]["fitting_interval"])

    plt.cla()
    data = pd.read_table("data/" + f + "_int", header=None, decimal=',').transpose()

    data.columns = [f + "_int"]

    # add read values into collected_data
    collected_data = pd.concat([collected_data, data], axis=1)


    values[f]["peak_params"] = []

    # plotting and annotating
    # plt.plot(
    #     data.index.values,
    #     data[f],
    #     label="Measurement",
    #     color="gray"
    # )

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
            50,
            data[f + "_int"][values[f]["marked_ch"][-depth+i]]
        ]

        fparams = vtp.fit(
            x_data,
            data[f + "_int"][interv[0]:interv[1]],
            distr,
            p0,
            [10]*len(x_data)
        )

        ## replace the old estimations with the fitted peak positions
        values[f]["marked_ch"][-depth+i]      = fparams[0].n
        values[f]["marked_ch_err"][-depth+i] += fparams[0].s
        values[f]["marked_ch_err"][-depth+i]  = values[f]["marked_ch_err"][-depth+i]/2

        values[f]["peak_params"].append([*fparams])
        print(*unp.nominal_values([*fparams]))

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
            label=e,
            zorder=2,
            elinewidth=1,
            fmt=".",
            ms=4
        )

## add the curve parameters to plot
plt.annotate(
    "Energy = Channel $\cdot$ m + c\n\nm = ${:.3fL}$ keV\nc  = ${:.3fL}$ keV".format(fit_en_m*1000, fit_en_c*1000),
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
plt.savefig("calibration.png")


ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()


# plot intensity
for e in values.keys():
    ax1.cla()
    ax1.plot(
        collected_data.index.values,
        collected_data[e + "_int"] - collected_data["underground_int"] / 71304 * values[e]["timeframe"],
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
                "ch = {:.0f}\nE  = ${:.3fL}$ MeV".format(pos, pos * fit_en_m+fit_en_c),
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
            distr(fit_xdata, values[e]["peak_params"][i][0].n, values[e]["peak_params"][i][1].n, values[e]["peak_params"][i][2].n)
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

    plt.savefig(e + "_int.png")


# plot intensity
# for ch in marked_channels[f]:
#     plt.vlines(ch, 0, data[f][ch], linestyles='dotted', colors='gray')
#     plt.annotate(
#         "ch = {}".format(ch),
#         xy=(ch, 0),
#         xycoords='data',
#         xytext=(0, 0),
#         textcoords='offset points',
#         fontsize=14,
#         bbox=dict(
#             boxstyle="round",
#             fc="1"
#         )
#     )

#########################################
# CODE GRAVEYARD                        #
#########################################

# x_data = np.arange(fparams[0].n-4*fparams[1].n, fparams[0].n+4*fparams[1].n)
# plt.plot(
#     x_data,
#     distr(x_data, fparams[0].n, fparams[1].n, fparams[2].n),
#     label="Lorentz-Fit No. {}".format(i+1)
# )
# plt.hlines(
#     fparams[2].n/2,
#     fparams[0].n-fparams[1].n,
#     fparams[0].n+fparams[1].n,
#     linestyles='dotted',
#     colors='gray'
# )
# plt.annotate(
#     "FWHM = ${:.2fL}$".format(2*fparams[1]),
#     xy=(fparams[0].n+fparams[1].n, fparams[2].n/2),
#     xycoords='data',
#     xytext=(0, 0),
#     textcoords='offset points',
#     fontsize=14,
#     bbox=dict(
#         boxstyle="round",
#         fc="1"
#     )
# )



# plot meta
# plt.xlabel("Channel")
# plt.ylabel("Counts")
# plt.legend()
#
# elm = f.split("_")[0][:2].capitalize()
# nc = f.split("_")[0][2:]
# plt.title("Intensityspectrum of $^{{{}}}${}".format(nc, elm))
# plt.savefig(f + ".png")




#
#
#
# plt.legend()
# plt.title("Calibration of channels vs Energy")
# plt.xlabel("Channel")
# plt.ylabel("Energy / MeV")
# plt.savefig("calibration.png")

# plotting calibration
# for e in theo_energy.keys():
#     for index, line in enumerate(theo_energy[e]):
#         plt.errorbar(
#             marked_channels[e+"_int"][-2+index],
#             theo_energy[e][index],
#             xerr=10,
#             fmt=".",
#             label="$^{{{}}}${}".format(e[2:],e[:2].capitalize())
#         )
# print(collected_data.columns.values)
