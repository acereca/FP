#! /usr/bin/python3

import VisTools.plotting as vtp
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
from scipy.interpolate import spline
import numpy as np

collected_data = pd.DataFrame()

values = {
    "co60":  {
        "marked_ch":        [76, 196, 708, 927, 1047],
        "marked_ch_err":    [5, 10, 30, 20, 10, 10],
        "energy_theo":      [1.17323, 1.33248],
        "fitting_interval": [[875,990],[990,1200]]
    },
    "cs137": {
        "marked_ch":        [78, 171, 370, 543],
        "marked_ch_err":    [5, 10, 15, 10],
        "energy_theo":      [0.6616],
        "fitting_interval": [[480,700]]
    },
    "mn54":  {
        "marked_ch":        [14, 75, 178, 497, 678],
        "marked_ch_err":    [2, 5, 10, 15, 10],
        "energy_theo":      [0.8353],
        "fitting_interval": [[600,800]]
    },
    "ba133": {
        "marked_ch":        [78, 102, 144, 257, 308],
        "marked_ch_err":    [5, 10, 10, 10, 10],
        "energy_theo":      [0.356],
        "fitting_interval": [[275,400]]
    },
    "na22":  {
        "marked_ch":        [76, 162, 420, 804, 1005],
        "marked_ch_err":    [5, 20, 10, 15, 15, 20],
        "energy_theo":      [1.2746],
        #"fitting_interval": [[375,500]]
        "fitting_interval": [[900,1100]]
    }
}


# plotting for intensity
for f in values.keys():

    depth = len(values[f]["fitting_interval"])

    plt.cla()
    data = pd.read_table("data/" + f + "_int", header=None, decimal=',').transpose()

    data.columns = [f + "_int"]

    # add read values into collected_data
    collected_data = pd.concat([collected_data, data], axis=1)

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

        distr = lambda x, m, gamma, intens: intens*(gamma**2/((x-m)**2+gamma**2))
        #print(f,len(fitting_intervals[f[:-4]]),i, -len(fitting_intervals[f[:-4]])+i, marked_channels[f][-len(fitting_intervals[f[:-4]])+i])
        #plt.plot(x_data, distr(x_data, marked_channels[f][-len(fitting_intervals[f[:-4]])+i], 50, 2000))

        p0 = [
            values[f]["marked_ch"][-depth+i],
            50,
            data[f + "_int"][values[f]["marked_ch"][-depth+i]]
        ]

        #print(p0)
        fparams = vtp.fit(
            x_data,
            data[f + "_int"][interv[0]:interv[1]],
            distr,
            p0,
            [10]*len(x_data)
        )
        #print([interv[0],interv[1]],fparams[0])

        values[f]["marked_ch"][-depth+i]      = fparams[0].n
        values[f]["marked_ch_err"][-depth+i] += fparams[0].s
        values[f]["marked_ch_err"][-depth+i]  = values[f]["marked_ch_err"][-depth+i]/2

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

#print(fit_en_m, fit_en_c)
#print(values)

# calibration plot
## figure setup
fig = plt.figure(figsize=(11.7,8.3))
plt.style.use('bmh')
plt.minorticks_on()
plt.grid(b=True, which="minor", color="#cccccc")

calib_xdata = np.array([200, 1100])
plt.plot(
    calib_xdata,
    calib_xdata*fit_en_m.n+fit_en_c.n,
    label="linear fit",
    color="#c0c0c0"
)

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
        #print(values[e]["marked_ch"][-depth +i],v)

plt.annotate(
    "Energy = Channel $\cdot$ m + c\nm = ${:.3fL}$ keV\nc = ${:.3fL}$".format(fit_en_m*1000, fit_en_c),
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
plt.legend()
plt.xlabel("channel")
plt.ylabel("Energy / MeV")
plt.title("Calibration Fit, Channel vs. Energy")
plt.savefig("calibration.png")


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
