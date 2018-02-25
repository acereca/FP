import VisTools.printing as vp
import VisTools.plotting as vt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

plt.figure(figsize=(8.27,5.83))
plt.style.use('bmh')

data = pd.read_csv('./5.csv', skipinitialspace=True)
print(data.columns)

naming_dict = {
    'V_i': '$V_{in}$ / V',
    'V_o': '$V_{out}$ / kV',
    "V_p1": "$V_{diode, ax. 1}$ / V",
    "V_p2": "$V_{diode, ax. 2}$ / V",
    "V_po1": r"$V_{diode, ax +\frac{1}{2}}$ / V",
    "V_po2": r"$V_{diode, ax -\frac{1}{2}}$ / V"
}
data.rename(columns=naming_dict, inplace=True)

# 5.2.1 Amplifier Calibration
plt.errorbar(
    data[naming_dict['V_i']],
    data[naming_dict['V_o']],
    fmt='.',
    yerr=.02,
    xerr=.02,
    label="measured data"
)

rm, rc = vt.fit_linear(
    data[naming_dict["V_i"]],
    data[naming_dict["V_o"]],
    [.3, 0],
    [.1]* len(data[naming_dict["V_i"]]),
    "linear fit: $V_{in}\cdot m + c = V_{out}$"
)

vt.annotate_unc(plt, rm*1000, data_pos=(3, .25), name="m")
vt.annotate_unc(plt, rc*1000, data_pos=(3, .1), name="c")

plt.ylabel(naming_dict["V_o"])
plt.xlabel(naming_dict["V_i"])
plt.title("Amplifier Calibration")
plt.legend()
<<<<<<< HEAD
plt.savefig("521.png")
plt.clf()

vp.unc_pp('rm', rm*1000, aftercomma=3)
vp.unc_pp('rc', rc*1000, 'V')
print()

# 5.2.2 Mach-Zehner Interferometer
fig, ax = plt.subplots(2,1, sharex=True, figsize=(8.27, 5.83))

# Axis 1
ax[0].errorbar(
    data[naming_dict["V_i"]],
    data[naming_dict["V_p1"]],
    xerr=.05,
    fmt='.',
    label="measurements"
)

ffunc_cos = lambda x,A,pl,dp,off: off+A*np.cos(x/pl+dp)
ffunc_cos_plist = ["A", "L", "Dp", "c"]
ffunc_cos_plistu = ["V", "V", "", "V"]
params = vt.fit(
    data[naming_dict["V_i"]],
    data[naming_dict["V_p1"]],
    ffunc_cos,
    [.75, 8*np.pi, -.5, 1.25],
    fig=ax[0]
)

for k, p in enumerate(params):
    vp.unc_pp(ffunc_cos_plist[k] + "1", p, unit=ffunc_cos_plistu[k])

print()

## Axis 2
ax[1].errorbar(
    data[naming_dict["V_i"]],
    data[naming_dict["V_p2"]],
    xerr=.05,
    fmt='.',
    label="measurements"
)

params= vt.fit(
    data[naming_dict["V_i"]],
    data[naming_dict["V_p2"]],
    ffunc_cos,
    [.05, 1/np.pi, .5, .69],
    fig=ax[1]
)

for k, p in enumerate(params):
    vp.unc_pp(ffunc_cos_plist[k] + "2", p, unit=ffunc_cos_plistu[k])

print()

## META
ax[1].set_xlabel(naming_dict["V_i"])
ax[0].set_ylabel(naming_dict["V_p1"])
ax[0].legend()
ax[1].set_ylabel(naming_dict["V_p2"])
ax[1].legend(loc=(0,.49))
ax[0].set_title("Interferometer")
plt.savefig("522.png")
plt.clf()

# 5.3 intensity Modulation
fig, ax = plt.subplots(2,1, sharex=True, figsize=(8.27, 5.83))

## Axis 1
ax[0].errorbar(
    data[naming_dict["V_i"]],
    data[naming_dict["V_po1"]],
    fmt='.',
    xerr=.05,
    label="measurements"
)

params = vt.fit(
    data[naming_dict["V_i"]],
    data[naming_dict["V_po1"]],
    ffunc_cos,
    [1.75, 1/np.pi, -.5, 2],
    fig=ax[0]
)

## Axis 2
ax[1].errorbar(
    data[naming_dict["V_i"]],
    data[naming_dict["V_po2"]],
    fmt='.',
    xerr=.05,
    label="measurements"
)

params = vt.fit(
    data[naming_dict["V_i"]],
    data[naming_dict["V_po2"]],
    ffunc_cos,
    [-1.75, 1/np.pi, -.5, 2],
    fig=ax[1]
)

## META
ax[0].set_title("Polarization Manipulation")
ax[0].set_ylabel(naming_dict["V_po1"])
ax[0].legend()
ax[1].set_ylabel(naming_dict["V_po2"])
ax[1].legend()
ax[1].set_xlabel(naming_dict["V_i"])
plt.savefig("53.png")
=======
plt.savefig("5.png")

vp.unc_pp('rm', rm*1000, aftercomma=3)
vp.unc_pp('rc', rc*1000, 'V')
>>>>>>> de8350f5bf3b3ae186d8b150a7ce3090b1c2c340
