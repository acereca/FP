import uncertainties as unc
import uncertainties.unumpy as unp
import VisTools.plotting as vt
import VisTools.printing as vp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')

dist_long = unc.ufloat(224, 10) # cm 

# 6.2.1.1 - Angular Measurements
print("# 6.2.1.1 - Angular Measurements")
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.27, 5.83))
data = pd.read_csv("./6211.csv", skipinitialspace=True)
naming_dict = {
    "f": "f / MHz",
    "df": "df",
    "Dx": r"$\Delta$x / cm",
    "dDx": "dDx",
    "alpha": r"$\alpha$ / $^\circ$",
    "dalpha": "dalpha"
}
data.rename(columns=naming_dict, inplace=True)

## f vs Dx
print("## f vs Dx")
ax[0].errorbar(
    data[naming_dict["f"]],
    data[naming_dict["Dx"]],
    xerr=data[naming_dict["df"]],
    yerr=data[naming_dict["dDx"]],
    fmt='.',
    label="measurements"
)


params = vt.fit_linear(
    data[naming_dict["f"]],
    data[naming_dict["Dx"]],
    [.2/10, 1.5],
    fitlabel="linear fit",
    fig=ax[0]
)

vp.unc_pp("m", params[0]/1e6*1e2, "s*m")
vp.unc_pp("c", params[1]/1e2, "m")

# f vs alpha
print("\n## f vs alpha")
data[naming_dict["alpha"]] = unp.arctan(unp.uarray(data[naming_dict["Dx"]], data[naming_dict["dDx"]]) / dist_long)

ax[1].errorbar(
    data[naming_dict["f"]],
    unp.nominal_values(data[naming_dict["alpha"]])/2/np.pi*360,
    xerr=data[naming_dict["df"]],
    yerr=unp.std_devs(data[naming_dict["alpha"]])/2/np.pi*360,
    fmt='.'
)

params = vt.fit_linear(
    data[naming_dict["f"]],
    unp.nominal_values(data[naming_dict["alpha"]])/2/np.pi*360,
    [.1, -.1],
    fitlabel="linear fit",
    fig=ax[1]
)

vp.unc_pp("m", params[0]*1e6, "deg/Hz")
vp.unc_pp("c", params[1], "deg")

## META
ax[0].set_title("Angular Relations - AOM")
ax[0].set_ylabel(naming_dict["Dx"])
ax[1].set_ylabel(naming_dict["alpha"])
ax[1].set_xlabel(naming_dict["f"])
plt.savefig("61.png")
plt.clf()

# 6.2.1.2 - Intensity Ratio vs Frequency
print("\n# 6.2.1.2 - Intensity Ratio vs Frequency")
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.27, 5.83))
data = pd.read_csv("./6212.csv", skipinitialspace=True)
naming_dict = {
    "f": "f / MHz",
    "df": "df",
    "V_0": "$V_{SO 0}$ / V",
    "dV_0": "dV_0",
    "V_1": "$V_{SO 1}$ / mV",
    "dV_1": "dV_1",
    "R": "ratio"
}
data.rename(columns=naming_dict, inplace=True)

data[naming_dict["R"]] = unp.uarray(data[naming_dict["V_1"]], data[naming_dict["dV_1"]])/unp.uarray(data[naming_dict["V_0"]], data[naming_dict["dV_0"]])/1000

## V_1
ax[0].errorbar(
    data[naming_dict["f"]],
    data[naming_dict["V_1"]],
    fmt=".",
    xerr=data[naming_dict["df"]],
    yerr=data[naming_dict["dV_1"]],
    label="measurements"
)

## R
ax[1].errorbar(
    data[naming_dict["f"]],
    unp.nominal_values(data[naming_dict["R"]]),
    fmt='.',
    xerr=data[naming_dict["df"]],
    yerr=unp.std_devs(data[naming_dict["R"]]),
    label="ratio"
)

## META
ax[0].set_title("Intensity Ratio vs. Frequency")
ax[0].set_ylabel(naming_dict["V_1"])
ax[1].set_ylabel(naming_dict["V_1"][:-5] + " / " + naming_dict["V_0"][:-3])
ax[1].set_xlabel(naming_dict["f"])
plt.savefig("62.png")
plt.clf()

# 6.2.1.3 - Intensity Ratio vs Amplitude inpot
print("\n# 6.2.1.3 - Intensity Ratio vs Input Amplitude")
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.27, 5.83))
data = pd.read_csv("./6213.csv", skipinitialspace=True)
data["V_li"] = unp.uarray(data["V_li"], .05)
data["V_lo"] = unp.uarray(data["V_lo"], data["dV_lo"])
data["V_0"] = unp.uarray(data["V_0"], data["dV_0"])
data["V_1"] = unp.uarray(data["V_1"], data["dV_1"])
data["R"] = data["V_1"] / data["V_0"]
naming_dict = {
    "V_li": "$V_{level,in}$ / V",
    "V_lo": "$V_{level,out}$ / V",
    "V_0": "V_0",
    "V_1": "$V_{SO 1}$ / V",
    "R": "$V_{SO 1}$ / $V_{SO 0}$"
}

## V_1
ax[0].errorbar(
    unp.nominal_values(data["V_li"]),
    unp.nominal_values(data["V_1"]),
    xerr=unp.std_devs(data["V_li"]),
    yerr=unp.std_devs(data["V_1"]),
    fmt='.',
    label='measurements'
)

## R
ax[1].errorbar(
    unp.nominal_values(data["V_li"]),
    unp.nominal_values(data["R"]),
    xerr=unp.std_devs(data["V_li"]),
    yerr=unp.std_devs(data["R"]),
    fmt='.',
    label="ratio"
)

#META
ax[0].set_title("Intensity Ratio vs Input Amplitude")
ax[0].set_ylabel(naming_dict["V_1"])
ax[1].set_ylabel(naming_dict["R"])
ax[1].set_xlabel(naming_dict["V_li"])
plt.savefig("63.png")
