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

naming_dict = {'V_i': '$V_{in}$ / V', 'V_o': '$V_{out}$ / kV'}
data.rename(columns=naming_dict, inplace=True)
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
plt.savefig("5.png")

vp.unc_pp('rm', rm*1000, aftercomma=3)
vp.unc_pp('rc', rc*1000, 'V')
