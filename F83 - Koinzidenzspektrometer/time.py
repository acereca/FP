#! /usr/bin/python3

import matplotlib.pyplot as plt
import VisTools.plotting as vtp
import VisTools.printing as vto
import pandas as pd
import numpy as np
import uncertainties.unumpy as unp
import scipy.optimize as opt

plt.figure(figsize=(11.7, 8.3)) # DIN A4
plt.style.use('bmh')
collected_data = pd.DataFrame()

distr = lambda x, m, gamma, intens: intens*(gamma**2/((x-m)**2+gamma**2))

# files to use for time calibration, and correlated timedelta
# theo_delay in ns
initial_data = {
    'delaynone':{
        'theo_delay': 0,
        'marked_ch':  1640,
        'fit_interv': [1500, 1800]
    },
    'delaystart':{
        'theo_delay': -40,
        'marked_ch':  1550,
        'fit_interv': [1500, 1800]
    },
    'delaystop':{
        'theo_delay': 40,
        'marked_ch':  1690,
        'fit_interv': [1500, 1800]
    }
}


for f in initial_data.keys():
    data = pd.read_table(
        "data/cs137_timespec_" + f,
        header=None,
        decimal=','
    ).transpose()

    data.columns = [f]

    collected_data = pd.concat([collected_data, data], axis=1)

    #fit to peak
    x_data = np.arange(*initial_data[f]['fit_interv'])

    p0 = [
        initial_data[f]['marked_ch'],
        50,
        data[f][initial_data[f]['marked_ch']]
    ]

    fparams = vtp.fit(
        x_data,
        data[f][initial_data[f]['fit_interv'][0]:initial_data[f]['fit_interv'][1]],
        distr,
        p0,
        [10]*len(x_data)
    )

    # print fitted params:
    vto.unc_pp('mean', fparams[0], formatting='f')
    vto.unc_pp('HWHM', fparams[1], formatting='f')
    #vto.unc_pp('int', fparams[2], formatting='f')

    initial_data[f]['fit_mean'] = fparams[0].n
    initial_data[f]['fit_dmean'] = fparams[0].s

    plt.errorbar(
        initial_data[f]['fit_mean'],
        initial_data[f]['theo_delay'],
        xerr=initial_data[f]['fit_dmean'],
        fmt='.'
    )

p0 = [
    .5,
    -720
]
print([initial_data[f]['fit_mean'] for f in initial_data.keys()])
fparams = opt.curve_fit(
    lambda x, m, c: x*m+c,
    [initial_data[f]['fit_mean'] for f in initial_data.keys()],
    [initial_data[f]['theo_delay'] for f in initial_data.keys()]
)
plt.plot(
    [initial_data[f]['fit_mean'] for f in initial_data.keys()],
    np.array([initial_data[f]['fit_mean'] for f in initial_data.keys()])*fparams[0][0]+fparams[0][1]
)

plt.title('Calibration Fit, Channel vs. Time')
plt.xlabel('channel')
plt.ylabel('Time / ns')
plt.savefig('time_calibration.png')


# apply calibration on co60

data = pd.read_table(
    "data/co60_timespec",
    header=None,
    decimal=','
).transpose()

data.columns = ['int']

x_data = np.arange(200, 450)

p0 = [
    325,
    50,
    50
]

fparams = opt.curve_fit(
    distr,
    x_data,
    data['int'][200:450],
    p0 = p0
)


plt.cla()
plt.plot(
    data.index.values[:-1],
    data['int'][:-1]
)

plt.plot(
    x_data,
    distr(np.array(x_data), *fparams[0])
)

plt.xlim([0, 500])

plt.savefig('time_co60.png')
