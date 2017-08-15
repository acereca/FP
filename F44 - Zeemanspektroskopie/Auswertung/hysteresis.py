#! /usr/bin/python3
import pandas as pd
import numpy as np
import scipy.optimize as opt
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt

#data import
magnetic_field_data_up = pd.read_csv("hysteresis_up.csv")
magnetic_field_data_down = pd.read_csv("hysteresis_down.csv")

#column creation
magnetic_field_data_up['dI/mT'] = .5
magnetic_field_data_up['Bu/mT'] = (magnetic_field_data_up['B1/mT']
                                + magnetic_field_data_up['B2/mT']
                                + magnetic_field_data_up['B3/mT'])/3
magnetic_field_data_up['dBu/mT'] = np.sqrt(3)*.01*magnetic_field_data_up['Bu/mT']
#print(magnetic_field_data_up)

magnetic_field_data_down['dI/mT'] = .5
magnetic_field_data_down['Bd/mT'] = (magnetic_field_data_down['B1/mT']
                                + magnetic_field_data_down['B2/mT']
                                + magnetic_field_data_down['B3/mT'])/3
magnetic_field_data_down['dBd/mT'] = np.sqrt(3)*.01*magnetic_field_data_down['Bd/mT']
#print(magnetic_field_data_down)

#sort by current
magnetic_field_data_down.sort_values('I/A', inplace=True)
#print(magnetic_field_data_down)

# create merged table with avg values
magnetic_field_data = pd.merge(
                            magnetic_field_data_up[['I/A','Bu/mT','dBu/mT']],
                            magnetic_field_data_down[['I/A','Bd/mT','dBd/mT']],
                            on='I/A'
                        )

#magnetic_field_data['deltaB/sigma'] = (magnetic_field_data['Bu/mT'] - magnetic_field_data['Bd/mT'])*2/(magnetic_field_data['dBu/mT']+magnetic_field_data['dBd/mT'])

# fitting to linear
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y-fitfunc(p, x)) / err
pinit = [100, 40]
fitted_params = []
plt.style.use('bmh')

for fitstr in ['Bu/mT', 'Bd/mT']:
    res = opt.leastsq(
        errfunc, pinit,
        args=(
            magnetic_field_data['I/A'],
            magnetic_field_data[fitstr],
            magnetic_field_data['d'+fitstr]
        ),
        full_output=1
    )

    pfinal = res[0]
    covar = res[1]

    pres = unp.uarray(pfinal, np.sqrt([covar[0,0], covar[1,1]]))

    theo = fitfunc(pres, magnetic_field_data['I/A'])
    magnetic_field_data['theo_'+fitstr] = unp.nominal_values(theo)
    magnetic_field_data['theo_d'+fitstr] = unp.std_devs(theo)
    fitted_params.append(pres)

    plt.plot(magnetic_field_data['I/A'], magnetic_field_data['theo_' + fitstr])
    plt.errorbar(
        magnetic_field_data['I/A'],
        magnetic_field_data[fitstr],
        yerr=magnetic_field_data['d'+fitstr],
        xerr=.1,
        fmt='.'
    )

for i in range(2):
    print('c_{} = {:.3f}'.format(i,fitted_params[i][0]))
    print('m_{} = {:.3f}'.format(i,fitted_params[i][1]))

    print('mean_{} = {:.3f}'.format(i, (fitted_params[0][i]+fitted_params[1][i])/2))

print('abw_c = {:.2} sigma'.format(abs(fitted_params[0][0].n - fitted_params[1][0].n) / (fitted_params[0][0].s + fitted_params[1][0].s) * 2))
print('abw_m = {:.2} sigma'.format(abs(fitted_params[0][1].n - fitted_params[1][1].n) / (fitted_params[0][1].s + fitted_params[1][1].s) * 2))

#write results into tables
print(magnetic_field_data)
magnetic_field_data.to_csv('hysteresis_res.csv', index=False)

#plot values
plt.xlabel('Stromstärke I / A')
plt.ylabel('Magnetfeld B / mT')
plt.savefig('hysteresis.png')
