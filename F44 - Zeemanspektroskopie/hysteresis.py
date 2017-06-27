#! /usr/bin/python3
import pandas as pd
import numpy as np

magnetic_field_data_up = pd.read_csv("hysteresis_up.csv")
magnetic_field_data_down = pd.read_csv("hysteresis_down.csv")



magnetic_field_data_up['dI/mT'] = .5
magnetic_field_data_up['Bu/mT'] = (magnetic_field_data_up['B1/mT']
                                + magnetic_field_data_up['B2/mT']
                                + magnetic_field_data_up['B3/mT'])/3
magnetic_field_data_up['dBu/mT'] = np.sqrt(3)*.01*magnetic_field_data_up['Bu/mT']
print(magnetic_field_data_up)

magnetic_field_data_down['dI/mT'] = .5
magnetic_field_data_down['Bd/mT'] = (magnetic_field_data_down['B1/mT']
                                + magnetic_field_data_down['B2/mT']
                                + magnetic_field_data_down['B3/mT'])/3
magnetic_field_data_down['dBd/mT'] = np.sqrt(3)*.01*magnetic_field_data_down['Bd/mT']
#print(magnetic_field_data_down)

magnetic_field_data_down.sort_values('I/A', inplace=True)
print(magnetic_field_data_down)

magnetic_field_data = pd.merge(
                            magnetic_field_data_up[['I/A','Bu/mT','dBu/mT']],
                            magnetic_field_data_down[['I/A','Bd/mT','dBd/mT']],
                            on='I/A'
                        )

magnetic_field_data['deltaB/sigma'] = (magnetic_field_data['Bu/mT'] - magnetic_field_data['Bd/mT'])*2/(magnetic_field_data['dBu/mT']+magnetic_field_data['dBd/mT'])

magnetic_field_data.to_csv('hyteresis_res.csv', index=False)
