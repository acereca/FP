#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 3 - A Single Neuron with Synaptic Input

Andreas Grübl, July 2016, agruebl@kip.uni-heidelberg.de
'''


import pyNN.hardware.spikey as pynn
import numpy as np
import matplotlib.pyplot as plt

weight             = 15.0           # synaptic weight in digital values
runtime            = 1000 * 1000.0  # runtime in biological time domain in ms
durationInterval   = 200.0          # interval between input spikes in ms
#neuronIndex        = 42             # choose neuron on chip in range(384)
synapseDriverIndex = 42             # choose synapse driver in range(256)
use_other_spikey_half = True

neuronParams = {
    'v_thresh'  : -30.0,
}

# turn off calibration of synapse line drivers
if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0

plotdata = {
	'single': [],
	'average': []
}

for neuronIndex in range(0, 191):
    print("\n{}\n".format(neuronIndex))

    pynn.setup(mappingOffset=neuronIndex+neuron_offset, calibSynDrivers=False)
    
    # convert weight to hardware units
    if weight > 0:
        weight *= pynn.minExcWeight()
        synapsetype = 'excitatory'
    else:
        weight *= -pynn.minInhWeight()
        synapsetype = 'inhibitory'
    
    # build network
    neurons = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)
    pynn.record_v(neurons[0], '')
    
    # allocate dummy synapse drivers sending no spikes
    if synapseDriverIndex > 0:
        stimuliDummy = pynn.Population(synapseDriverIndex, pynn.SpikeSourceArray,
                                        {'spike_times': []})
        prj = pynn.Projection(stimuliDummy, neurons,
                            pynn.AllToAllConnector(weights=0), target='inhibitory')
    
    # allocate synapse driver and configure spike times
    stimProp = {'spike_times':
                    np.arange(durationInterval, runtime - durationInterval,
                                durationInterval)}
    stimuli = pynn.Population(1, pynn.SpikeSourceArray, stimProp)
    prj = pynn.Projection(stimuli, neurons, pynn.AllToAllConnector(weights=weight),
                            target=synapsetype)
    
    # modify properties of synapse driver
    # drvifall controls the slope of the falling edge of the PSP shape.
    # smaller values increase the length, thus the total charge transmitted by
    # the synapse, thus the PSP height.
    print 'Range of calibration factors of drvifall for excitatory connections', \
                prj.getDrvifallFactorsRange('inh')
    prj.setDrvifallFactors([.02])
    prj.setDrvioutFactors([.02])
    
    # run network
    pynn.run(runtime)
    mem = pynn.membraneOutput
    time = pynn.timeMembraneOutput
    pynn.end()

    ######
    # calculate spike-triggered average of membrane potential
    timeNorm = time - time[0]
    # number of data points per interval
    lenInterval = np.argmin(abs(time - durationInterval))
    # number of intervals
    numInterval = int(len(mem) / lenInterval)
    # trim membrane data
    memCut = mem[:numInterval * lenInterval]
    # split membrane data into intervals
    memInterval = memCut.reshape(numInterval, lenInterval)
    # average membrane data
    # note that first and last interval are omitted, because without stimulus
    memAverage = np.mean(memInterval[1:-1], axis=0)

    plotdata['single'].append(np.max(memInterval[1]))
    plotdata['average'].append(np.max(memAverage))

# plot results
plt.figure(figsize=(19.2, 10.8))
plt.style.use('bmh')
#plt.plot(timeNorm[:lenInterval], memInterval[1], 'b')
#plt.plot(timeNorm[:lenInterval], memAverage, 'r')

ax1 = plt.subplot(131)
ax1.plot(range(0, len(plotdata['single'])), plotdata['single'], 'b')
ax1.plot(range(0, len(plotdata['average'])), plotdata['average'], 'r')

ax1.legend(['single EPSP', 'average across {} EPSPs'.format(numInterval)])
ax1.set_xlabel('synapse drive')
ax1.set_ylabel('membrane voltage (mV)')

ax2 = plt.subplot(132)
ax2.hist(plotdata['single'], bins=10, color='b')
ax2.set_title('Distribution of maxima \nfor single EPSPs')
ax2.set_xlabel('membrane Voltage [mV]')
ax2.set_ylabel('number of synapse drives')

print('single:', np.mean(plotdata['single']), np.std(plotdata['single']))


ax3 = plt.subplot(133)
ax3.hist(plotdata['average'], bins=10, color='r')
ax3.set_title('Distribution of maxima \nfor average across {} EPSPs'.format(numInterval))
ax3.set_xlabel('membrane Voltage [mV]')
ax3.set_ylabel('number of synapse drives')

print('average:', np.mean(plotdata['average']), np.std(plotdata['average']))

plt.savefig('epsp.png')
