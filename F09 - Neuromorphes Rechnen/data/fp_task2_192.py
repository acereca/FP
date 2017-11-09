#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 2 - Calibrating Neuron Parameters

Andreas Gruebl, July 2016, agruebl@kip.uni-heidelberg.de
'''
# load PyNN interface for the Spikey neuromorphic hardware
import numpy as np

import pyNN.hardware.spikey as pynn
# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa

####################################################################
# experiment parameters
# in biological time and parameter domain
####################################################################

use_other_spikey_half = False
runtime = 200.0  # ms -> 0.1ms on hardware

neuronParams = {
    'v_reset'   : -80.0,  # mV
    'e_rev_I'   : -75.0,  # mV
    'v_rest'    : -50.0,  # mV
    'v_thresh'  : -61.0,  # mV  - default value.
                          # Change to result of your calculation!
    'g_leak'    :  20.0,  # nS  -> tau_mem = 0.2nF / 20nS = 10ms
    'tau_refrac':   1.0   # ms 
}

####################################################################
# procedural experiment description
####################################################################

# necessary setup
if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(calibTauMem=False, mappingOffset=neuron_offset)

# set up one neuron
# create neurons
neurons = [pynn.Poputation(1, pynn.IF_facets_hardware1, neuronParams) for i in range(0,192)]
#neuron1 = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)
#neuron2 = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)
#neuron3 = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)
#neuron4 = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)

# increase refractory period by reducing hardware parameter icb
pynn.hardware.hwa.setIcb(0.2)

# individually set leakage conductance:
# -> these are default values! They should be tuned for identical firing rate.
#neuron1.set({'g_leak' : 20.0*816/980})
#neuron2.set({'g_leak' : 20.0*816/660})
#neuron3.set({'g_leak' : 20.0})
#neuron4.set({'g_leak' : 20.0*816/700})

# define which observables to record
# spike times
#neuron1.record()
#neuron2.record()
#neuron3.record()
#euron4.record()

for neur in neurons:
	neur.record()

# membrane potential
# when recording more than one membrane voltage, the on-board ADC cannot be
# used, anymore! (it will record a flat line). Instead, only oscilloscope
# recordings are possible.
#pynn.record_v([neuron1[0], neuron2[0], neuron3[0], neuron4[0]], '')


pynn.record_v([elem[0] for elem in neurons], '')

# execute the experiment
pynn.run(runtime)

# evaluate results
#spikes1 = neuron1.getSpikes()[:, 1]
#spikes2 = neuron2.getSpikes()[:, 1]
#spikes3 = neuron3.getSpikes()[:, 1]
#spikes4 = neuron4.getSpikes()[:, 1]
spikes = [neuron.getSpikes()[:, 1] for neuron in neurons]

membrane = pynn.membraneOutput
membraneTime = pynn.timeMembraneOutput

pynn.end()

####################################################################
# data visualization
####################################################################

print 'average membrane potential:', np.mean(membrane), 'mV'
print 'sampling step for membrane potential: {} ms'.format(
                                        membraneTime[1] - membraneTime[0])

# draw raster plot
ax = plt.subplot(211)  # row, col, nr
for spike in spikes1:
    ax.axvline(x=spike)
ax.set_xlim(0, runtime)
ax.set_ylabel('spikes')
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

# draw membrane potential
axMem = plt.subplot(212)
axMem.plot(membraneTime, membrane)
axMem.set_xlim(0, runtime)
axMem.set_xlabel('time (ms)')
axMem.set_ylabel('membrane potential (mV)')

plt.savefig('example.png')


############################
# custom addendum          #
############################

v_th = neuronParams['v_rest']-1/np.exp(1)*(neuronParams['v_rest']-neuronParams['v_reset'])
print("\nV_th = {} mV".format(v_th))


