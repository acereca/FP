#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
XOR Network
'''
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa
plt.close('all')

# load PyNN interface for the Spikey neuromorphic hardware
import pyNN.hardware.spikey as pynn  # noqa

# adapt these parameters to tune the network
# hint: start by turning off all connections and match the behavior iteratively

use_other_spikey_half = True
jitter = 0.1     # width of the normal distribution in ms
deltat = 100.   # time till the next example in ms
I2Yw = 10        # input to y_i weight
Y2Hw = 4       # weight from y_i to h_i (excitatory)
Y2Hi = 8        # weight from y_i to h_j (inhibitory)
H2Ow = 7        # weight from h_i to o
popsize  = 2    # size of the population
skipsize = 2    # space between two used neurons

labels = ['i1', 'i2', 'y1', 'y2', 'h1', 'h2', 'o']
vlabel = 'o'   # record membrane of this neuron
                # -> use to determine appropriate weights
skip_if_unreliable_list = ['h2']    # if you enter a label here a dummy population
                                # will be placed before the labeled population
                                # is placed. Use this if you cannot reproduce
                                # the correct behavior for a population.

####################################################################
# experiment parameters
# in biological time and parameter domain
####################################################################

runtime = 4000.0  # ms -> 0.4ms on hardware

InputNeuronParams = {
    'v_reset'   : -70.0,  # mV shared
    'e_rev_I'   : -80.0,  # mV shared
    'v_rest'    : -54.0,  # mV shared
    'v_thresh'  : -50.0,  # mV shared
    'g_leak'    :  20.0,  # nS  -> tau_mem = 0.2nF / 20nS = 10ms individual
}

HiddenNeuronParams = InputNeuronParams
OutNeuronParams = InputNeuronParams

# Input spikes 1s-2s only i1, 2s-3s only i2, 3s-4s both
# desired output 0s-1s 0, 1s-3s 1, 3s-4s 0
i1spkt = np.concatenate((np.arange(1000., 2000., deltat),
                         np.arange(3000., 4000., deltat)))
i2spkt = np.concatenate((np.arange(2000., 3000., deltat),
                         np.arange(3000., 4000., deltat)))
nsamples = int(1000./deltat)
oexpected = nsamples * [0] + 2 * nsamples * [1] + nsamples * [0]

if jitter != 0.:
    i1spkt += np.random.normal(0., jitter, size=i1spkt.shape)
    i2spkt += np.random.normal(0., jitter, size=i2spkt.shape)

####################################################################
# procedural experiment description
####################################################################

# necessary setup
if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(mappingOffset=neuron_offset)

we = pynn.minExcWeight()
wi = pynn.minInhWeight()

# set up network

# I want another neuron, so I need to build some dummy neurons, because
pynn.Population(1, pynn.IF_facets_hardware1, InputNeuronParams)

# create & record neurons
populations = {}
parrotsE = {}
parrotsI = {}

# stimuli
populations['i1'] = pynn.Population(popsize, pynn.SpikeSourceArray,
                                    {'spike_times': i1spkt})
populations['i2'] = pynn.Population(popsize, pynn.SpikeSourceArray,
                                    {'spike_times': i2spkt})

# neurons
for label in labels[2:]:
    # workaround for unreliable neurons
    if label in skip_if_unreliable_list:
        pynn.Population((popsize+skipsize)*3, pynn.IF_facets_hardware1)
    populations[label] = pynn.Population(popsize, pynn.IF_facets_hardware1,
                                                        InputNeuronParams)
    # skip some neurons to reduce crosstalk
    pynn.Population(skipsize, pynn.IF_facets_hardware1, InputNeuronParams)
    # now the excitatory parrot neurons
    parrotsE[label] = pynn.Population(popsize, pynn.IF_facets_hardware1,
                                                        InputNeuronParams)
    # skip some neurons to reduce crosstalk
    pynn.Population(skipsize, pynn.IF_facets_hardware1, InputNeuronParams)
    # now the inhibitory parrot neurons
    parrotsI[label] = pynn.Population(popsize, pynn.IF_facets_hardware1,
                                                        InputNeuronParams)
    # skip some neurons to reduce crosstalk
    pynn.Population(skipsize, pynn.IF_facets_hardware1, InputNeuronParams)

# parrot neuron connections
for label in labels[2:]:
    pynn.Projection(populations[label], parrotsE[label],
                pynn.AllToAllConnector(weights=15*we), target="excitatory")
    pynn.Projection(populations[label], parrotsI[label],
                pynn.AllToAllConnector(weights=15*we), target="excitatory")
    pynn.Projection(parrotsI[label], populations[label],
                pynn.AllToAllConnector(weights=15*wi), target="inhibitory")
    pynn.Projection(parrotsI[label], parrotsE[label],
                pynn.AllToAllConnector(weights=15*wi), target="inhibitory")
    # inhibitory parrots kill themselves
    pynn.Projection(parrotsI[label], parrotsI[label],
                pynn.AllToAllConnector(weights=15*wi), target="inhibitory")

# 1st layer is stimulated by background
pynn.Projection(populations['i1'], populations['y1'],
                pynn.AllToAllConnector(weights=I2Yw*we), target="excitatory")
pynn.Projection(populations['i2'], populations['y2'],
                pynn.AllToAllConnector(weights=I2Yw*we), target="excitatory")

# 2nd layer
pynn.Projection(parrotsE['y1'], populations['h1'], pynn.AllToAllConnector(
                weights=Y2Hw*we), synapse_dynamics=None, target="excitatory")
pynn.Projection(parrotsE['y2'], populations['h2'], pynn.AllToAllConnector(
                weights=Y2Hw*we), synapse_dynamics=None, target="excitatory")
pynn.Projection(parrotsE['h1'], populations['o'], pynn.AllToAllConnector(
                weights=H2Ow*we), synapse_dynamics=None, target="excitatory")
pynn.Projection(parrotsE['h2'], populations['o'], pynn.AllToAllConnector(
                weights=H2Ow*we), synapse_dynamics=None, target="excitatory")
pynn.Projection(parrotsI['y1'], populations['h2'], pynn.AllToAllConnector(
                weights=Y2Hi*wi), target="inhibitory")
pynn.Projection(parrotsI['y2'], populations['h1'], pynn.AllToAllConnector(
                weights=Y2Hi*wi), target="inhibitory")

for label in labels:
    populations[label].record()
pynn.record_v(populations[vlabel][0], '')

# execute the experiment
pynn.run(runtime)

# evaluate results
spiketrains = {}
for label in labels:
    spiketrains[label] = populations[label].getSpikes()
    print(label, "spike rate: ", len(spiketrains[label])/runtime, "kHz")

# check if the computation was correct
correct = 0
for i in range(4):
    for j in range(nsamples):
        lower = (i*nsamples + j) * deltat
        upper = (i*nsamples + j + 1) * deltat
        if np.logical_and(spiketrains['o'] > lower,
                          spiketrains['o'] < upper).any():
            if oexpected[i*nsamples + j] == 1:
                correct += 1
        else:
            if oexpected[i*nsamples + j] == 0:
                correct += 1

print("Correct identified: {} of {}".format(correct, nsamples*4))

vm = pynn.membraneOutput
tm = pynn.timeMembraneOutput

pynn.end()

####################################################################
# data visualization
####################################################################
print 'average membrane potential:', np.mean(vm), 'mV'
print 'sampling step for membrane potential:', tm[1] - tm[0], 'ms'

colors = ['k', 'm', 'g', 'r', 'b', 'y', 'c']

# draw raster plot
fig = plt.figure()
ax = fig.add_subplot(211)  # row, col, nr
h = 0
for label in labels:
    ax.vlines(spiketrains[label], 2*h, 2*h+1, label=label, color=colors[h])
    h += 1
ticks = np.arange(0, 2*len(labels), 2) + .5
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_ylim(-1, 2*len(labels))
ax.set_xlim(tm[0], tm[-1])

# draw membrane potential
ax = fig.add_subplot(212)
ax.plot(tm, vm)
ax.set_xlabel('time (ms)')
ax.set_ylabel('vm(' + vlabel + ') [mV]')

plt.show()
plt.savefig('xor.png')
