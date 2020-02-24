# +-----------------------------------------------------------------------------+
# |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                         |
# |                                                                             |
# |  Licensed under the Apache License, Version 2.0 (the "License");            |
# |  you may not use this file except in compliance with the License.           |
# |  You may obtain a copy of the License at                                    |
# |                                                                             |
# |      http://www.apache.org/licenses/LICENSE-2.0                             |
# |                                                                             |
# |  Unless required by applicable law or agreed to in writing, software        |
# |  distributed under the License is distributed on an "AS IS" BASIS,          |
# |  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   |
# |  See the License for the specific language governing permissions and        |
# |  limitations under the License.                                             |
# +-----------------------------------------------------------------------------+
# |  Authors: Lorenz K. Mueller, Pascal Stark                                   |
# +-----------------------------------------------------------------------------+
"""
Models a Mach Zehnder Modulator.
"""

from colna.analyticnetwork import Network, Edge, SymNum
from colna.analyticnetwork import Testbench
from colna.analyticnetwork import PhysicalNetwork, Device, DeviceLink
from colna.lossconversion import dBcm_to_attenuation
import numpy as np
import matplotlib.pyplot as plt

# Global defaults
wavelength = 1.55e-6
neff = 2.6

# Splitter configuration
splitter_length = 10e-6
splitter_phase_default = 2 * np.pi / wavelength * neff * splitter_length
splitter_efficiency00 = SymNum(name='split_{00}', default=1 / np.sqrt(2), product=True)
splitter_efficiency01 = SymNum(name='split_{01}', default=1 / np.sqrt(2), product=True)
splitter_phase00 = SymNum(name='split_{\phi00}', default=splitter_phase_default, product=False)
splitter_phase01 = SymNum(name='split_{\phi01}', default=splitter_phase_default, product=False)

splitter = Device('splitter')
splitter.init_from_phase_and_attenuation_matrix(np.array([[splitter_efficiency00, splitter_efficiency01]]),
                                                np.array([[splitter_phase00, splitter_phase01]]), delay=0)
splitter.add_input('i0')

# Combiner configuration
combiner_length = 10e-6
combiner_phase_default = 2 * np.pi / wavelength * neff * combiner_length
combiner_efficiency00 = SymNum(name='comb_{00}', default=1 / np.sqrt(2), product=True)
combiner_efficiency01 = SymNum(name='comb_{10}', default=1 / np.sqrt(2), product=True)
combiner_phase00 = SymNum(name='comb_{\phi00}', default=combiner_phase_default, product=False)
combiner_phase10 = SymNum(name='comb_{\phi10}', default=combiner_phase_default, product=False)

combiner = Device('combiner')
combiner.init_from_phase_and_attenuation_matrix(np.array([[combiner_efficiency00], [combiner_efficiency01]]),
                                                np.array([[combiner_phase00],[combiner_phase10]]),
                                                delay=0)
combiner.add_output('o0')

# Waveguide Configuration
arm_length = 500e-6
default_phase = 2 * np.pi / wavelength * neff * arm_length
arm_0_amplitude = SymNum(name='arm0amp', default=1.0, product=True)
arm_0_phase = SymNum(name='arm0phase', default=default_phase, product=False)
arm_1_amplitude = SymNum(name='arm1amp', default=1.0, product=True)
arm_1_phase = SymNum(name='arm1phase', default=default_phase, product=False)

# Create Network
physnet = PhysicalNetwork()
physnet.add_device(splitter)
physnet.add_device(combiner)
physnet.add_devicelink(
    DeviceLink('splitter', 'combiner', 'o0', 'i0', phase=arm_0_phase, attenuation=arm_0_amplitude, delay=0))
physnet.add_devicelink(
    DeviceLink('splitter', 'combiner', 'o1', 'i1', phase=arm_1_phase, attenuation=arm_1_amplitude, delay=0))

# Visualize the network
physnet.visualize(path='./visualizations/mziweight', format='svg', full_graph=True)

# Evaluate and get HTML output
physnet.evaluate()
physnet.get_html_result('device:combiner:o0',path='./visualizations/equation.html')

# Get default output
output_name = physnet.outputs[0]
print(physnet.get_reduced_output(output_name))
# >>> (array([1.]), array([1.62146718]), array([0.]))

# Sweep Heater
dn_dT = 1e-4 # Thermooptic coefficient (index change per temperature change)
temperature_range = np.arange(0,20,0.01)

# Ideal splitter and combiner
amplitudes = np.zeros(shape=temperature_range.shape)
for i, temperature in enumerate(temperature_range):
    feed_dict = {'arm0phase': 2 * np.pi / wavelength * (neff+dn_dT*temperature) * arm_length}
    amp, phase, delay = physnet.get_reduced_output(output_name, feed_dict=feed_dict)
    amplitudes[i] = amp**2

plt.plot(temperature_range, amplitudes, label='Ideal Splitter and Combiner')

# Splitter non-ideal splitting ratio
amplitudes = np.zeros(shape=temperature_range.shape)
for i, temperature in enumerate(temperature_range):
    feed_dict = {'arm0phase': 2 * np.pi / wavelength * (neff+dn_dT*temperature) * arm_length, 'split_{01}':1/np.sqrt(2)*0.9, 'split_{00}':1/np.sqrt(2)}
    amp, phase, delay = physnet.get_reduced_output(output_name, feed_dict=feed_dict)
    amplitudes[i] = amp**2

plt.plot(temperature_range, amplitudes, label='Non-ideal splitter')

# Splitter and Combiner non-ideal
amplitudes = np.zeros(shape=temperature_range.shape)
for i, temperature in enumerate(temperature_range):
    feed_dict = {'arm0phase': 2 * np.pi / wavelength * (neff+dn_dT*temperature) * arm_length, 'split_{01}':1/np.sqrt(2)*0.9, 'split_{00}':1/np.sqrt(2),
                 'comb_{00}':1/np.sqrt(2), 'comb_{10}':1/np.sqrt(2)*0.8}
    amp, phase, delay = physnet.get_reduced_output(output_name, feed_dict=feed_dict)
    amplitudes[i] = amp**2

plt.plot(temperature_range, amplitudes, label='Non-ideal splitter and combiner')
plt.xlabel('Temperature')
plt.ylabel('|E|^2')
plt.legend(loc='best')
plt.grid()
plt.savefig('./visualizations/output_power.svg')

plt.show()
