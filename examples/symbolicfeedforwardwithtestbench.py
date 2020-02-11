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
""" Creates a very simple recurrent network with some symbolic edge properties and a testbench.

The network topology is as follows:

A -> B > D
     v
     C
"""

from colna.analyticnetwork import Network, Edge, SymNum
from colna.analyticnetwork import Testbench
import numpy as np
import matplotlib.pyplot as plt

amp1 = SymNum(name='a1', default=1.5, product=True)
amp2 = SymNum(name='a2', default=2, product=True)
phi1 = SymNum(name='phi1', default=2, product=False)
phi2 = SymNum(name='phi2', default=3, product=False)

net = Network()

net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')
net.add_node(name='d')

net.add_edge(Edge(start='a',end='b',phase=phi1,attenuation=amp1,delay=1))
net.add_edge(Edge(start='b',end='c',phase=phi2,attenuation=amp2,delay=2))
net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

### Create a testbench with a feed dictionary
tb = Testbench(network=net, timestep=0.1, feed_dict={'a1':0.2,'a2':1.5,'phi1':2,'phi2':3})

x_in_a = np.sin(np.linspace(0,15,500))+1.5 # create the input signal (Dimensino N)
t_in = np.linspace(0, 10, num=501) # create the input time vector (Dimension N+1)
tb.add_input_sequence(node_name='a',x=x_in_a,t=t_in)
tb.add_output_node(node_name='c')
tb.add_output_node(node_name='d')

# evaluate the network (through the testbench)
tb.evaluate_network(amplitude_cutoff=1e-3,use_shared_default=False)

# Calculate the output signal at the output nodes
tb.calculate_output(n_threads=8, use_shared_default=False) # uses multithreading with at most 8 threads
t, x = tb.t_out.transpose(), tb.x_out.transpose()

### Plot the signals
plt.plot(tb.input_t[0][:-1], np.abs(tb.input_x[0][:-1]), 'o') # Input signal
plt.plot(t, np.abs(x), 'x') # Output signal


# Set a different feed dict and recompute the
tb.set_feed_dict({'a1':1.2,'a2':1.5,'phi1':2,'phi2':3})
tb.calculate_output(n_threads=8, use_shared_default=False) # uses multithreading with at most 8 threads
t, x = tb.t_out.transpose(), tb.x_out.transpose()

### Plot the signals
plt.plot(t, np.abs(x), 'x') # Output signal

### Format the plot
plt.xlabel('Time')
plt.ylabel('|x|')
plt.legend(['Input', 'Output C', 'Output D', 'Output C (Feed Dict 2)', 'Output D (Feed Dict 2)'], loc='lower left')
plt.grid()
# plt.savefig('./visualizations/symnum_feedforward_tb_output.svg')
plt.show()