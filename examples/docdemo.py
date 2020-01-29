# +-----------------------------------------------------------------------------+
# |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                                       |
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

""" Creates a simple recurrent network with some symbolic edge properties and a testbench.

The network topology is a triangular network.
"""

from colna.analyticnetwork import Network, Edge, SymNum
from colna.analyticnetwork import Testbench
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
matplotlib.rc('text', usetex=False)
matplotlib.rc('axes', linewidth=2)
matplotlib.rc('lines',linewidth=2)
matplotlib.rc('font', **font)


net = Network()

amp1 = SymNum(name='v1', product=True)
amp2 = SymNum(name='v2', product=True)
amp3 = SymNum(name='v3', product=True)
phi3 = SymNum(name='v4', product=False)


net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')

net.add_edge(Edge(start='a',end='b',phase=2,attenuation=amp1,delay=1))
net.add_edge(Edge(start='b',end='c',phase=1,attenuation=amp2,delay=2))
net.add_edge(Edge(start='c',end='a',phase=phi3,attenuation=0.5*amp3,delay=3))
net.add_input('a')
net.add_input('b')

net.evaluate(amplitude_cutoff=0.001)
net.visualize(path='./visualizations/docdemo',format='png')

print(net.get_result('b'))
print(net.get_latex_result('b'))
net.get_html_result(['c','b'],path='./visualizations/docdemo_latex.html')
### Create a testbench with a feed dictionary
tb = Testbench(network=net, timestep=0.01, feed_dict={'v1':0.8,'v2':0.8,'v3':0.9,'v4':3})

x_in_a = np.sin(np.linspace(0,2*np.pi,500)) # create the input signal (Dimensino N)
t_in = np.linspace(0, 20, num=501) # create the input time vector (Dimension N+1)
tb.add_input_sequence(node_name='a',x=x_in_a,t=t_in)
tb.add_output_node(node_name='b')

# evaluate the network (through the testbench)
tb.evaluate_network(amplitude_cutoff=1e-3,use_shared_default=False)

# Calculate the output signal at the output nodes
tb.calculate_output(n_threads=0, use_shared_default=False) # uses multithreading with at most 8 threads
t, x = tb.t_out.transpose(), tb.x_out.transpose()

### Plot the signals
plt.figure(figsize=(6,4))
plt.plot(tb.input_t[0][:-1], np.abs(tb.input_x[0][:-1]), 'o') # Input signal
plt.plot(t, np.abs(x), 'x') # Output signal


### Format the plot
plt.xlabel('Time')
plt.ylabel('|x|')
plt.legend(['Input', 'Output'], loc='lower left')
plt.grid()
plt.tight_layout()
plt.savefig('./visualizations/docdemo_tb_output.png', dpi=600)
plt.show()