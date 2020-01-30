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
""" Creates a very simple feedforward network with constant input and some symbolic edge properties.

The network topology is as follows:

A -> B > D
     v
     C
"""

from colna.analyticnetwork import Network, Edge, SymNum, Testbench

amp1 = SymNum(name='a1', default=1.5, product=True)
amp2 = SymNum(name='a2', default=2.0, product=True)
phi1 = SymNum(name='phi1', default=2.0, product=False)
phi2 = SymNum(name='phi2', default=3.0, product=False)

net = Network()

net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')
net.add_node(name='d')

net.add_edge(Edge(start='a',end='b',phase=phi1,attenuation=amp1,delay=1))
net.add_edge(Edge(start='b',end='c',phase=phi2,attenuation=amp2,delay=2))
net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

net.add_input(name='a',amplitude=1.0, phase=0)

net.visualize(path='./visualizations/symbolicfeedforward', format='svg')

net.evaluate(use_shared_default=False, feed_dict=None)

# print('paths leading to c:', net.get_paths('c'))
# print('paths leading to d:', net.get_paths('d'))

print('waves arriving at c:', net.get_result('c'))
print('waves arriving at d:', net.get_result('d'))
net.get_html_result(['c','d'],path='./visualizations/symbolicfeedforward.html')
# Evaluation without feed dictionary, using the default value of each SymNum
waves = [tuple([w.eval(feed_dict=None, use_shared_default=False) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation without feed dictionary, with global defaults
waves = [tuple([w.eval(feed_dict=None, use_shared_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation with a feed dictionary
feed = {'a1': 1, 'a2': 2, 'phi1': 2, 'phi2': 4}
waves = [tuple([w.eval(feed_dict=feed, use_shared_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation with a partial feed dictionary
feed = {'a1': 0.5, 'phi2': 4}
waves = [tuple([w.eval(feed_dict=feed, use_shared_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')
