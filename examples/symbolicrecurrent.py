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
""" Example of a basic recurrent network with symbolic variables.

Network topology:

A -> B
^    v
D <- C

"""
from colna.analyticnetwork import Network, Edge, SymNum


####
# Define Network
####
nodes = ['a', 'b', 'c', 'd']
edges = [Edge('a', 'b', phase=SymNum('ph1', default=0.4, product=False), attenuation=0.95, delay=1.0),
         Edge('b', 'c', .5, SymNum('amp1', default=0.95), 1.),
         Edge('c', 'd', .5, 0.95, SymNum('del1', default=1.2, product= False)),
         Edge('d', 'a', .5, 0.95, 1.)]

net = Network()
for node in nodes:
    net.add_node(node)
for edge in edges:
    net.add_edge(edge)
net.add_input('a', amplitude=1.0)

net.visualize(path='./visualizations/symbolicrecurrent')

####
# Evaluate Network
####
net.evaluate(amplitude_cutoff=1e-2, max_endpoints=1e6, use_shared_default=False)

print('paths leading to a:', net.get_paths('a'))
waves = [tuple([w.eval() if hasattr(w, 'eval') else w for w in inner]) for inner in net.get_result('a')]
print('waves arriving at a:', waves, '\n')
net.print_stats()

####
# Inserting variable values
###
waves = [tuple([w.eval(feed_dict={'amp1': .5, 'ph1': .2}) if hasattr(w, 'eval') else w for w in inner])
         for inner in net.get_result('a')]
print('waves arriving at a (different variable values):', waves, '\n')

####
# Showing symbolic values
####
print('symbolic values', net.get_result('a'), '\n')
