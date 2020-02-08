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
""" Example of a basic recurrent network.

Network topology:

A -> B
^    v
D <- C

"""
from colna.analyticnetwork import Network, Edge

###Define Network

nodes = ['a', 'b', 'c', 'd']
edges = [Edge('a', 'b', phase=0.5, attenuation=1.1, delay=1.0), # some edges can have gain, but the overall gain of loops must be <1
         Edge('b', 'c', phase=1, attenuation=0.9, delay=2.0),
         Edge('c', 'd', phase=0.2, attenuation=0.98, delay=0.5),
         Edge('d', 'a', phase=-0.5, attenuation=0.8, delay=1.5)]

net = Network()
for node in nodes:
    net.add_node(node)
for edge in edges:
    net.add_edge(edge)
net.add_input('a', amplitude=1.0)
net.visualize(path='./visualizations/recurrent')

####
#Evaluate Network
####
net.evaluate(amplitude_cutoff=1e-1, max_endpoints=1e6)


####
#Print data
####
print('paths leading to a:', net.get_paths('a'))
print('waves arriving at a:', net.get_result('a'))
net.print_stats()