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

from colna.analyticnetwork import Network, Edge
import numpy as np
import matplotlib.pyplot as plt

# 4x4 network
# P <- N <- L <- J
# v    v    v    ^
# O <- M <- K <- I
# v    v    ^    ^
# A -> C -> E -> G
# v    v    ^    ^
# B -> D -> F -> H


####
# Define Network
####
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
edges = [Edge('a', 'b'), Edge('a', 'c'), Edge('b', 'd'), Edge('c', 'd'), Edge('c', 'e'), Edge('d', 'f'),
         Edge('f', 'e'), Edge('e', 'g'), Edge('f', 'h'), Edge('h', 'g'), Edge('g', 'i'), Edge('e', 'k'),
         Edge('i', 'j'), Edge('i', 'k'), Edge('j', 'l'), Edge('l', 'k'), Edge('l', 'n'), Edge('k', 'm'),
         Edge('n', 'm'), Edge('n', 'p'), Edge('p', 'o'), Edge('m', 'o'), Edge('o', 'a'), Edge('m', 'c')]

net = Network()
for node in nodes:
    net.add_node(node)
for edge in edges:
    net.add_edge(edge)
net.add_input('a', amplitude=1.0)
for edge in net.edges:
    edge.attenuation = 0.75
    edge.phase = np.random.uniform(0, 2 * np.pi)


net.visualize(path='./visualizations/mediumexample')
####
# Evaluate Network
####
net.evaluate(amplitude_cutoff=1e-3, max_endpoints=1e6)

####
# Print and plot
####
for node in net.nodes:
    print('number of paths to ' + node + ':', len(net.get_paths(node)))
print('final path to a added:', net.get_paths('a')[-1])
net.print_stats()

phases = np.asarray([val[1] for val in net.get_result('a')])
phases = phases % 2 * np.pi
amplitudes = np.asarray([val[0] for val in net.get_result('a')])
plt.hist(phases, weights=amplitudes, bins=30)
plt.title("amplitude weighted, binned phase contributions to a")
plt.ylabel('amplitude')
plt.xlabel('phase')
plt.show()
