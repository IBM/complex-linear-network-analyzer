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
