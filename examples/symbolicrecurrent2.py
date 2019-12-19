from colna.analyticnetwork import Network, Edge, SymNum
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


class SymbolicEdge(Edge):
    def __init__(self, start, end, name=None, phase=.4, attenuation=.75, delay=1.0):
        super().__init__(start, end, phase, attenuation, delay)
        if name is None:
            name = "Edge_"+start+end
        self.attenuation = SymNum(name, default=0.75, global_default=0.75)


####
# Define Network
####
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
edges = [SymbolicEdge('a', 'b'), SymbolicEdge('a', 'c'), SymbolicEdge('b', 'd'), SymbolicEdge('c', 'd'),
         SymbolicEdge('c', 'e'), SymbolicEdge('d', 'f'), SymbolicEdge('f', 'e'), SymbolicEdge('e', 'g'),
         SymbolicEdge('f', 'h'), SymbolicEdge('h', 'g'), SymbolicEdge('g', 'i'), SymbolicEdge('e', 'k'),
         SymbolicEdge('i', 'j'), SymbolicEdge('i', 'k'), SymbolicEdge('j', 'l'), SymbolicEdge('l', 'k'),
         SymbolicEdge('l', 'n'), SymbolicEdge('k', 'm'), SymbolicEdge('n', 'm'), SymbolicEdge('n', 'p'),
         SymbolicEdge('p', 'o'), SymbolicEdge('m', 'o'), SymbolicEdge('o', 'a'), SymbolicEdge('m', 'c')]

net = Network()
for node in nodes:
    net.add_node(node)
for edge in edges:
    net.add_edge(edge)
net.add_input('a', amplitude=1.0)
for i, edge in enumerate(net.edges):
    edge.phase = np.random.uniform(0, 2 * np.pi)

####
# Evaluate Network
####
net.evaluate(amplitude_cutoff=1e-3, max_endpoints=1e6,use_global_default=True)
waves = [tuple([w.eval() if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('a')]

####
# Print and plot
####
for node in net.nodes:
    print('number of paths to ' + node + ':', len(net.get_paths(node)))
print('final path to a added:', net.get_paths('a')[-1])
print('10 first waves arriving at a:', waves[:10])
net.print_stats()

#####
# Reset variable
#####
feed = {"Edge_ab": 0.8}
waves = [tuple([w.eval(feed_dict=feed) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('a')]
print('10 first waves arriving at a:', waves[:10], '\n')

####
# Print Symbolic values
####
print('3 last waves at a (symbolic)', net.get_result('a')[-3:], '\n')

#####
# Plot
#####
phases = np.asarray([val[1].eval() if hasattr(val[1], 'eval') else val[1] for val in net.get_result('a')])
phases = phases % 2 * np.pi
amplitudes = np.asarray([val[0].eval() if hasattr(val[0], 'eval') else val[0] for val in net.get_result('a')])
plt.hist(phases, weights=amplitudes, bins=30)
plt.title("amplitude weighted, binned phase contributions to a")
plt.ylabel('amplitude')
plt.xlabel('phase')
plt.show()
