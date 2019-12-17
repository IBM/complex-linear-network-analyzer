""" Creates a very simple feedforward network with constant input.

This is the example used in the basic usage guide.
The network topology is as follows:

A -> B > D
     v
     C
"""

from colna.analyticnetwork import Network, Edge


net = Network()

net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')
net.add_node(name='d')

net.add_edge(Edge(start='a',end='b',phase=1,attenuation=0.8,delay=1))
net.add_edge(Edge(start='b',end='c',phase=2,attenuation=0.6,delay=2))
net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

net.add_input(name='a',amplitude=1.0, phase=0)

net.evaluate(amplitude_cutoff=1e-3)

print('paths leading to c:', net.get_paths('c'))
print('paths leading to d:', net.get_paths('d'))

print('waves arriving at c:', net.get_result('c'))
print('waves arriving at d:', net.get_result('d'))


# net.visualize(path='simple_network')