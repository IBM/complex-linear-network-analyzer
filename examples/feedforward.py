""" Creates a simple feedforward network with constant input.

This example is part of the :ref:`User Manual`.

The network topology is as follows:

A > B > D
    v
    C
"""
from colna.analyticnetwork import Network, Edge

# Create a network
net = Network()

# Add nodes
net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')
net.add_node(name='d')

# Add edges
net.add_edge(Edge(start='a',end='b',phase=1,attenuation=0.8,delay=1))
net.add_edge(Edge(start='b',end='c',phase=2,attenuation=0.6,delay=2))
net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

# Add input
net.add_input(name='a',amplitude=1.0, phase=0)

# Visualize the network
net.visualize(path='./visualizations/feedforward', format='svg')

# Evaluate the network
net.evaluate(amplitude_cutoff=1e-3)

# Compute output and show results
print('paths leading to c:', net.get_paths('c'))
print('paths leading to d:', net.get_paths('d'))
print('waves arriving at c:', net.get_result('c'))
print('waves arriving at d:', net.get_result('d'))