""" Example of a basic recurrent network.

Network topology:

A -> B
^    v
D <- C

"""

from colna.analyticnetwork import Network, Edge

###Define Network

nodes = ['a', 'b', 'c', 'd']
edges = [Edge('a', 'b', phase=0.5, attenuation=1.95, delay=1.0),
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