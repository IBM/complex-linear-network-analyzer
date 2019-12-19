""" Creates a very simple feedforward network with constant input and some symbolic edge properties.

The network topology is as follows:

A -> B > D
     v
     C
"""

from colna.analyticnetwork import Network, Edge, SymNum, Testbench

amp1 = SymNum(name='a1', default=1.5, product=True, global_default=2)
amp2 = SymNum(name='a2', default=2, product=True, global_default=2)
phi1 = SymNum(name='phi1', default=2, product=False, global_default=3)
phi2 = SymNum(name='phi2', default=3, product=False, global_default=3)

net = Network()

net.add_node(name='a')
net.add_node(name='b')
net.add_node(name='c')
net.add_node(name='d')

net.add_edge(Edge(start='a',end='b',phase=phi1,attenuation=amp1,delay=1))
net.add_edge(Edge(start='b',end='c',phase=phi2,attenuation=amp2,delay=2))
net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

net.add_input(name='a',amplitude=1.0, phase=0)

net.visualize(path='./visualizations/symbolicfeedforward')

net.evaluate(use_global_default=False, feed_dict=None)

# print('paths leading to c:', net.get_paths('c'))
# print('paths leading to d:', net.get_paths('d'))

print('waves arriving at c:', net.get_result('c'))
print('waves arriving at d:', net.get_result('d'))

# Evaluation without feed dictionary, using the default value of each SymNum
waves = [tuple([w.eval(feed_dict=None, use_global_default=False) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation without feed dictionary, with global defaults
waves = [tuple([w.eval(feed_dict=None, use_global_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation with a feed dictionary
feed = {'a1': 1, 'a2': 2, 'phi1': 2, 'phi2': 4}
waves = [tuple([w.eval(feed_dict=feed, use_global_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')

# Evaluation with a partial feed dictionary
feed = {'a1': 0.5, 'phi2': 4}
waves = [tuple([w.eval(feed_dict=feed, use_global_default=True) if hasattr(w,'eval') else w for w in inner]) for inner in net.get_result('c')]
print('Waves arriving at c:', waves, '\n')
