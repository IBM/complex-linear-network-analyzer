from colna.analyticnetwork import Network, Edge, SymNum

# 2x2 grid (cycle)
# A -> B
# ^    v
# D <- C

####
# Define Network
####
nodes = ['a', 'b', 'c', 'd']
edges = [Edge('a', 'b', phase=SymNum('ph1', default=0.4, product=False), attenuation=0.95, delay=1.0),
         Edge('b', 'c', .5, SymNum('amp1', default=0.95), 1.),
         Edge('c', 'd', .5, 0.95, SymNum('del1', default=1.2, product= False, global_default=1.2)),
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
net.evaluate(amplitude_cutoff=1e-2, max_endpoints=1e6, use_global_default=False)

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
