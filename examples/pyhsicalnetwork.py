from colna.analyticnetwork import PhysicalNetwork, Device, DeviceLink
import numpy as np

# define a physical network
physnet = PhysicalNetwork()

# create a splitter and a combiner Device based on their scattering matrix
splitter = Device(name='0', devicetype='Splitter', scattering_matrix=np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)]]),
                  delay=1)
combiner = Device(name='0', devicetype='Combiner', scattering_matrix=np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]),
                  delay=1)

# add input and output nodes to the devices
splitter.add_input('i0', amplitude=1)
combiner.add_output('o0')

# add the devices to the physical network
physnet.add_device(splitter)
physnet.add_device(combiner)

# connect the devices using two DeviceLinks
physnet.add_devicelink(
    DeviceLink(startdevice='0', startdevicetype='Splitter', startnode='o0', enddevicetype='Combiner', enddevice='0',
               endnode='i0', phase=5 * np.pi, attenuation=0.5, delay=2))
physnet.add_devicelink(
    DeviceLink(startdevice='0', startdevicetype='Splitter', startnode='o1', enddevicetype='Combiner', enddevice='0',
               endnode='i1', phase=6 * np.pi, attenuation=0.7, delay=3)
)

# visualize the full and simplified
physnet.visualize(full_graph=True, path='./visualizations/Physical_Network_Full', format='svg')
physnet.visualize(full_graph=False, path='./visualizations/Physical_Network_Simplified', format='svg')

# evaluate network
physnet.evaluate()
out = physnet.get_outputs()
print(out)