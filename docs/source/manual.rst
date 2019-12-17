User Manual
=================================

Installation
------------

This package runs on Python 3.5+. Use pip to install it. Download/Clone the repository on your computer and run the
install command in the base directory of the package (ComplexLinearNetworkAnalyzer).

.. code-block:: console

    pip install .

All dependencies required for the core features will be installed automatically.

Optional Features
#################

Networks can be visualized using Graphviz. If you intend to use this feature you need to install Graphviz manually
and add it to your system path (get Graphviz here `<https://www.graphviz.org/>`_). Afterwards use

.. code-block:: console

    pip install .[Visualization]

to install the COLNA package and all its dependencies, including the Graphviz Python wrapper.

Basic Usage
-----------
A simple network
################

The fundamental class of the COLNA module is the :class:`.Network` class. The network class describes
the network through which we propagate a signal. Networks are defined by nodes and edges (class :class:`.Edge`).

You can create a network by instantiating a new :class:`.Network` object:

.. code-block:: python

    from colna.analyticnetwork import Network
    net = Network()
    print(net)
    >>> <colna.analyticnetwork.Network object at 0x...>

You can add nodes and edges to the network using its :meth:`~.Network.add_node` and :meth:`~.Network.add_edge` methods.
To create edges you need to import the :class:`.Edge` class as well.

.. code-block:: python

    from colna.analyticnetwork import Network, Edge
    net = Network()

    net.add_node(name='a')
    net.add_node(name='b')
    net.add_node(name='c')
    net.add_node(name='d')

    net.add_edge(Edge(start='a',end='b',phase=1,attenuation=0.8,delay=1))
    net.add_edge(Edge(start='b',end='c',phase=2,attenuation=0.6,delay=2))
    net.add_edge(Edge(start='b',end='d',phase=3,attenuation=0.4,delay=3))

The :meth:`~.Network.add_node` method takes a node name as argument, :meth:`~.Network.add_edge` takes an :class:`.Edge` object as
argument. The Edge initializer takes the name of the start and end node (by node name) and edge properties (phase, attenuation and delay).

The network initialized before looks as follows.

.. _simplenetworklabel:
.. figure:: /figures/simple_network.svg
    :align: center

    The labels at the edges give the attenuation (a), phase (p) and delay (d) of the respective edge.

In the next step you should add a constant input and then you can evaluate the network.

.. code-block:: python

    net.add_input(name='a',amplitude=1.0, phase=0)
    net.evaluate()

The :meth:`~.Network.add_input` method takes the name of the node where the constant signal is injected and it's amplitude and phase.
The :meth:`~.Network.evaluate` evaluates the network, which means it computes all paths leading from the input node(s) to each node.
You can print the evaluated paths using the :meth:`~.Network.get_path`, which takes a node name as argument.

.. code-block:: python

    print('paths leading to c:', net.get_paths('c'))
    print('paths leading to d:', net.get_paths('d'))

    >>> paths leading to c: ['-a-b-c']
    >>> paths leading to d: ['-a-b-d']

You can calculate the waves arriving at the output node, for this we use the

.. code-block:: python

    print('waves arriving at c:', net.get_result('c'))
    print('waves arriving at d:', net.get_result('d'))

    >>> waves arriving at c: [(0.48, 3, 3.0)]
    >>> waves arriving at d: [(0.32000000000000006, 4, 4.0)]

If you have installed the visualization feature (see :ref:`Installation`), you can visualize the graph by running:

.. code-block:: python

    net.visualize(path='simple_network')

The visualization method creates a dot file (at the given output path) and renders it into a pdf file, using Graphviz.
The resulting visualization is shown in :ref:`the figure above<simplenetworklabel>`.

Testbench
#########

So far we have only injected constant signals into the network. To inject time dependant signals, we can use a :class:`.Testbench` object.
The Testbench is used to inject signals to nodes of the network and read the output. This is illustrated in the figure
below.

.. image:: /figures/network_testbench_diagram.svg
    :align: center






Specialized classes for physical systems and symbolic computation that extend the basic functionality are available and will be discussed later on in this guide.


Basic Concepts
--------------

Networks
#########

Networks consist of nodes and directed edges. Networks are represented by a directed graph. COLNA computes all paths
leading to the output nodes (including recurrent paths) down to a certain accuracy threshold.

.. note::

  If a network contains recurrent paths (loops), the user must ensure that there is no gain in the network (i.e. attenuation < 1), otherwise the amplitude at the output will never fall below the threshold.

Nodes
~~~~~

Nodes are linear.

Edges
~~~~~


Edges connect two nodes in a directed manner. Edges add a phase shift (:math:`\phi`), delay (:math:`d`) and
attenuation (:math:`a`) to the signal. The input to output relation of an edge is given by:

.. math::
    x_{out}(t) = a \cdot x_{in}(t-d) \cdot e^{j\phi}

Edge properties can be constant or symbolic numbers (variables).

Physical Networks
##################

PhysicalNetwork is a child class of Network that allows for a more natural implementation of physical (hardware) networks.
Physical networks are made out of devices and device links, which connect devices.

Devices
~~~~~~~

A device has a number of input and output ports and the input-to-output relation is given by a scattering matrix and a
delay. Device is a child class of network. It provides convenience methods to create the device from it's complex
scattering matrix (matrix describing input-output relation). Nodes are renamed automatically, based on the device type,
device name and port number.

If the network is visualized, devices can be shown as single blocks or as a full blown network visualization in the diagram.

Devicelink
~~~~~~~~~~

A devicelink is an edge that connects to devices.  Devicelinks are given the name of source and target device as well as
source and target node within the device. Otherwise they function like the parent class Edge.

Testbench
#########

