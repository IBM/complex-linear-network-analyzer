# +-----------------------------------------------------------------------------+
# |  Copyright 2019 IBM Research - Zurich                                       |
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

from time import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle
import warnings
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing


class Edge(object):
    """
    Edges connect two nodes in a directed manner. Edges add a phase shift (:math:`\phi`), delay (:math:`d`) and
    attenuation (:math:`a`) to the signal. The input to output relation of an edge is given by:

    .. math::
        x_{out}(t) = a \cdot x_{in}(t-d) \cdot e^{j\phi}

    Edge properties can be constant or symbolic numbers (variables).
    """

    def __init__(self, start, end, phase=.4, attenuation=.8, delay=1.0):
        """
        :param start: name of start vertex connected by this edge
        :param end: name of end vertex connected by this edge
        :param phase: phase shift added by this element (stacks additively)
        :param attenuation: attenuation caused by element (stacks multiplicatively)
        :param delay: time delay added by this element (stacks additively)
        """
        self.start = start
        self.end = end
        self.phase = phase
        self.attenuation = attenuation
        self.delay = delay


class Network(object):
    """
    Networks consist of linear nodes and directed edges. Networks are represented by a directed graph. COLNA computes all paths
    leading to the output nodes (including recurrent paths) down to a certain accuracy threshold.

    .. note::

      If a network contains recurrent paths (loops), the user must ensure that there is no gain in the network (i.e. attenuation < 1), otherwise the amplitude at the output will never fall below the threshold.

    """

    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.edges = []
        self.nodes_to_output = {}

    def add_node(self, name):
        """
        add a new node to the network

        :param name: the name of the node
        :type name: string
        """
        assert not (name in self.nodes), "node of name " + name + " already exists"
        self.nodes.append(name)

    def add_edge(self, edge):
        """
        Add a new edge to the network

        :param edge: the edge object to add
        """
        assert edge.start in self.nodes, "attempted to add edge from undefined node"
        assert edge.end in self.nodes, "attempted to add edge to undefined node"
        self.edges.append(edge)

    def add_input(self, name, amplitude=1.0, phase=0.0, delay=0.0):
        """
        Define input points of network. The evaluation assumes signals with the given amplitude, phase and delay are
        propagating through the network when computing the analytical waveforms at each node.

        :param name: name of the node that is to receive the input
        :param amplitude: amplitude of the input
        :param phase: phase of the input (relative to other inputs)
        :param delay: delay of the input (relative to other inputs)
        """
        assert name in self.nodes, "attempted to give input to inexistent node " + name
        self.inputs.append((amplitude, phase, delay, name))

    def get_result(self, name):
        """
        Returns a list of waves mixing at this node

        :param name: name of the node to get result from
        :return: a list of waves mixing at this node (amplitude, phase, delay)
        """
        assert name in self.nodes, "node does not exist"
        return [entry[0:3] for entry in self.nodes_to_output[name]]

    def get_result_np(self, name):
        """
        Returns a result at a given node as numpy array

        :param name: name of the node to get result from
        :return: x; x[0]: amp, x[1]: phase, x[2]: delay
        """
        assert name in self.nodes, "node does not exist"
        amp = np.asarray([entry[0] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        phase = np.asarray([entry[1] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        delay = np.asarray([entry[2] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        return np.concatenate([amp, phase, delay], 0)

    def get_paths(self, name):
        """
        Find all paths leading to node.

        :return: all paths leading to a node
        """
        assert name in self.nodes, "node does not exist"
        return [entry[3] for entry in self.nodes_to_output[name]]

    def print_stats(self):
        """
        Prints some statistics of the evaluated network
        """
        n_paths = sum([len(val) for val in self.nodes_to_output.values()])
        print('total number of paths:', n_paths)

    @staticmethod
    def stopping_criterion(amplitude, cutoff):
        return amplitude < cutoff

    def evaluate(self, amplitude_cutoff=0.01, max_endpoints=100000, use_global_default=True, feed_dict=None,
                 hide_tqdm_progress=False):
        """
        Evaluate the network.

        :param amplitude_cutoff:  amplitude below which a wave is not further propagated through the network
        :param max_endpoints: evaluation is interrupted early, if more than max_endpoints exist in evaluation
        :param use_global_default: set to true if global defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: True
        :type use_global_default: Boolean
        :param feed_dict: Feed dictionary for SymNum variables. Default: None
        :return:
            updates self.nodes_to_output
            a dictionary whose keys are node names. For each node name a list of quadruplets is given
            [(amplitude, phase, delay, path), (amplitude, phase, delay, path), ...].

            .. note::
                phases are not reset to a finite range, but simply added

        """

        for node in self.nodes:
            self.nodes_to_output[node] = []

        # at the start of the evaluation the endpoints are at the inputs
        current_endpoints = [input[3] for input in self.inputs]
        endpoint = {'point': [input[3] for input in self.inputs],
                    'delay': [input[2] for input in self.inputs],
                    'phase': [input[1] for input in self.inputs],
                    'amp': [input[0] for input in self.inputs],
                    'path': ['' for node in current_endpoints]}

        # keep propagating waves, while there is a front endpoint that is above the amplitude cutoff
        pbar = tqdm(disable=hide_tqdm_progress, unit='paths', desc='Network evaluation in progress')

        while len(current_endpoints) > 0:
            assert len(current_endpoints) < max_endpoints, "evaluation interrupted, too many endpoints"

            # in these we will collect the parameters of the next endpoints
            new_endpoints, new_delays, new_phases, new_amplitudes, new_paths = [], [], [], [], []

            # iterate over all current endpoints
            for node_index, node in enumerate(current_endpoints):
                # add the current endpoint to the final output
                self.nodes_to_output[node].append((endpoint['amp'][node_index],
                                                   endpoint['phase'][node_index],
                                                   endpoint['delay'][node_index],
                                                   endpoint['path'][node_index] + '-' + node))

                # check if any edge's start is the current endpoint
                for edge in self.edges:
                    current_attn = (endpoint['amp'][node_index] * edge.attenuation)
                    current_attn_fl = current_attn.eval(feed_dict=feed_dict,
                                                        use_global_default=use_global_default) if hasattr(current_attn,
                                                                                                          'eval') else current_attn
                    if (node == edge.start
                            and not self.stopping_criterion(current_attn_fl, amplitude_cutoff)):
                        # if yes, add the new endpoint to the new endpoints (unless the amp. is too low)
                        new_endpoints.append(edge.end)
                        new_delays.append(endpoint['delay'][node_index] + edge.delay)
                        new_phases.append(endpoint['phase'][node_index] + edge.phase)
                        new_amplitudes.append(current_attn)
                        new_paths.append(endpoint['path'][node_index] + '-' + node)

                pbar.update(1)

            # set the current endpoint parameters to the new ones and go to the top
            current_endpoints = new_endpoints
            endpoint['delay'] = new_delays
            endpoint['amp'] = new_amplitudes
            endpoint['phase'] = new_phases
            endpoint['path'] = new_paths

    def visualize(self, show_edge_labels=True, path='network.gv', skip_colon=False, format='pdf'):
        try:
            from graphviz import Digraph
        except ModuleNotFoundError as err:
            warnings.warn("Graphviz Package was not found, visualization is skipped.")
            return 0
        s = Digraph('structs', graph_attr={'ranksep': '0.5', 'overlap': 'false', 'splines': 'true', 'rankdir': 'TB',
                                           'constraint': 'true', 'nodesep': '2'}, node_attr={'shape': 'record'},
                    edge_attr={}, engine='dot')

        for node in self.nodes:
            if skip_colon and ':' in node:
                continue
            s.node(node, node)

        for edge in self.edges:
            if show_edge_labels:
                s.edge(edge.start.replace(":", ""), edge.end.replace(":", ""),
                       label='a{}, p{}, d{}'.format(edge.attenuation, edge.phase, edge.delay))
            else:
                s.edge(edge.start.replace(":", ""), edge.end.replace(":", ""))

        s.render(path, view=False, format=format)


class SymNum:
    """
    symbolic numbers for edge properties in analytic networks
    """

    def __init__(self, name, default=0.9, product=True, global_default=0.8, numerical=None):
        """
        Symbolic numbers for the analytic network.

        Instantiates a symbolic number (variable) of name 'name' with default value 'default'. 'product' is used
        to distinguish attenuations (stacking multiplicatively) and phases / delays (stacking additively).

        :param name: the name of the variable. The name should be unique for each SymNum present in the network.
        :param default: the default value substituted, when we evaluate this variable
        :param product: whether this variable is composed as a product (True) or a sum (False)
        :param global_default: this is assumed to be the value of the variable when we evaluate the network if use_global_defaults is set.
        :param numerical: initial value of numerical part (numerical factor for product variables, numerical addition for additive variables). Can be set to none for automatic initialization (1.0 for product variables, 0.0 for additive variables)
        """
        # the numerical part of the number's value
        self.numerical = numerical if numerical is not None else 1.0 * product

        # the symbolic part of the number's value
        self.symbolic = {name: 1}

        # dict. of the values that are inserted for each variable if no particular value is specified at evaluation
        self.defaults = {name: default}

        # indicates whether the number stacks multiplicatively or additively
        self.product = product

        # when the network is evaluated we use this value to determine when to cut the path due to attenuation limit
        self.global_default = global_default

    def __copy__(self):
        copy = SymNum('', product=self.product, global_default=self.global_default, numerical=self.numerical)
        copy.symbolic = deepcopy(self.symbolic)
        copy.defaults = deepcopy(self.defaults)
        return copy

    def __add__(self, other):
        assert not self.product, "do not add product variables"
        copy = self.__copy__()
        if isinstance(other, SymNum):
            assert self.product == other.product, "ensure that product type of symbolic numbers are the same"
            copy.numerical += other.numerical
            for symbol in other.symbolic.keys():
                if not (symbol in copy.symbolic.keys()):
                    copy.symbolic[symbol] = 1
                    copy.defaults[symbol] = other.defaults[symbol]
                else:
                    copy.symbolic[symbol] += other.symbolic[symbol]
        else:
            copy.numerical += other
        return copy

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        assert self.product, "do not multiply non-product variables"
        copy = self.__copy__()
        if isinstance(other, SymNum):
            assert self.product == other.product, "ensure that product type of symbolic numbers are the same"
            copy.numerical *= other.numerical
            for symbol in other.symbolic.keys():
                if not (symbol in copy.symbolic.keys()):
                    copy.symbolic[symbol] = 1
                    copy.defaults[symbol] = other.defaults[symbol]
                else:
                    copy.symbolic[symbol] += other.symbolic[symbol]
        else:
            copy.numerical *= other
        return copy

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert self.product, "do not multiply non-product variables"
        copy = self.__copy__()
        if isinstance(other, SymNum):
            assert self.product == other.product, "ensure that product type of symbolic numbers are the same"
            copy.numerical /= other.numerical
            for symbol in other.symbolic.keys():
                if not (symbol in copy.symbolic.keys()):
                    copy.symbolic[symbol] = -1
                    copy.defaults[symbol] = other.defaults[symbol]
                else:
                    copy.symbolic[symbol] -= other.symbolic[symbol]
        else:
            copy.numerical /= other
        return copy

    def eval(self, feed_dict=None, verbose=False, use_global_default=True):
        """
        evaluate the numerical value of the number

        :param feed_dict: a dictionary specifying values of variables by name(if not given default values are used)
        :param verbose: print the number in symbolic form before returning
        :param use_global_default: set to true if global defaults should be used with SymNum's (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not none. Default: True
        :type use_global_default: Boolean
        :return: numerical value of the number (float)
        """
        symbolic_evaluated = 1 if self.product else 0
        if feed_dict is None and use_global_default == True:
            total_count = sum(self.symbolic.values())
            symbolic_evaluated = self.global_default ** total_count if self.product else self.global_default * total_count
        else:
            for symbol in self.symbolic.keys():
                val = self.defaults[symbol] if (feed_dict is None) or not (symbol in feed_dict.keys()) else feed_dict[
                    symbol]
                if self.product:
                    symbolic_evaluated *= val ** self.symbolic[symbol]
                else:
                    symbolic_evaluated += val * self.symbolic[symbol]
        if verbose:
            print(self.symbolic.keys(), self.symbolic.values(), self.numerical, symbolic_evaluated)
        return symbolic_evaluated * self.numerical if self.product else symbolic_evaluated + self.numerical

    def __str__(self):
        op1 = " * " if self.product else " + "
        op2 = "**" if self.product else "*"
        out = ""
        for key in self.symbolic.keys():
            out += op1
            out += key + op2 + str(self.symbolic[key])

        return str(self.numerical) + out

    def __repr__(self):
        """
        convenience overload to print lists of these numbers easily.
        """
        return '<SymNum{' + str(self)+'}>'


class Device(Network):
    """
    Defines a device object for physical networks.

    All nodes in a device get their 'name' changed to 'devicetype:devicename:name'
    """

    def __init__(self, name, devicetype='device', scattering_matrix=None, delay=None):
        """
        :param scattering_matrix: device scattering matrix, if both :attr:`scattering_matrix` and :attr:`delay` are provided, the device will be initalized with the corresponding scattering matrix and delay
        :param delay: device delay, if both :attr:`scattering_matrix` and :attr:`delay` are provided, the device will be initalized with the corresponding scattering matrix and delay
        :param name: name of the device (str)
        :param devicetype: typename of the device (str)
        """
        super().__init__()
        self.name = name
        self.devicetype = devicetype
        self.outputs = []
        if scattering_matrix is not None and delay is not None:
            self.init_from_scattering_matrix(scattering_matrix, delay)

    def add_output(self, nodename):
        """
        adds a node (by name) to the outputs of the physical network

        :param nodename: name (str) of the node to add to the outputs
        """
        name = self.devicetype + ":" + self.name + ":" + nodename
        assert name in self.nodes, "attempted to add output to inexistent node"
        self.outputs.append(name)

    def add_input(self, nodename, amplitude=1.0, phase=0.0, delay=0.0):
        """
        adds an external input to a node

        :param nodename: name (str) of the node
        :param amplitude: amplitude of the input
        :param phase: phase shift of the input
        :param delay: time delay of the input
        """
        name = self.devicetype + ":" + self.name + ":" + nodename
        super().add_input(name, amplitude, phase, delay)

    def add_node(self, nodename):
        """
        add a new node to the network

        :param nodename: the name of the node (a string, not containing ':')
        """
        assert ":" not in nodename, "avoid use of : in nodenames"
        name = self.devicetype + ":" + self.name + ":" + nodename
        super().add_node(name)

    def add_edge(self, edge):
        """
        add a new edge to the network

        :param edge: the edge object to add
        """
        edge.start = self.devicetype + ":" + self.name + ":" + edge.start
        edge.end = self.devicetype + ":" + self.name + ":" + edge.end
        super().add_edge(edge)

    def init_from_scattering_matrix(self, smatrix, delay=1.0):
        """
        defines a device by its scattering matrix (complex matrix) and a delay

        for a device with n inputs and m outputs pass a nxm complex matrix specifying attenuation and phase-shift
        creates n+m nodes linked by n*m edges with the given parameters, all with time delay 'delay'
        input node number k will be named 'i+str(k)'
        output node number k will be named 'o+str(k)'

        :param smatrix: complex matrix defining the input to output mapping
        :param delay: scalar time delay
        """
        smatrix = np.atleast_2d(smatrix)
        n_in = smatrix.shape[0]
        n_out = smatrix.shape[1]

        for i in range(n_in):
            in_name = "i" + str(i)
            self.add_node(in_name)
            for j in range(n_out):
                out_name = "o" + str(j)
                name = self.devicetype + ":" + self.name + ":" + out_name
                if name not in self.nodes:
                    self.add_node(out_name)
                edge = Edge(in_name, out_name,
                            phase=np.angle(smatrix[i, j]),
                            attenuation=np.absolute(smatrix[i, j]),
                            delay=delay)
                self.add_edge(edge)


class DeviceLink(Edge):
    """
    Defines a device link object for physical networks.

    Device links are special edges that link devices. They are given the name of source and target device as well as
    source and target node within the device. Otherwise they function like the parent class Edge.

    :param startdevice: name of the start device (string)
    :param enddevice: name of the end device (string)
    :param startnode: name of the node within the start device (string)
    :param endnode: name of the node within the end device (string)
    :param startdevicetype: name of the device type of startdevice (optinal, defaults to 'device')
    :param enddevicetype: name of the device type of enddevice (optinal, defaults to 'device')
    :param phase: phase shift of the device link
    :param attenuation: attenuation of the device link
    :param delay: time delay of the device link
    """

    def __init__(self, startdevice, enddevice, startnode, endnode, startdevicetype='device', enddevicetype='device',
                 phase=.4, attenuation=.8, delay=1.0):
        for string in [startdevice, enddevice, startnode, endnode, startdevicetype, enddevicetype]:
            assert isinstance(string, str), "device links require string inputs on some arguments"
        super().__init__(startnode, endnode, phase, attenuation, delay)
        self.start = startdevicetype + ":" + startdevice + ":" + startnode
        self.end = enddevicetype + ":" + enddevice + ":" + endnode


class PhysicalNetwork(Network):
    """
    Defines a physical network for simulation.

    Extension of the Network class that allows for a more natural implementation of physical networks using
    a description as a collection of devices, device links, input and output sites.
    """

    def __init__(self):
        super().__init__()
        self.outputs = []

    def add_device(self, device):
        """
        add an device to the network

        :param device: device to add
        """
        for node in device.nodes:
            self.add_node(node)
        for edge in device.edges:
            self.add_edge(edge)
        for input in device.inputs:
            self.add_input(input[3], input[0], input[1], input[2])
        for output in device.outputs:
            self.outputs.append(output)

    def add_devicelink(self, devicelink):
        """
        add a device link to the network

        :param devicelink: the device link to add
        """
        assert devicelink.start in self.nodes, "attempted to add device link from inexistent node " + devicelink.start
        assert devicelink.end in self.nodes, "attempted to add device link to inexistent node " + devicelink.end
        self.add_edge(devicelink)

    def get_outputs(self):
        """
        get the computed wave forms at all output sites

        :return: a list of the output waves at all output nodes
        """
        return [self.get_result(name) for name in self.outputs]

    def visualize(self, show_edge_labels=True, path='network.gv', format='pdf', full_graph=False):
        """
        Draws the network graph.

        :param show_edge_labels: If true labels showing attenuation, phase and delay for each edge are drawn in the graph.
        :param path: output path and filename
        :param full_graph: if true, inner edges of devices are shown as well
        """
        if full_graph:
            super().visualize(show_edge_labels, path, True, format=format)
        else:
            try:
                from graphviz import Digraph
            except ModuleNotFoundError as err:
                warnings.warn("Graphviz Package was not found, visualization is skipped.")
                return 0
            s = Digraph('structs', graph_attr={'ranksep': '0.5', 'overlap': 'false', 'splines': 'true', 'rankdir': 'TB',
                                               'constraint': 'true', 'nodesep': '1'}, node_attr={'shape': 'record'},
                        edge_attr={}, engine='dot')
            previous_node = ''

            nodenames = []
            for node in self.nodes:
                nodesplit = node.split(':')
                nodenames.append(nodesplit[0] + nodesplit[1])

            for node in np.unique(nodenames):
                s.node(node, node)

            for edge in self.edges:

                startsplit = edge.start.split(':')
                edgestart = startsplit[0] + startsplit[1]

                stopslit = edge.end.split(':')
                edgeend = stopslit[0] + stopslit[1]

                if edgestart != edgeend:
                    if show_edge_labels == True:
                        s.edge(edgestart, edgeend,
                               label='a{}, p{}, d{}'.format(edge.attenuation, edge.phase, edge.delay))
                    else:
                        s.edge(edgestart, edgeend)

            s.render(path, format=format, view=False)


class Testbench():
    """ Class that allows to find the output sequence for a given input signal sequence. """

    def __init__(self, network: Network, timestep=1, disable_progress_bars=True, feed_dict=None) -> object:
        """

        :param timestep: timestep on which the network is evaluated. Ideally it should be a factor of all delays in the network.
        :param network: network where signals should be applied
        """

        self.inputs = []
        self.input_x = []
        self.input_t = []
        self.input_nodes = []
        self.output_nodes = []
        self.timestep = timestep
        self.model = network
        self.model.inputs = []
        self.t0 = 0
        self.t1 = 0
        self.disable_tqdm = True
        self.feed_dict = feed_dict

    def clear_inputoutput(self):
        """
        Clears the input and output lists which store the input nodes and corresponding signals and the output node names.
        """
        self.inputs = []
        self.input_x = []
        self.input_t = []
        self.input_nodes = []
        self.output_nodes = []
        self.model.inputs = []

    def set_feed_dict(self, feed_dict):
        self.feed_dict = feed_dict

    def add_output_node(self, node_name):
        self.output_nodes.append(node_name)

    def add_input_sequence(self, node_name, x, t):
        """
        Add input signal sequence to network node

        :param node_name: Name of node that receives the input
        :param x: input signal (complex), dimension: 1xN
        :param t: time of signal, if none we assume that the signal vector provides the signal at each time step. The value x[n] is applied to the input Node during the right-open time interval [t[n], t[n+1]), dimension: 1x(N+1). Time values must be in increasing order.
        """
        if len(x) + 1 != len(t):
            raise ValueError("Size of time vector does not match size of signal vector.")

        if not node_name in self.model.nodes:
            raise NameError("attempted to give input to inexistent node " + node_name)

        if node_name in self.input_nodes:
            raise OverflowError("At most one input sequence can be added per node. You added two at " + node_name)

        self.model.add_input(node_name)
        self.input_nodes.append(node_name)
        self.input_x.append(x)
        self.input_t.append(t)

    def _extract_min_max_signal_time(self):
        """
        Extracts the start and stop time of all signals. Stores the minimum (self.t0) and maximum (self.t1)
        """
        self.t0 = min([time[0] for time in self.input_t])
        self.t1 = max([time[-1] for time in self.input_t])

    def _prepare_signals(self):
        """
        Converts all signals to cover the time interval [self.t0, self.t1].
        Signals are set to zero whenever they were not defined.
        Use `extract_min_max_signal_time` to extract t0 and t1 automatically from the provided input signals.
        """
        for i, x in enumerate(tqdm(self.input_x, disable=self.disable_tqdm)):
            t = self.input_t[i]
            t_s, x_s = self._convert_signal_to_timestep(x=x, t=t, timestep=self.timestep)
            self.input_t[i] = t_s
            self.input_x[i] = x_s

    def _convert_signal_to_timestep(self, x, t, timestep):
        """
        Resamples the input signals with sampling rate according to timestep. Sets signal to 0 if the signal is not defined at that time.

        :param x: Signal vector
        :param t: Time vector
        :param timestep: timestep to which signal is resampled
        :return: t_sampled, x_sampled: resampled data and time
        """
        t_sampled = np.linspace(start=self.t0, stop=self.t1, num=1 + round((self.t1 - self.t0) / timestep))
        x_sampled = self._interpolate_constant(x=t_sampled, xp=t, yp=x)
        return t_sampled, x_sampled

    def _interpolate_constant(self, x, xp, yp):
        """
        Interpolation method that interpolates signals to the nearest left neighbour of the sampling point.

        This sampling is used, as input signal y[n] is defined to be applied at during a right open time interval
        [t[n], t[n+1]).

        :param x: x coordinates where signal should be sampled
        :param xp: x coordinate of signal to be sampled, assuming array is sorted in increasing order (typically time vector)
        :param yp: y coordinate of signal to be sampled
        :return: sampled signal, signal is set to zero in regions outside of
        """
        # Interpolate only in the range where xp (typically time vector) is defined, for sampling values outside of this range create zero values.
        x_l = np.searchsorted(x, xp[0])
        x_r = np.searchsorted(x, xp[-1])
        indices = np.searchsorted(xp, x[x_l:x_r], side='right')
        y = np.concatenate(([0], yp))
        # create zeros for times where the signal
        z_l = np.zeros(shape=(x_l,), dtype=np.complex128)
        z_r = np.zeros(shape=(len(x) - x_r,), dtype=np.complex128)
        return np.concatenate((z_l, y[indices], z_r))

    def evaluate_network(self, amplitude_cutoff=1e-3, max_endpoints=1e6, use_global_default=True):
        """
        Evaluates the network.

        :param amplitude_cutoff:
        :param max_endpoints:
        :param use_global_default: set to true if global defaults should be used with SymNum's (higher speed) when no\
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy).\
        The value is ignored if feed_dict is not none. Default: True
        :type use_global_default: Boolean
        """
        self.model.evaluate(amplitude_cutoff, max_endpoints, use_global_default=use_global_default,
                            feed_dict=self.feed_dict)

    def calculate_output(self, use_global_default=False, n_threads=0):
        """
        Calculates the output signals for the given input signals.

        :param use_global_default: set to true if global defaults should be used with SymNum's (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not none. Default: False
        :param n_threads: Number of threads that are used for evaluation (set to 0 to disable multithreading)
        :type use_global_default: Boolean

        """

        self.x_out = []
        self.t_out = []

        if n_threads == 0:
            for ind, node_out in enumerate(self.output_nodes):
                assert node_out in self.model.nodes, "node {} does not exist".format(node_out)
                t, x = self.calculate_output_sequence(node_name=node_out, use_global_default=use_global_default)
                self.x_out.append(x)
                self.t_out.append(t)

        else:
            args = []
            for ind, node_out in enumerate(self.output_nodes):
                assert node_out in self.model.nodes, "node {} does not exist".format(node_out)
                args.append((node_out, use_global_default))

            pool = ThreadPool(n_threads)
            result = pool.starmap(self.calculate_output_sequence, args)
            pool.close()
            pool.join()

            for res in result:
                self.x_out.append(res[1])
                self.t_out.append(res[0])

        self.x_out = np.array(self.x_out)
        self.t_out = np.array(self.t_out)

    def add_const_output(self, bias):
        """
        Adds a constant signal to the output vector.

        :param bias: value of constant output signal
        """
        self.x_out = np.concatenate((self.x_out, np.atleast_2d(bias * np.ones(shape=self.input_t[0].shape))))
        self.t_out = np.concatenate((self.t_out, np.atleast_2d(self.input_t[0])))

    def calculate_output_sequence(self, node_name, use_global_default=False):
        """
        Calculates the output sequence at a given node.

        The output sequence is calculated according to the input sequence(s) added prior to executing this method to the
        testbench. Before executing make sure :meth:`self.evaluate_network()` was executed.

        :param node_name: Name of node for which the output is returned.
        :param use_global_default: set to true if global defaults should be used with SymNum's (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not none. Default: False
        :type use_global_default: Boolean
        :return: tuple containing time and signal vector at the given output node
        """
        assert node_name in self.model.nodes, "node does not exist"
        self._extract_min_max_signal_time()
        self._prepare_signals()

        nodes_to_output = self.model.nodes_to_output[node_name]  # amplitude, phase, delay, path

        t_out = self.input_t[0]  # all signals have a common time vector after resampling
        x_out = np.zeros(shape=t_out.shape, dtype=np.complex128)

        for path in tqdm(nodes_to_output, disable=self.disable_tqdm):
            end_index = path[3][1:-1].find('-') + 1
            input_signal_name = path[3][1:end_index]
            input_index = self.input_nodes.index(input_signal_name)
            delay = path[2] if not hasattr(path[2], 'eval') else path[2].eval(feed_dict=self.feed_dict,
                                                                              use_global_default=use_global_default)
            shift_steps = int(round(delay / self.timestep))
            x = self.input_x[input_index]
            attn = path[0] if not hasattr(path[0], 'eval') else path[0].eval(feed_dict=self.feed_dict,
                                                                             use_global_default=use_global_default)
            phase = path[1] if not hasattr(path[1], 'eval') else path[1].eval(feed_dict=self.feed_dict,
                                                                              use_global_default=use_global_default)
            if shift_steps <= len(x) and shift_steps > 0:
                x = np.hstack((np.zeros(shape=(shift_steps,)), x[0:-shift_steps]))
                x_out += attn * np.exp(1j * phase) * x
            elif shift_steps == 0:
                x_out += attn * np.exp(1j * phase) * x
        return t_out, x_out


if __name__ == "__main__":
    multiprocessing.freeze_support()
