# +-----------------------------------------------------------------------------+
# |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                                       |
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

import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import os


class Edge(object):
    """
    Edges connect two nodes in a directed manner.

    Edges add a phase shift (:math:`\phi`), delay (:math:`d`) and
    attenuation (:math:`a`) to the signal. The input to output relation of an edge is given by:

    .. math::
        x_{out}(t) = a \cdot x_{in}(t-d) \cdot e^{j\phi}

    Edge properties can be constant or symbolic numbers (variables).
    """

    def __init__(self, start, end, phase=.4, attenuation=.8, delay=1.0):
        """
        :param start: name of start vertex connected by this edge
        :type start: str
        :param end: name of end vertex connected by this edge
        :type end: str
        :param phase: phase shift added by this element (stacks additively)
        :type phase: float
        :param attenuation: attenuation caused by element (stacks multiplicatively)
        :type attenuation: float
        :param delay: time delay added by this element (stacks additively)
        :type delay: float
        """
        self.start = start
        self.end = end
        self.phase = phase
        self.attenuation = attenuation
        self.delay = delay

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and self.phase == other.phase and self.attenuation == other.attenuation and self.delay == other.delay


class Network(object):
    """
    Networks consist of linear nodes and directed edges.

    Networks are represented by a directed graph. COLNA computes all paths leading from input nodes to the output nodes
    (including recurrent paths) until the attenuation of each path falls below a given threshold.

    .. note::

      If a network contains recurrent paths (loops), the user must ensure that there is no gain in the network (i.e.
      attenuation < 1), otherwise the amplitude at the output will never fall below the threshold.

    """

    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.edges = []
        self.nodes_to_output = {}

    def add_node(self, name):
        """
        Add a new node to the network

        :param name: the name of the node
        :type name: str

        :raises ValueError: If a node with the same name already exists.
        """

        if name in self.nodes:
            raise (ValueError("node of name " + name + " already exists"))

        self.nodes.append(name)

    def add_edge(self, edge):
        """
        Add a new edge to the network

        The start and end nodes of the edge must already exist in the network.

        :param edge: the edge object to add
        :type edge: :class:`.Edge`

        :raises ValueError: If the start or end node of the edge does not exist in the network.

        """
        if not edge.start in self.nodes:
            raise (ValueError("attempted to add edge from undefined node" + edge.start))
        if not edge.end in self.nodes:
            raise (ValueError("attempted to add edge to an undefined node" + edge.end))

        self.edges.append(edge)

    def add_input(self, name, amplitude=1.0, phase=0.0, delay=0.0):
        """
        Define an input point of the network.

        The evaluation assumes signals with the given amplitude, phase and delay are
        propagating through the network from the given node when computing the analytical waveforms at each node.

        :param name: name of the node that is to receive the input
        :type name: str
        :param amplitude: amplitude of the input
        :type amplitude: float
        :param phase: phase of the input (relative to other inputs)
        :type phase: float
        :param delay: delay of the input (relative to other inputs)
        :type delay: float

        :raises ValueError: If the node with the provided name does not exist in the network.
        """

        if not name in self.nodes:
            raise (ValueError("attempted to give input to inexistent node " + name))

        self.inputs.append((amplitude, phase, delay, name))

    def get_reduced_output(self, name, feed_dict=None, use_shared_default=False):
        """
        Returns the output waves at this node. Paths with identical delay are added together.

        Waves with same delay are combined into a single entry using linear superposition of phase and amplitude.
        Phase is reduced to cover only 2*pi range.

        :param name: name of the node to get result from
        :type name: str
        :param use_shared_default: set to true if shared defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: True
        :type use_shared_default: bool
        :param feed_dict: Feed dictionary for SymNum variables. Default: None
        :type feed_dict: dict

        :return: amplitude, phase, delay (all numpy arrays)

        """

        waves = [np.array(list([w.eval(feed_dict=feed_dict, use_shared_default=use_shared_default) if hasattr(w, 'eval') else w for w in inner])) for
                 inner in self.get_result_np(name)]
        delays = np.unique(waves[2])
        values = []

        for delay in delays:
            amp_temp = waves[0][np.where(waves[2]==delay)]
            phase_temp = waves[1][np.where(waves[2]==delay)]
            values.append(np.sum(amp_temp*np.exp(1j*phase_temp)))
        values = np.array(values)
        return np.abs(values), np.angle(values), delays

    def get_eval_result(self, name, feed_dict=None, use_shared_default=False):
        """
        Returns a list of waves mixing at this node and evaluates symbolic numbers.

        :param name: name of the node to get result from
        :type name: str
        :param use_shared_default: set to true if shared defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: False
        :type use_shared_default: bool
        :param feed_dict: Feed dictionary for SymNum variables. Default: None
        :type feed_dict: dict

        :return: a list of waves mixing at this node, given as a tuple with entries (amplitude, phase, delay)
        """
        return [tuple([w.eval(feed_dict=feed_dict, use_shared_default=use_shared_default) if hasattr(w, 'eval') else w for w in inner]) for
                 inner in self.get_result(name)]

    def get_result(self, name):
        """
        Returns a list of waves mixing at this node

        :param name: name of the node to get result from
        :type name: str

        :raises ValueError: If the node with the provided name does not exist in the network.

        :return: a list of waves mixing at this node, given as a tuple with entries (amplitude, phase, delay)
        """
        if not name in self.nodes:
            raise (ValueError("attempted to retrive wave at non-existing node " + name))

        return [entry[0:3] for entry in self.nodes_to_output[name]]

    def get_result_np(self, name):
        """
        Returns a result at a given node as numpy array

        :param name: name of the node to get result from
        :type name: str

        :raises ValueError: If the node with the provided name does not exist in the network.

        :return: x; x[0]: amp, x[1]: phase, x[2]: delay
        """
        if not name in self.nodes:
            raise (ValueError("attempted to retrive wave at non-existing node " + name))

        amp = np.asarray([entry[0] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        phase = np.asarray([entry[1] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        delay = np.asarray([entry[2] for entry in self.nodes_to_output[name]]).reshape([1, -1])
        return np.concatenate([amp, phase, delay], 0)

    def get_paths(self, name):
        """
        Find all paths leading to a node.

        :param name: name of the node to get result from
        :type name: str
        :raises ValueError: If the node with the provided name does not exist in the network.
        :return: all paths leading to a node
        """
        if not name in self.nodes:
            raise (ValueError("attempted to retrive path to non-existing node " + name))

        return [entry[3] for entry in self.nodes_to_output[name]]

    def print_stats(self):
        """
        Prints some statistics of the evaluated network

        Currently this prints the number of evaluated path. In future implementations the statistics method might be
        enhanced.
        """
        n_paths = sum([len(val) for val in self.nodes_to_output.values()])
        print('total number of paths:', n_paths)

    @staticmethod
    def stopping_criterion(amplitude, cutoff):
        """
        Stopping criterion

        Used together with network evaluation. If the stopping criterion is fulfilled, the analysis of the current path
        is stopped.

        :param amplitude: current amplitude
        :type amplitude: float
        :param cutoff: threshold for cutoff criterion
        :type cutoff: float
        :return: True if amplitude is less than cutoff, otherwise False.
        """
        return amplitude < cutoff

    def evaluate(self, amplitude_cutoff=0.01, max_endpoints=100000, use_shared_default=True, feed_dict=None,
                 hide_tqdm_progress=False):
        """
        Evaluate the network.

        The network evaluation method works by walking through the graph from all input nodes, along all possible paths.
        The evaluation of each path is stopped as soon as the total path amplitude falls below the amplitude_cutoff limit.

        :param amplitude_cutoff: amplitude below which a wave is not further propagated through the network
        :type amplitude_cutoff: float
        :param max_endpoints: evaluation is interrupted early, if more than max_endpoints exist in evaluation
        :type max_endpoints: int
        :param use_shared_default: set to true if shared defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: True
        :type use_shared_default: bool
        :param feed_dict: Feed dictionary for SymNum variables. Default: None
        :type feed_dict: dict

        :return:
            updates self.nodes_to_output
            a dictionary whose keys are node names. For each node name a list of quadruplets is given
            [(amplitude, phase, delay, path), (amplitude, phase, delay, path), ...].

            .. note::
                Phases are simply added together and not reset to a finite range.

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
                                                        use_shared_default=use_shared_default) if hasattr(current_attn,
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

    def visualize(self, show_edge_labels=True, path='network', skip_colon=False, format='pdf'):
        """
        Visualize the network

        :param show_edge_labels: if True, edge labels showing the amplitude, phase and delay of the edge are drawn.
        :type show_edge_labels: bool
        :param path: output path for file. If the path does not exist it will be created automatically.
        :type path: str
        :param skip_colon: Skip nodes which contain ':' in their name. This is used for PhysicalNetwork visualization.
        :type skip_colon: bool
        :param format: output format (supports all format options of Graphviz), e.g. 'pdf', 'svg'
        :type format: str
        :return: Writes a dot file at the given path and renders it to the desired output format using graphviz.
        :return: Returns the path to the file (can be relative).
        """
        try:
            from graphviz import Digraph
        except ModuleNotFoundError as err:
            warnings.warn("Graphviz Package was not found, visualization is skipped.")
            return 0
        s = Digraph('structs', graph_attr={'ranksep': '0.5', 'overlap': 'false', 'splines': 'true', 'rankdir': 'TB',
                                           'constraint': 'true', 'nodesep': '2'}, node_attr={'shape': 'record'},
                    edge_attr={}, engine='dot')

        for node in self.nodes:
            if not (skip_colon and ':' in node):
                s.node(node, node)

        for edge in self.edges:
            if show_edge_labels:
                s.edge(edge.start.replace(":", ""), edge.end.replace(":", ""),
                       label='a{}, p{}, d{}'.format(edge.attenuation, edge.phase, edge.delay))
            else:
                s.edge(edge.start.replace(":", ""), edge.end.replace(":", ""))

        head, tail = os.path.split(path)
        if head != '':
            Path(head).mkdir(parents=True, exist_ok=True)
        return s.render(path, view=False, format=format)

    def get_html_result(self, name, time_symbol='t', evaluate=False, feed_dict=None, use_shared_default=False,
                        linebreak_limit=1, precision=0, path='out.html'):
        """
        Creates a html file with a rendered math equation describing all waves arriving at the given node.

        .. warning:: To correctly render the equations in the browser, MathJax is required. The script is loaded automatically when you open the html file in a browser, if an internet connection is available.

        :param name: Name of the node to get result from. If it is a list, results will be retrieved for all nodes in the list and compiled in a single html file.
        :type name: str or list
        :param time_symbol: character used to describe time/delays in the equation
        :type time_symbol: str
        :param evaluate: If evaluate is True, SymNum's will be evaluated using the feed_dict and use_shared_default values specified. Otherwise SymNums are represented by their name as variables.
        :type evaluate: bool
        :param feed_dict: a dictionary specifying values of variables by name. If only some variables are specified, for all other variables the default value will be used.
        :type feed_dict: dict
        :param use_shared_default: set to true if shared defaults should be used with SymNums (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not None. Default: False
        :type use_shared_default: bool
        :param linebreak_limit: A line break will be added roughly every linebreak_limit chars in the latex string. Set to 1 for a linebreak after each term. Set to 0 to get a latex string on a single line. Default: 1
        :type linebreak_limit: int
        :param path: Output path where html file containing the MathJax code is stored.  If the path does not exist it will be created automatically.
        :type path: str
        :param precision: Number of significant digits to be output. Set to 0 to use the default value of str() method.
        :type precision: int

        :raises ValueError: If the node with the provided name does not exist in the network.
        :raises IOError: If the output file can not be created or accessed.

        :return: writes a html file at the given path

        """

        template = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width">
          <title>{}</title>
          <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
          <script id="MathJax-script" async
                  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
          </script>
        </head>
        <body>
        {}
        </body>
        </html>
        """

        if isinstance(name, list) == False:
            name = [name]

        raw_string = ''
        for node in name:
            if not node in self.nodes:
                raise (ValueError("attempted to retrive wave at non-existing node " + name))

            raw_string += '<p> Waves at node ' + node + '<br><br> \(' \
                          + self.get_latex_result(name=node,
                                                  time_symbol=time_symbol,
                                                  evaluate=evaluate,
                                                  feed_dict=feed_dict,
                                                  use_shared_default=use_shared_default,
                                                  linebreak_limit=linebreak_limit,
                                                  precision=precision) + '\)</p>'

        output_html = template.format('waves at nodes' + str(name), raw_string)

        head, tail = os.path.split(path)
        if head != '':
            Path(head).mkdir(parents=True, exist_ok=True)

        try:
            with open(path, 'w') as file:
                file.write(output_html)
        except IOError as e:
            return e



    def get_latex_result(self, name, time_symbol='t', evaluate=False, feed_dict=None, use_shared_default=False,
                         linebreak_limit=0, precision=0):
        """
        Returns a latex string that describes all waves arriving at the given node.

        SymNums are shown as variables, unless evaluate is set to True.

        :param name: Name of the node to get result from
        :type name: str
        :param time_symbol: character used to describe time/delays in the equation
        :type time_symbol: str
        :param evaluate: If evaluate is True, SymNum's will be evaluated using the feed_dict and use_shared_default values specified. Otherwise SymNums are represented by their name as variables.
        :type evaluate: bool
        :param feed_dict: a dictionary specifying values of variables by name. If only some variables are specified, for all other variables the default value will be used.
        :type feed_dict: dict
        :param use_shared_default: set to true if shared defaults should be used with SymNums (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not None. Default: False
        :type use_shared_default: bool
        :param linebreak_limit: A line break will be added roughly every linebreak_limit chars in the latex string. Set to 1 for a linebreak after each term. Set to 0 to get a latex string on a single line. Default: 1
        :type linebreak_limit: int
        :param precision: Number of significant digits to be output. Set to 0 to use the default value of str() method.
        :type precision: int

        :raises ValueError: If the node with the provided name does not exist in the network.

        :return: a list of waves mixing at this node, given as a tuple with entries (amplitude, phase, delay)
        """
        if not name in self.nodes:
            raise (ValueError("attempted to retrive wave at non-existing node " + name))

        latex_string = r''
        last_linebreak = 0
        align_next_line = False
        for value in self.nodes_to_output[name]:
            for i, elem in enumerate(value[0:3]):
                elem_type = i % 4
                # get the string representation of the value
                if type(elem) == SymNum:
                    if evaluate == True:
                        temp_eval_val = elem.eval(feed_dict=feed_dict, use_shared_default=use_shared_default)
                        str_elem_value = '%.*g' % (precision, temp_eval_val) if precision != 0 else str(temp_eval_val)
                    else:
                        str_elem_value = elem.to_latex(precision=precision)
                else:
                    temp_val = elem
                    str_elem_value = '%.*g' % (precision, temp_val) if precision != 0 else str(temp_val)

                # stich together the latex string depending on the type of the element (amplitude, phase, delay)
                if elem_type == 0:  # amplitude
                    latex_string += '+' + str_elem_value + '\cdot' if align_next_line == False else '+&' + str_elem_value + '\cdot'
                    align_next_line = False
                elif elem_type == 1:  # phase
                    latex_string += '\exp(j (' + str_elem_value + '))\cdot '
                elif elem_type == 2:  # delay
                    in_node_name = value[3].split('-')[1]
                    in_node_name = in_node_name.replace(':','\_')
                    latex_string += in_node_name + '_{in}(' + time_symbol + '-' + str_elem_value + ')'
                    # Linebreak
                    if len(latex_string) - last_linebreak > linebreak_limit and linebreak_limit > 0:
                        last_linebreak = len(latex_string)
                        latex_string += r'\\'
                        align_next_line = True

        latex_string = latex_string[1:]  # removes the leading +
        if linebreak_limit > 0:
            latex_string = r'\begin{equation}\begin{split}&' + latex_string + r'\end{split}\end{equation}'

        return latex_string


class SymNum:
    """
    Symbolic number class.

    Symbolic numbers can be used for all edge properties in analytic networks.

    """

    def __init__(self, name, default=0.9, product=True, numerical=None):
        """
        :param name: the name of the variable. The name should be unique for each SymNum present in the network.
        :type name: str
        :param default: the default value substituted, when we evaluate this variable. Default = 0.9
        :type default: float
        :param product: whether this variable is composed as a product (True) or a sum (False). Product is used to distinguish attenuations (stacking multiplicatively) and phases / delays (stacking additively). Default: True
        :type product: bool
        :param numerical: initial value of numerical part (numerical factor for product variables, numerical addition for additive variables). Can be set to none for automatic initialization (1.0 for product variables, 0.0 for additive variables). Default: None
        :type numerical: float or None

        """
        # the numerical part of the number's value
        self.numerical = numerical if numerical is not None else 1.0 * product

        # the symbolic part of the number's value
        self.symbolic = {name: 1}

        # dict of the values that are inserted for each variable if no particular value is specified at evaluation
        self.defaults = {name: default}

        # indicates whether the number stacks multiplicatively or additively
        self.product = product

        # when the network is evaluated we use this value to determine when to cut the path due to attenuation limit
        # for SymNums that result from addition/multiplication of other SymNums (a and b) this is
        # max(a.shared_default, b.shared_default)
        self.shared_default = default

    def __eq__(self, other):
        return self.symbolic == other.symbolic and self.numerical == other.numerical and self.defaults == other.defaults and self.product == other.product and self.shared_default == other.shared_default

    def __copy__(self):
        """ Creates a copy of a Symbolic Number."""
        copy = SymNum('', product=self.product, numerical=self.numerical)
        copy.shared_default = self.shared_default
        copy.symbolic = deepcopy(self.symbolic)
        copy.defaults = deepcopy(self.defaults)
        return copy

    def __add__(self, other):
        """ Adds a Symbolic number.

        :param other: Number to be added to this SymNum
        :return: New symbolic number which contains addition of this and other SymNum.
        :rtype: class:`SymNum`
        """
        if self.product == True:
            raise (ValueError("do not add product variables"))

        copy = self.__copy__()
        if isinstance(other, SymNum):
            if not self.product == other.product:
                raise (ValueError("ensure that product type of symbolic numbers are the same"))
            copy.numerical += other.numerical
            copy.shared_default = max(self.shared_default, other.shared_default)
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
        """
        Multiplies a symbolic numbers.

        :param other: Symbolic number to be multiplied to this SymNum
        :return: New symbolic number which contains the result of the multiplication of this and other SymNum.
        :rtype: class:`SymNum`
        """
        if self.product == False:
            raise (ValueError("do not multiply non-product variables"))

        copy = self.__copy__()
        if isinstance(other, SymNum):
            if not self.product == other.product:
                raise (ValueError("ensure that product type of symbolic numbers are the same"))
            copy.numerical *= other.numerical
            copy.shared_default = max(self.shared_default, other.shared_default)
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
        """
        Divides symbolic number.

        :param other: The value of this SymNum is divided by the other SymNum.
        :return: New symbolic number which contains the result of the division of this and other SymNum.
        :rtype: class:`SymNum`
        """
        if self.product == False:
            raise (ValueError("do not divide non-product variables"))
        copy = self.__copy__()
        if isinstance(other, SymNum):
            if not self.product == other.product:
                raise (ValueError("ensure that product type of symbolic numbers are the same"))
            copy.numerical /= other.numerical
            copy.shared_default = max(self.shared_default, other.shared_default)
            for symbol in other.symbolic.keys():
                if not (symbol in copy.symbolic.keys()):
                    copy.symbolic[symbol] = -1
                    copy.defaults[symbol] = other.defaults[symbol]
                else:
                    copy.symbolic[symbol] -= other.symbolic[symbol]
        else:
            copy.numerical /= other
        return copy

    def eval(self, feed_dict=None, verbose=False, use_shared_default=True):
        """
        Evaluate the numerical value of the number

        :param feed_dict: a dictionary specifying values of variables by name. If only some variables are specified, for all other variables the default value will be used.
        :type feed_dict: dict
        :param verbose: print the number in symbolic form before returning
        :type verbose: bool
        :param use_shared_default: set to true if shared defaults should be used with SymNums (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not None. Default: True
        :type use_shared_default: bool
        :return: numerical value of the number (float)
        """
        symbolic_evaluated = 1 if self.product else 0
        if feed_dict is None and use_shared_default:
            # this option leads to the fastest evaluation of the network. It assumes all symbols have the same shared default value.
            total_count = sum(self.symbolic.values())
            if self.product:
                symbolic_evaluated = self.shared_default ** total_count
            else:
                symbolic_evaluated = self.shared_default * total_count
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

    def to_latex(self, precision=0):
        op1 = " " if self.product else " + "
        op2 = "^" if self.product else " \cdot "
        out = ""
        for key in self.symbolic.keys():
            out += op1
            out += key + op2 + str(self.symbolic[key])

        if precision == 0:
            numerical_str = str(self.numerical)
        else:
            numerical_str = '%.*g' % (precision, self.numerical)
        return numerical_str + out

    def __repr__(self):
        """
        convenience overload to print lists of these numbers easily.
        """
        return '<SymNum{' + str(self) + '}>'


class Device(Network):
    """
    Defines a device object for physical networks.

    Device is a child class of network. It provides convenience methods to create the device from it's complex
    scattering matrix (matrix describing input-output relation). Nodes are renamed automatically, based on the device type,
    device name and port/node number ('devicetype:devicename:name').

    """

    def __init__(self, name, devicetype='device', scattering_matrix=None, delay=None):
        """
        :param scattering_matrix: device scattering matrix, if both :attr:`scattering_matrix` and :attr:`delay` are provided, the device will be initalized with the corresponding scattering matrix and delay
        :type scattering_matrix: numpy.array
        :param delay: device delay, if both :attr:`scattering_matrix` and :attr:`delay` are provided, the device will be initalized with the corresponding scattering matrix and delay
        :type delay: float
        :param name: name of the device
        :type name: str
        :param devicetype: typename of the device
        :type devicetype: str
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

        :param nodename: name of the node to add to the outputs
        :type nodename: str
        """
        name = self.devicetype + ":" + self.name + ":" + nodename
        if not name in self.nodes:
            raise (ValueError("attempted to add output to inexistent node " + str(nodename)))

        self.outputs.append(name)

    def add_input(self, nodename, amplitude=1.0, phase=0.0, delay=0.0):
        """
        Define an input point of the PhysicalNetwork.

        The evaluation assumes signals with the given amplitude, phase and delay are
        propagating through the network from the given node when computing the analytical waveforms at each node.

        :param name: name of the node that is to receive the input
        :type name: str
        :param amplitude: amplitude of the input
        :type amplitude: float
        :param phase: phase of the input (relative to other inputs)
        :type phase: float
        :param delay: delay of the input (relative to other inputs)
        :type delay: float

        :raises ValueError: If the node with the provided name does not exist in the network.
        """

        name = self.devicetype + ":" + self.name + ":" + nodename
        super().add_input(name, amplitude, phase, delay)

    def add_node(self, nodename):
        """
        Add a new node to the network

        :param name: the name of the node
        :type name: str

        :raises ValueError : If a node with the same name already exists.
        :raises ValueError : If a nodename contains a colon (':').

        """
        if ":" in nodename:
            raise (ValueError("Use of : in nodenames is forbidden for Devices."))
        name = self.devicetype + ":" + self.name + ":" + nodename
        super().add_node(name)

    def add_edge(self, edge):
        """
        Add a new edge to the network

        The start and end nodes of the edge must already exist in the network.

        :param edge: the edge object to add
        :type edge: :class:`.Edge`

        :raises ValueError: If the start or end node of the edge does not exist in the network.
        """
        edge.start = self.devicetype + ":" + self.name + ":" + edge.start
        edge.end = self.devicetype + ":" + self.name + ":" + edge.end
        super().add_edge(edge)

    def init_from_scattering_matrix(self, smatrix, delay=1.0):
        """
        Defines a device by its complex scattering matrix and a delay

        For a device with n inputs and m outputs pass a nxm complex matrix specifying attenuation and phase-shift from
        input node i to output node j at matrix entry (i, j). Creates n+m nodes linked by n*m edges with the given parameters,
        all with time delay 'delay'.
        input node number k will be named 'i+str(k)'
        output node number k will be named 'o+str(k)'

        .. warning::
           This method can not be used with SymNums. Use :meth:`~.Device.initialize_from_phase_and_attenuation_matrix` or define the
           nodes and edges manually instead.

        :param smatrix: complex scattering matrix defining the input to output mapping
        :type smatrix: numpy.array
        :param delay: time delay from input to output
        :type delay: float
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

    def init_from_phase_and_attenuation_matrix(self, attenuationmatrix, phasematrix, delay):
        """
        Defines a device by its attenuation and phase matrix and a delay.

        For a device with n inputs and m outputs pass a nxm matrix specifying attenuation and an nxm matrix specifiying
        phase-shift from input node i to output node j at matrix entry (i, j). Creates n+m nodes linked by n*m edges with
        the given parameters, all with time delay 'delay'.

        input node number k will be named 'i+str(k)'
        output node number k will be named 'o+str(k)'

        Use this method if you want to use symbolic numbers to define the phase and or amplitudes.

        :param attenuationmatrix: matrix defining attenuation between input to output node
        :type attenuationmatrix: numpy.array
        :param phasematrix: matrix defining phase between input to output node
        :type phasematrix: numpy.array
        :param delay: time delay from input to output
        :type delay: float
        """

        attenuationmatrix = np.atleast_2d(attenuationmatrix)
        n_in = attenuationmatrix.shape[0]
        n_out = attenuationmatrix.shape[1]

        for i in range(n_in):
            in_name = "i" + str(i)
            self.add_node(in_name)
            for j in range(n_out):
                out_name = "o" + str(j)
                name = self.devicetype + ":" + self.name + ":" + out_name
                if name not in self.nodes:
                    self.add_node(out_name)
                edge = Edge(in_name, out_name,
                            phase=phasematrix[i, j],
                            attenuation=attenuationmatrix[i, j],
                            delay=delay)
                self.add_edge(edge)

class DeviceLink(Edge):
    """
    Defines a device link object for physical networks.

    Device links are special edges that link devices. They are given the name of source and target device as well as
    source and target node within the device. Otherwise they function like the parent class Edge.

    """

    def __init__(self, startdevice, enddevice, startnode, endnode, startdevicetype='device', enddevicetype='device',
                 phase=.4, attenuation=.8, delay=1.0):
        """
        :param startdevice: name of the start device
        :type startdevice: str
        :param enddevice: name of the end device
        :type enddevice: str
        :param startnode: name of the node within the start device
        :type startnode: str
        :param endnode: name of the node within the end device
        :type endnode: str
        :param startdevicetype: name of the device type of startdevice (optional, defaults to 'device')
        :type startdevicetype: str
        :param enddevicetype: name of the device type of enddevice (optional, defaults to 'device')
        :type enddevicetype: str
        :param phase: phase shift of the device link
        :type phase: float
        :param attenuation: attenuation of the device link
        :type attenuation: float
        :param delay: time delay of the device link
        :type delay: float
        """
        for string in [startdevice, enddevice, startnode, endnode, startdevicetype, enddevicetype]:
            if not isinstance(string, str):
                raise (TypeError("device links require string inputs on some arguments"))

        super().__init__(startnode, endnode, phase, attenuation, delay)
        self.start = startdevicetype + ":" + startdevice + ":" + startnode
        self.end = enddevicetype + ":" + enddevice + ":" + endnode


class PhysicalNetwork(Network):
    """
    Defines a physical network.

    Extension of the Network class that allows for a more natural implementation of physical networks using
    a description as a collection of devices, device links, input and output sites.
    """

    def __init__(self):
        super().__init__()
        self.outputs = []

    def add_device(self, device):
        """
        Adds a device to the network

        Adds all nodes, edges, inputs and outputs of the device to the network.

        :param device: device to be added to the network
        :type device: :class:`.Device`
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
        Adds a device link to the network

        :param devicelink: the device link to be added
        :type devicelink: :class:`.DeviceLink`
        :raises ValueError: If either the start or end nodes of the devicelink do not exist.
        """

        if not devicelink.start in self.nodes:
            raise (ValueError("attempted to add device link from inexistent node " + devicelink.start))
        if not devicelink.end in self.nodes:
            raise (ValueError("attempted to add device link to inexistent node " + devicelink.end))
        self.add_edge(devicelink)

    def get_outputs(self):
        """
        Get the computed wave forms at all output sites

        :returns: a list of the output waves at all output nodes
        """
        return [self.get_result(name) for name in self.outputs]

    def visualize(self, show_edge_labels=True, path='network.gv', format='pdf', full_graph=False):
        """
        Visualizes the network

        :param show_edge_labels: if True, edge labels showing the amplitude, phase and delay of the edge are drawn.
        :type show_edge_labels: bool
        :param path: output path for file
        :type path: str
        :param format: output format (supports all format options of Graphviz), e.g. 'pdf', 'svg'
        :type format: str
        :return: Writes a dot file at the given path and renders it to the desired output format using graphviz.
        :return: Returns the path to the file (can be relative).
        """
        if full_graph:
            return super().visualize(show_edge_labels, path, True, format=format)
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

            return s.render(path, format=format, view=False)


class Testbench():
    """
    Implements a Testbench.

    Calculates the resulting output sequence for a given input signal sequence.
    """

    def __init__(self, network: Network, timestep=1, disable_progress_bars=True, feed_dict=None) -> object:
        """
        :param timestep: timestep on which the network output signal is evaluated. All delays in the network and of the input signal period should be an integer multiple of the timestep.
        :type timestep: float
        :param network: network where signals should be applied
        :type network: :class:`.Network`
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
        Clears the input and output lists.

        The input and output lists store the input nodes together with the corresponding input signals and the output node names.
        """
        self.inputs = []
        self.input_x = []
        self.input_t = []
        self.input_nodes = []
        self.output_nodes = []
        self.model.inputs = []

    def set_feed_dict(self, feed_dict):
        """
        Sets the feed dict.

        :param feed_dict: a dictionary specifying values of variables by name. If only some variables are specified, for all other variables the default value will be used.
        :type feed_dict: dict

        """
        self.feed_dict = feed_dict

    def add_output_node(self, node_name):
        """
        Add an output node by name.

        Output signals are calculated at each output node.

        :param name: the name of the node
        :type name: str
        """
        self.output_nodes.append(node_name)

    def add_input_sequence(self, node_name, x, t):
        """
        Add input signal sequence to a node

        :param name: the name of the node that receives the signal
        :type name: str
        :param x: input signal (complex), dimension: 1xN
        :type x: numpy.array
        :param t: time of signal, if none we assume that the signal vector provides the signal at each time step. The value x[n] is applied to the input Node during the right-open time interval [t[n], t[n+1]), dimension: 1x(N+1). Time values must be in increasing order.
        :type t: numpy.array

        :raises ValueError: If the dimensions of signal and time vector do not match
        :raises ValueError: If the node does not exist in the network.
        :raises OverflowError: If more than one input sequence is added to a single node.
        """
        if len(x) + 1 != len(t):
            raise ValueError("Size of time vector does not match size of signal vector.")

        if not node_name in self.model.nodes:
            raise ValueError("attempted to give input to inexistent node " + node_name)

        if node_name in self.input_nodes:
            raise OverflowError("At most one input sequence can be added per node. You added two at " + node_name)

        if node_name in self.output_nodes:
            raise ValueError(
                "A node must be an input or output sequence node, it can not be both. You added both for node " + node_name)

        self.model.add_input(node_name)
        self.input_nodes.append(node_name)
        self.input_x.append(x)
        self.input_t.append(t)

    def _extract_min_max_signal_time(self):
        """
        Extracts the start and stop time of all signals.

        Stores the minimum (self.t0) and maximum (self.t1) time.
        """
        self.t0 = min([time[0] for time in self.input_t])
        self.t1 = max([time[-1] for time in self.input_t])

    def _prepare_signals(self):
        """
        Prepares the input signals.

        Converts all signals to cover the time interval [self.t0, self.t1].  Signals are set to zero whenever they were not defined.
        Use `extract_min_max_signal_time` to extract t0 and t1 automatically from the provided input signals.
        """
        for i, x in enumerate(tqdm(self.input_x, disable=self.disable_tqdm)):
            t = self.input_t[i]
            t_s, x_s = self._convert_signal_to_timestep(x=x, t=t, timestep=self.timestep)
            self.input_t[i] = t_s
            self.input_x[i] = x_s

    def _convert_signal_to_timestep(self, x, t, timestep):
        """
        Resamples the input signals.

        Sets signal to 0 if the signal is not defined at that time.

        :param x: Signal vector
        :type: numpy.array
        :param t: Time vector, sorted in increasing order
        :type: numpy.array
        :param timestep: Defines the sampling rate.
        :return: t_sampled, x_sampled: resampled time and signal vector
        """
        t_sampled = np.linspace(start=self.t0, stop=self.t1, num=int(1 + round((self.t1 - self.t0) / timestep)))
        x_sampled = self._interpolate_constant(x=t_sampled, xp=t, yp=x)
        return t_sampled, x_sampled

    @staticmethod
    def _interpolate_constant(x, xp, yp):
        """
        Interpolation method that interpolates signals to the nearest left neighbour of the sampling point.

        This sampling is used, as input signal y[n] is defined to be applied during the right open time interval
        [t[n], t[n+1]).

        :param x: x coordinates where signal should be interpolated
        :type x: numpy.array
        :param xp: x coordinate of signal to be sampled, assuming array is sorted in increasing order (typically time vector)
        :type xp: numpy.array
        :param yp: y coordinate of signal to be sampled
        :type yp: numpy.array
        :return: interpolated signal, signal is set to zero when it was not defined in that time range.
        """
        # Interpolate only in the range where xp (typically time vector) is defined,
        # for sampling values outside of this range create zero values.
        x_l = np.searchsorted(x, xp[0])
        x_r = np.searchsorted(x, xp[-1])
        indices = np.searchsorted(xp, x[x_l:x_r], side='right')
        y = np.concatenate(([0], yp))
        # create zero entries wherever the signal is not specified
        z_l = np.zeros(shape=(x_l,), dtype=np.complex128)
        z_r = np.zeros(shape=(len(x) - x_r,), dtype=np.complex128)
        return np.concatenate((z_l, y[indices], z_r))

    def evaluate_network(self, amplitude_cutoff=1e-3, max_endpoints=1e6, use_shared_default=True):
        """
        Evaluate the network.

        Uses the :meth:`.Network.evaluate` method, self.feed_dict is used as a feed dictionary.

        :param amplitude_cutoff: amplitude below which a wave is not further propagated through the network
        :type amplitude_cutoff: float
        :param max_endpoints: evaluation is interrupted early, if more than max_endpoints exist in evaluation
        :type max_endpoints: int
        :param use_shared_default: set to true if shared defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: True
        :type use_shared_default: bool
        """
        self.model.evaluate(amplitude_cutoff, max_endpoints, use_shared_default=use_shared_default,
                            feed_dict=self.feed_dict)

    def calculate_output(self, use_shared_default=False, n_threads=0):
        """
        Calculates the output signals for the given input signals.

        self.feed_dict is used as the feed dictionary.

        :param use_shared_default: set to true if shared defaults should be used with SymNum's (higher speed),
         set to false if the default value of each SymNum should be used instead (higher accuracy). Default: False
        :type use_shared_default: bool
        :param n_threads: Number of threads that are used for evaluation (set to 0 to disable multithreading)
        :type n_threads: int

        :raises ValueError: If the output node does not exist.

        """

        self.x_out = []
        self.t_out = []

        if n_threads == 0:
            for ind, node_out in enumerate(self.output_nodes):
                if not node_out in self.model.nodes:
                    raise (ValueError("node {} does not exist".format(node_out)))

                t, x = self.calculate_output_sequence(node_name=node_out, use_shared_default=use_shared_default)
                self.x_out.append(x)
                self.t_out.append(t)

        else:
            args = []
            for ind, node_out in enumerate(self.output_nodes):
                if not node_out in self.model.nodes:
                    raise (ValueError("node {} does not exist".format(node_out)))

                args.append((node_out, use_shared_default))
            # Creates a thread per output node
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
        Adds a constant bias signal to the output vector.

        :param bias: value of constant output bias signal
        :type bias: float
        """
        self.x_out = np.concatenate((self.x_out, np.atleast_2d(bias * np.ones(shape=self.input_t[0].shape))))
        self.t_out = np.concatenate((self.t_out, np.atleast_2d(self.input_t[0])))

    def calculate_output_sequence(self, node_name, use_shared_default=False):
        """
        Calculates the output sequence at a given node.

        The output sequence is calculated for the input sequence(s) added prior to executing this method to the
        testbench. self.feed_dict is used as the feed dictionary.

        .. note::
            Before executing make sure :meth:`~.Testbench.evaluate_network()` was executed.

        :param node_name: Name of node for which the output is returned.
        :type node_name: str
        :param use_shared_default: set to true if global defaults should be used with SymNum's (higher speed) when no \
        feed_dict is provided, set to false if the default value of each SymNum should be used instead (higher accuracy). \
        The value is ignored if feed_dict is not none. Default: False
        :type use_shared_default: bool
        :return: tuple containing time and signal vector at the given output node
        :rtype: tuple of numpy.array
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
                                                                              use_shared_default=use_shared_default)
            shift_steps = int(round(delay / self.timestep))
            x = self.input_x[input_index]
            attn = path[0] if not hasattr(path[0], 'eval') else path[0].eval(feed_dict=self.feed_dict,
                                                                             use_shared_default=use_shared_default)
            phase = path[1] if not hasattr(path[1], 'eval') else path[1].eval(feed_dict=self.feed_dict,
                                                                              use_shared_default=use_shared_default)
            if shift_steps <= len(x) and shift_steps > 0:
                x = np.hstack((np.zeros(shape=(shift_steps,)), x[0:-shift_steps]))
                x_out += attn * np.exp(1j * phase) * x
            elif shift_steps == 0:
                x_out += attn * np.exp(1j * phase) * x
        return t_out, x_out


if __name__ == "__main__":
    multiprocessing.freeze_support()
