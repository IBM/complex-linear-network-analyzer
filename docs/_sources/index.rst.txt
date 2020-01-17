..
    # +-----------------------------------------------------------------------------+
    # |  Copyright 2019-2020 IBM Corp. All Rights Reserved.                         |
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
    # |  Authors: Pascal Stark                                                      |
    # +-----------------------------------------------------------------------------+

Complex Linear Network Analyzer (COLNA)
=======================================

The complex linear network analyzer (COLNA) python package analytically computes the propagation of complex valued signals in networks with linear nodes and directed, complex valued and delayed edges (Fig. 1).

COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used in other areas, where signal propagation through linear complex networks is of relevance.

COLNA offers an easy and well documented interface, which allows to quickly build network models and understand their behaviour.

COLNA computes all paths leading from input to output nodes (including recurrent paths) down to a certain amplitude accuracy threshold.

COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties.

COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

The core functionality of COLNA is visualized in the figure below.

.. figure:: /figures/colna_features.png
    :align: center

    Visualization of the basic functions of COLNA. With COLNA one can quickly assemble a network and visulize it,
    evaluate how waves propagate from input to output nodes and with a testbench signals can be injected to the network
    and the corresponding output signals are computed.


Contents
========

To start working with COLNA we suggest to take a look at the user manual and some of the examples.

.. toctree::
   :maxdepth: 2

   manual.rst
   examples.rst
   colna.rst
   license.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
