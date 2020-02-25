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

The complex linear network analyzer (COLNA) python package analytically computes the propagation of complex valued signals in networks with linear nodes and directed, complex valued and delayed edges.

* COLNA can be used to model linear, photonic circuits (e.g. Si photonic integrated circuits). It provides an analytic expression of the systems transfer matrix, which is often useful to better understand the effect of system parameters on the output signal.

* COLNA offers an easy and well documented interface, which allows to quickly build network models and understand their behaviour.

* COLNA computes all paths leading from input to output nodes down to a certain amplitude accuracy threshold. It also supports the evaluation of recurrent paths (loops).

* COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties.

* COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

* COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used in other areas, where signal propagation through linear complex valued networks is of relevance.

* COLNA is well suited for educational purposes, where analytic expression help to better understand the functionality of simple photonic networks, like for example a Mach-Zehnder interferometer.

The core functionality of COLNA is visualized in the figure below.

.. figure:: /figures/colna_features_extended_plain.svg
    :align: center

    Visualization of the basic functions of COLNA. With COLNA one can quickly assemble a network and visulize it,
    evaluate how waves propagate from input to output nodes. Using a testbench signals can be injected to nodes
    and the corresponding output signals are computed.


Contents
========

To start working with COLNA we suggest to take a look at the user manual, read through the short `JOSS paper <https://joss.theoj.org/papers/10.21105/joss.02073>`_  and play around with
some of the examples.

.. toctree::
   :maxdepth: 2

   manual.rst
   examples.rst
   photonicdesignexamples.rst
   colna.rst
   paperandcitation.rst
   license.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
