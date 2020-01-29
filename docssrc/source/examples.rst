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
    # |  Authors: Lorenz Müller, Pascal Stark                                       |
    # +-----------------------------------------------------------------------------+

Examples
========

The following code examples are included in the examples/ directory of the COLNA project.

Feedforward Network
-------------------

.. literalinclude:: ../../examples/feedforward.py
    :language: python
    :lines: 1-
    :linenos:

.. figure:: /figures/feedforward.svg
    :align: center

Output of the :meth:`.Network.get_html_result`:

.. raw:: html
   :file: ./figures/feedforward.html

Feedforward Network with Testbench
------------------------------------

.. literalinclude:: ../../examples/feedforwardwithtestbench.py
    :language: python
    :lines: 1-
    :linenos:

.. figure:: /figures/feedforward_with_testbench_output.svg
    :align: center


Physical Network
------------------------------------

.. literalinclude:: ../../examples/physicalnetwork.py
    :language: python
    :lines: 1-
    :linenos:

.. figure:: /figures/PhysicalNetworkSimplified.svg
    :align: center

    Simplified visualization of physical network.

.. figure:: /figures/PhysicalNetworkFull.svg
    :align: center

    Full visualization of physical network.

Small Recurrent Network
------------------------------------

.. literalinclude:: ../../examples/recurrent.py
    :language: python
    :lines: 1-
    :linenos:

.. figure:: /figures/recurrent_net.svg
    :align: center


Recurrent Network with Testbench
------------------------------------

.. literalinclude:: ../../examples/recurrentwithtestbench.py
    :language: python
    :lines: 1-
    :linenos:

Symbolic Feedforward Network
-------------------------------------------

.. literalinclude:: ../../examples/symbolicfeedforward.py
    :language: python
    :lines: 1-
    :linenos:

.. figure:: /figures/symbolicfeedforward.svg
    :align: center


Symbolic Feedforward Network with Testbench
-------------------------------------------

.. literalinclude:: ../../examples/symbolicfeedforwardwithtestbench.py
    :language: python
    :lines: 1-
    :linenos:

Symbolic Recurrent Network
-------------------------------------------

.. literalinclude:: ../../examples/symbolicrecurrent.py
    :language: python
    :lines: 1-
    :linenos:

Large Symbolic Recurrent Network
-------------------------------------------

.. literalinclude:: ../../examples/largesymbolicrecurrentnetwork.py
    :language: python
    :lines: 1-
    :linenos: