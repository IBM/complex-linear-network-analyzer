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

Photonic Design Examples
===========================

Here we show how COLNA can be used to model simple photonic circuits. Did you use COLNA to model a photonic circuit? Let
us know and we will be very happy to add your example on this page.

The examples can be downloaded from COLNA's github `repository <https://github.com/IBM/complex-linear-network-analyzer>`_.

Mach-Zehnder Interferometer Weight
----------------------------------

In this example we model a Mach-Zehnder interferometer (MZI) used as an optical transmission weight:
By tuning the mode index in one of the MZI arms we set the transmission for a constant input signal.
In this example a heater is used to tune the index through the thermo-optic effect.
The figure below shows a sketch of the structure of our MZI interferometer.

.. figure:: /figures/mzi_example.svg
    :align: center

We use a :class:`.PhysicalNetwork` to model our circuit. As we use our MZI as a constant transmission weight
we are not interested in the time delay caused by the MZI, therefore all time delays are set to zero.

After setting the global parameters (wavelength, effective mode index in waveguide) we
start by modeling the y-splitter. Due to fabrication imperfections the input power will not be split exactly equally
between the two output ports of the splitter; the same is true for the phase relations. We create symbolic numbers for the
splitting efficiency and phases and create the corresponding :class:`.Device`. The combiner is modeled in a similar way.

.. literalinclude:: ../../designexamples/mziweight.py
    :language: python
    :lines: 29-44

The parameters of the two waveguide arms are defined:

.. literalinclude:: ../../designexamples/mziweight.py
    :language: python
    :lines: 60-66

Next we are ready to assemble the network.

.. literalinclude:: ../../designexamples/mziweight.py
    :language: python
    :lines: 68-78

You can visualize the network and as usual you can get the html description:

.. figure:: /figures/mziweight.svg
    :align: center

As we just use a constant input wave we can use the :meth:`.Network.get_reduced_output` method
