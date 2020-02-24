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
# |  Authors: Lorenz K. Mueller, Pascal Stark                                   |
# +-----------------------------------------------------------------------------+

"""
The complex linear network analyzer (COLNA) python package analytically computes the propagation of complex valued signals in networks with linear nodes and directed, complex valued and delayed edges.
COLNA can be used to model linear, photonic circuits (e.g. Si photonic integrated circuits). It provides an analytic expression of the systems transfer matrix, which is often useful to better understand the effect of
system parameters on the output signal.

* COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used in other areas, where signal propagation through linear complex valued networks is of relevance.

* COLNA provides an analytic expression of the networks transfer matrix, which is often useful to better understand the effect of system parameters' variation on the output signal.

* COLNA offers an easy and well documented interface, which allows to quickly build network models and understand their behaviour.

* COLNA computes all paths leading from input to output nodes down to a certain amplitude accuracy threshold. It also supports the evaluation of recurrent paths (loops).

* COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties.

* COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

* COLNA is well suited for educational purposes, where analytic expression help to better understand the functionality of simple photonic networks, like for example a Mach-Zehnder interferometer.

"""