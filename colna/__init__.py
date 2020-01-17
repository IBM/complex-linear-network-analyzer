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

"""
ComplexLinearNetworkAnalyzer (COLNA) Package

The complex linear network analyzer package can be used to compute analytically the propagation of complex valued
signals in networks with linear nodes and directed, complex valued and delayed edges.

COLNA computes all paths leading from input to output nodes (including recurrent paths) down to a certain amplitude accuracy threshold.

COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties, returning either a numeric or

COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used
in other areas, where signal propagation through linear complex networks is of relevance.
"""

