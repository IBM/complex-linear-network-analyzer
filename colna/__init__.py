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