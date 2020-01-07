Complex Linear Network Analyzer (COLNA)
=======================================

The complex linear network analyzer package can be used to compute analytically the propagation of complex valued
signals in networks with linear nodes and directed, complex valued and delayed edges.

COLNA computes all paths leading from input to output nodes (including recurrent paths) down to a certain amplitude accuracy threshold.

COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties, returning either a numeric or

COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used
in other areas, where signal propagation through linear complex networks is of relevance.

Documentation
-------------
The full documentation can be found at ....., including a full reference, tutorials and examples.

Installation
------------
Pip install the COLNA package. All required packages are installed automatically.
If you intend to use the visualization feature, Graphviz must be installed and added to the path.

More details for the installation are given in the User Manual.



Requirements
------------
Numpy, Scipy, Matplotlib, tqdm

For visualization: Graphviz + Graphviz Python Package (see installation instructions for details)
