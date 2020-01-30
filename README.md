Complex Linear Network Analyzer (COLNA)
=======================================

The complex linear network analyzer (COLNA) python package analytically computes the propagation of complex valued signals in networks with linear nodes and directed, complex valued and delayed edges (Fig. 1).

COLNA was developed to compute coherent wave propagation in linear photonic circuits but it can be used in other areas, where signal propagation through linear complex networks is of relevance.

COLNA offers an easy and well documented interface, which allows to quickly build network models and understand their behaviour.

COLNA computes all paths leading from input to output nodes (including recurrent paths) down to a certain amplitude accuracy threshold.

COLNA supports the mixed use of numeric and symbolic numbers (variable) for all edge properties.

COLNA can inject complex valued signals to the network and the compute the resulting signals at the output nodes using a testbench.

The core functionality of COLNA is visualized in the figure below.

![](docs/_images/colna_features_extended_plain.svg)

Documentation
-------------
Documentation including a full reference, tutorials and examples is available [here](https://ibm.github.io/complex-linear-network-analyzer/)

Installation
------------
Pip install the COLNA package. All required packages are installed automatically.
If you intend to use the visualization feature, Graphviz must be installed and added to the path.

More details for the installation are given in the [user manual](https://ibm.github.io/complex-linear-network-analyzer/).

Requirements
------------
Numpy, Scipy, Matplotlib, tqdm

For visualization: Graphviz + Graphviz Python Package (see installation instructions for details)

Contributing
------------

Contributions are very welcome! Please follow the the [contribution guidelines](CONTRIBUTING.md).
