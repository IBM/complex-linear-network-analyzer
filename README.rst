Complex Linear Network Analyzer (COLNA)
=======================================

The complex linear network analyzer package can be used to compute analytically the propagation of coherent, complex
signals in complex valued networks. The network consists of linear nodes and directed, complex valued and delayed edges.
Originally, COLNA was developed to compute coherent wave propagation in linear photonic circuits.

COLNA computes all paths leading to output nodes (including recurrent paths) down to a certain accuracy
threshold. Using a testbench, complex valued signals can be injected to the network and the output signal is computed.

COLNA supports the mixed use of constants and symbolic numbers (variable). It is for example possible to describe



Quickstart
----------

Install
#######

Pip install the COLNA package. All required packages are installed automatically.
If you intend to use the visualization feature, Graphviz must be installed manually.



Requirements
------------
Numpy, Scipy, Matplotlib, tqdm

For visualization: Graphviz + Graphviz Python Package (see installation instructions for details)