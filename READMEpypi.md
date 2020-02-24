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

Documentation
-------------
Documentation including a full reference, tutorials and examples is available [here](https://ibm.github.io/complex-linear-network-analyzer/).

Installation
------------
Pip install the COLNA package. All required packages are installed automatically.

```
pip install complex-linear-network-analyzer
```

If you intend to use the visualization feature, Graphviz must be installed and added to the path and the COLNA package must be
installed as follows: 

```
pip install complex-linear-network-analyzer[Visualization]
```

Details for the installation are given in the [user manual](https://ibm.github.io/complex-linear-network-analyzer/).

Requirements
------------
Numpy, Scipy, Matplotlib, tqdm

For visualization: Graphviz + Graphviz Python Package (see installation instructions for details)
