---
title: 'COLNA: A Python package to analyze complex, linear networks'
tags:
  - Python
  - Photonic networks
  - Integrated Photonics
  - Complex networks
  - Simulation
  - Circuit Simulation
  - Photonic Circuits
  
authors:
  - name: Pascal Stark
    affiliation: 1
  - name: Lorenz K. Müller
    affiliation: 1

affiliations:
 - name: IBM Research – Zürich, Säumerstrasse 4, 8803 Rüschlikon, Switzerland
   index: 1
date: 15 January 2020
bibliography: paper.bib
---

## Summary

The complex linear network analyzer (COLNA) python package analytically computes the propagation of complex valued 
signals in networks with linear nodes and directed, complex valued and delayed edges (Fig. 1).  COLNA offers an easy and 
well documented interface, which allows to quickly build network models and understand their behaviour. Its main purpose 
is to compute coherent wave propagation through linear photonic circuits, but COLNA might be useful in any research area, 
where signal propagation through linear complex networks is of practical relevance.

![Example of a recurrent network with 3 nodes that can be modelled using COLNA. The edge parameters ($a$, $\varphi$ and 
 $\Delta t$)](./figures/basic_net.eps)
 
 
## Functionality and Features
Fig. 2 illustrates COLNAs core functionality. Networks are assembled by adding nodes and edges. To verify the 
assembly, networks can be visualized as a graph. In a next step, COLNA computes all paths leading from input to output 
nodes, including recurrent paths, down to a certain amplitude accuracy threshold. 
It supports the mixed use of numeric and symbolic numbers (variables) for all edge properties, returning either a numeric 
or analytic description of the waves' phase and amplitude at the output nodes. Testbenches provide functionality to inject 
complex valued signals to the network and compute the resulting signals at the output nodes.

![Illustration of COLNA's core functions. Assembly and visualization of complex valued networks,
evaluation of the wave propagation from input to output nodes. A testbench injects signals to the network
and records the signals at the output nodes.](./figures/colna_features_plain.eps)

# Statement of Need and Commercial Alternatives
Today, integrated Si photonic circuits (PIC) cover a large range of functionality, from optical interconnects [@Pavesi2016] to 
neuromorphic computing [@Vandoorne2014a] and sensing applications [@Bogaerts2012]. COLNA makes it possible to model such systems and provides an
analytic expression of the systems transfer matrix. The analytic expression is often useful to better understand the effect of 
system parameters' (i.e. edge parameters) variation on the output signal. With its simple interface COLNA is also well 
suited for educational purposes, where analytic expression help to better understand the functionality of simple photonic 
networks, like for example a Mach-Zehnder interferometer. Commercial alternatives to model photonic circuits include for 
example Lumerical Intereconnect [@LumericalInc] or Caphe [@Fiers2012a], which both simulate the signal propagation through 
photonic systems using time- and frequency domain methods. In contrast to COLNA they support non-linear components, but 
do not provide an analytic description of the network. 


# Acknowledgements

This project has received funding from the EU-H2020 research and innovation program under grant no.
688579 (PHRESCO) and from Swiss National Science Foundation under grant no. 175801 (Novel Architectures for Photonic
Reservoir Computing).

# References



