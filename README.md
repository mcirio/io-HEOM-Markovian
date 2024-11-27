# input-output Lindblad equation
## Mauro Cirio, Pengfei Liang, Neill Lambert

Implementation of an input-output extension of a Lindblad master equation modeling a two level system coupled to a 1-dimensional Markovian bath at zero temperature.
It is a special case of the more general [input-output Hierarchical Equations of Motion](https://arxiv.org/pdf/2408.12221).

Specifically, this code computes the reduced system dynamics and the occupation density along a 1-dimensional, zero-temperature environment initially prepared with an additional 1-photon wave-packet propagating towards and scattering upon a two-level system. The results using the input-ouput Lindblad equation are compared to the corresponding analytical solution.

The /scripts directory contains the two files which compute the numerical and analytical solutions. The results are stored in the /data directory. The /plots directory contains a file to plot this data.
