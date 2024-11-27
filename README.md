# io-HEOM
This is an python implementation of an input-output extension of a Lindblad master equation modeling a two level system coupled to a 1-dimensional Markovian bath at zero temperature.
It constitutes a specific case of the more general [input-output Hierarchical Equations of Motion](https://arxiv.org/pdf/2408.12221).

Specifically, this code computes the reduced system dynamics and the occupation density along the the 1-dimensional, zero-temperature environment initially prepared with an additional 1-photon wave-packet propagating towards the two-level system. It further compares these results with the corresponding analytical expression.

The two files in the /scripts directory provide the numerical and analytical solutions. The results are stored in the /data directory. The file in the /plots directory uses this data to produce the plots.
