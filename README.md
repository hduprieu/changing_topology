# The role of topology in decentralized learning: extension to random graphs

Project by Henri Duprieu, Valentin Dorseuil, and Pierre Aguié, supervised by El Mahdi El Mhamdi and Aymeric Dieuleveut (CMAP, École polytechnique). This project was conducted as part of the EA "Emerging topics in Machine Learning: Collaborative Learning".

The work presented in this repository buils upon the following paper on decentralized learning. Some of the code in this repository comes from https://github.com/epfml/topology-in-decentralized-learning.  

@article{vogels2022bsg,
  author    = {Thijs Vogels and Hadrien Hendrikx and Martin Jaggi},
  title     = {Beyond spectral gap: The role of the topology in decentralized learning},
  journal   = {CoRR},
  volume    = {abs/TODO},
  year      = {2022},
}

In this article, Vogels et al. study the influence of a communication graph's topology on the performance (# of steps to convergence) of decentralized SGD for static graphs. We extend this study to the case of time-varying graphs, where graphs are randomly sampled from a given distribution at each iteration of the decentralized SGD (D-SGD) algorithm.

The results are presented in `results.ipynb`. The class Topology, used in our practical implementation of D-SGD, is defined in `topologies.py`. This class allows us to extract communication matrices $W$ linked to specific topologies, used to update the parameters of our model at each iteration of D-SDG, and to study the spectral properties of these matrices.
