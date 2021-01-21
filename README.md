# Metaheuristic-Optimization
In order to study the Vehice Routing Problem (VRP) and the CWS algorithm we developed 2 algorithms:


# CWS-BR
To improve the CWS algorithm we introduce a biased randomized behaviour, in concrete, we use a geometric distribution to guide the random search process,
 
By default, CWS chooses edges in an orderly manner, from highest to lowest savings. Instead of doing this, we apply a random selection using a  geometric distribution with a parameter alpha, which is choosen randomly through a uniform distribution in a range [0.05, 0.35].

One thing to consider is that a geometric distribution is unbound, so to get a random number from it  in such a way that it is in the range defined by the length of the savings list we need a little trick. If q is the random number extrated from the geometric distribution, we’ll compute “q module length of savings list”.

Every time we choose an edge from the saving list we’ll apply this process.

This whole process can be improved using a very simple mechanism, a cache of pseudo-optimized routes. In the end, we choose the solution with the lowest cost. But how can we take advantage of the other routes built for the other solutions? Every time we build a route, we save it on a list. This list will contain all the routes of all the solutions. To improve the best solution, we look to see if any of its routes are on this list, and if we find a route with the same edges, we will replace it with this one if its cost is lower than the original one.

# MD-VRP
Algorithm that solves an extension of the VRP. In particular, the multi-depot VRP.
We select k-depots randomly and we use euclidean distance to create k-clusters so that we'll finally have k CWS-BR problems.
For future work we could apply the K-means algorithm to find the k most optimal clusters.

# This work is based on:
1- On the use of Monte Carlo simulation, cache and splitting techniques to improve the Clarke and Wright savings heuristics - AA Juan 1 , J Faulin 2∗ , J Jorba 1 , D Riera 1 , D Masip 1 and B Barrios 2 - 1 Open University of Catalonia, Barcelona, Spain; and 2 Public University of Navarre, Pamplona, Spain - http://hdl.handle.net/2117/17351

2- Biased randomization of heuristics using skewed probability distributions: A survey and some applications - Alex Grasas, Angel A. Juan, Javier Faulin, Jesica de Armas, Helena Ramalhinho. - https://doi.org/10.1016/j.cie.2017.06.019
