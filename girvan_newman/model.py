"""
Girvan-Newman algorithm are summarized below

    1. The betweenness of all existing edges in the network is calculated first.
    2. The edge(s) with the highest betweenness are removed.
    3. The betweenness of all edges affected by the removal is recalculated.
    4. Steps 2 and 3 are repeated until no edges remain.

Slightly modified
    Return the graph with highest modularity, NOT until no edges remain
    Unweighted graph

Reference
    - Rajaraman, Anand, and Jeffrey David Ullman. Mining of massive datasets. Cambridge University Press, 2011.
    - Newman, Mark EJ, and Michelle Girvan. "Finding and evaluating community structure in networks." Physical review E 69.2 (2004): 026113.
    Girvan-newman algorithm: https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm
    Modularity: https://en.wikipedia.org/wiki/Modularity_(networks)
"""