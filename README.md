# Community-Detection
Community detection in a graph with Girvan-Newman algorithm. It's originally a course project of [INF553](https://web-app.usc.edu/soc/syllabus/20191/32414.pdf), and is rewritten to be a more general project

# Quick Start
See [demo.ipynb](./demo.ipynb)

# Girvan-Newman Algorithm
## Definition
- Betweeness: the betweenness of an edge (a,b) to be the number of pairs of nodes x and y such that the edge (a,b) lies on the shortest path between x and y. To be more precise, since there can be several shortest paths between x and y, edge (a,b) is credited with the fraction of those shortest paths that include the edge (a,b). As in golf, a high score is bad. It suggests that the edge (a,b) runs between two different communities; that is, a and b do not belong to the same community.
- Label: the number of shortest paths that reach it from the root. Start by labeling the root 1. Then, from the top down, label each node Y by the sum of the labels of its parents.
- Credit: 
  - Each leaf in the DAG (a leaf is a node with no DAG edges to nodes at levels below) gets a credit of 1
  - Each node that is not a leaf gets a credit equal to 1 plus the sum of the credits of the DAG edges from that node to the level below.
  - A DAG edge e entering node Z from the level above is given a share of the
credit of Z proportional to the fraction of shortest paths from the root to
## Summary
  1. The betweenness of all existing edges in the network is calculated first.
  2. The edge(s) with the highest betweenness are removed.
  3. The betweenness of all edges affected by the removal is recalculated.
  4. Steps 2 and 3 are repeated until no edges remain.

# References
- [Rajaraman, Anand, and Jeffrey David Ullman. Mining of massive datasets (pp.349-356). Cambridge University Press, 2011.](http://infolab.stanford.edu/~ullman/mmds/ch10.pdf)
- Newman, Mark EJ, and Michelle Girvan. "Finding and evaluating community structure in networks." Physical review E 69.2 (2004): 026113.
