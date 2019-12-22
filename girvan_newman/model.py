"""
Girvan-Newman algorithm are summarized below

    1. The betweenness of all existing edges in the network is calculated first.
    2. The edge(s) with the highest betweenness are removed.
    3. The betweenness of all edges affected by the removal is recalculated.
    4. Steps 2 and 3 are repeated until no edges remain.

Modification
    1. Return the graph with highest modularity, NOT until no edges remain

Reference
    - Rajaraman, Anand, and Jeffrey David Ullman. Mining of massive datasets. Cambridge University Press, 2011.
    - Newman, Mark EJ, and Michelle Girvan. "Finding and evaluating community structure in networks." Physical review E 69.2 (2004): 026113.
    Girvan-newman algorithm: https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm
    Modularity: https://en.wikipedia.org/wiki/Modularity_(networks)
"""
# TODO, theory about modularity, betweeness (weight and credit)
# TODO, type check for user provided setting and data
# if isinstance(data, list):
#     raise TypeError('Input data can not be a list.')
# TODO, directed graph -> undirected graph
# TODO, graph visualization (visualizer.py) and finish demo.py


from __future__ import division, print_function
from operator import add
from collections import deque
import logging
from functools import lru_cache
from copy import deepcopy


__all__ = ["GNModel"]
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GNBetweenessNode:
    """
    Node for calculating betweeness in Girvan-Newman algorithm
    """
    def __init__(self, val):
        self.val = val
        self.parents = []
        self.children = []
        self._credit = None
        self._weight = None

    def add_parent(self, parent):
        self.parents.append(parent)

    def add_child(self, child):
        self.children.append(child)

    @property
    def is_leaf(self):
        return False if self.children else True

    @property
    def weight(self):
        """
        Returns:
            int -- Number of shortest path that begin from root and end at this node
        """
        if not self._weight:
            if not self.parents:
                self._weight = 1
            else:
                self._weight = sum(p.weight for p in self.parents)
        return self._weight

    @property
    def credit(self):
        """
        Returns:
            int -- Credit(some kind of inner state) to calculate betweeness
        """
        if self._credit is None:
            if self.is_leaf:
                self._credit = 1
            else:
                self._credit = 1
                for c in self.children:
                    frac = self.weight / c.weight
                    self._credit += frac * c.credit
        return self._credit

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return self.val == other.val


class GNBetweenessGraph:
    """
    Graph for calculating betweeness in Girvan-Newman algorithm
    """

    def __init__(self, dict_graph):
        self.dict_graph = dict_graph

    @lru_cache()
    def _get_GNBetweenessNode(self, node_val):
        return GNBetweenessNode(node_val)

    def _construct_singleroot_graph(self, root_val):
        """
        Summary
            Get betweeness graph with a given root node
            One node could have multiple parents
            Calculateing weight and credit for each node

        Arguments:
            root {str} -- Value of root node
        
        Returns:
            GNNode -- Root node of this graph
        """
        root_node = self._get_GNBetweenessNode(root_val)
        cur_layer = {root_node}
        past_layers = {root_node}
        while cur_layer:
            next_layer = set()
            for node in cur_layer:
                for new_node in [
                    self._get_GNBetweenessNode(v) for v in self.dict_graph[node.val]
                ]:
                    if new_node in past_layers:
                        continue
                    next_layer.add(new_node)
                    node.add_child(new_node)
                    new_node.add_parent(node)
            past_layers = set.union(past_layers, next_layer)
            cur_layer = next_layer
        self._get_GNBetweenessNode.cache_clear()
        return root_node

    def betweeness(self, root_val):
        root_node = self._construct_singleroot_graph(root_val)
        q = deque([root_node])
        visited = {root_node}
        betweeness = []
        while q:
            node = q.popleft()
            for child in node.children:
                key = (min(node.val, child.val), max(node.val, child.val))
                frac = node.weight / child.weight
                value = frac * child.credit
                betweeness.append([key, value])
                if child not in visited:
                    visited.add(child)
                    q.append(child)
        return betweeness


class GNModularityGraph:
    """
    Graph for calculating modularity in Girvan-Newman algorithm
    """

    def __init__(self, dict_original_graph):
        self.dict_original_graph = deepcopy(dict_original_graph)

    def communities(self, dict_graph):
        """Get communities(connected graphs)
        
        Returns:
            List[List[str]] -- Each community is represented by the vertices of it
        """

        def _dfs(cur_community, node, visited, dict_graph):
            if node in visited:
                return
            else:
                visited.add(node)
                cur_community.append(node)
                for child_node in dict_graph[node]:
                    _dfs(cur_community, child_node, visited, dict_graph)

        _communities = []
        visited_nodes = set()
        for node in dict_graph:
            if node in visited_nodes:
                continue
            cur_community = []
            _dfs(cur_community, node, visited_nodes, dict_graph)
            _communities.append(cur_community)

        _communities = [sorted(c) for c in _communities]
        _communities = sorted(_communities, key=lambda x: (len(x), x[0]))
        return _communities
    
    def modularity(self, dict_graph):
        res = 0
        m = self._num_edges
        for graph in self.communities(dict_graph):
            for v1 in graph:
                for v2 in graph:
                    ki, kj = self._degree(v1), self._degree(v2)
                    aij = self._Aij(v1, v2)
                    res += aij - ki * kj / (2 * m)
        return res / (2 * m)

    def _degree(self, i):
        return len(self.dict_original_graph[i])

    def _Aij(self, i, j):
        return int(j in self.dict_original_graph[i])

    @property
    def _num_edges(self):
        return sum(len(v) for k, v in self.dict_original_graph.items()) // 2


class GNModel:
    """
    Girvan-newman model
    """

    def __init__(self, gndataset, modulairty_decrease_threshold=0.05):
        self.modulairty_decrease_threshold = modulairty_decrease_threshold
        self.rdd_graph = gndataset.rdd.persist()
        self.dict_graph = self.rdd_graph.collectAsMap()

    def highest_betweeness_edges(self, dict_graph):
        gn_betweeness_graph = GNBetweenessGraph(dict_graph)
        res = (
            self.rdd_graph.flatMap(lambda x: gn_betweeness_graph.betweeness(x[0]))
            .reduceByKey(add)
            .map(lambda x: (x[1], [x[0]]))
            .reduceByKey(add)
            .max(lambda x: x[0])
        )
        return res[1]

    @property
    def communities(self):
        original_graph = self.dict_graph
        cur_graph = deepcopy(self.dict_graph)

        gn_modularity_graph = GNModularityGraph(original_graph)
        _communities = None
        highest_modularity = -1
        num_edges = sum(len(v) for k, v in original_graph.items()) // 2

        for iter in range(1, num_edges + 1):
            for i, j in self.highest_betweeness_edges(cur_graph):
                cur_graph[i].remove(j)
                cur_graph[j].remove(i)
            cur_modularity = gn_modularity_graph.modularity(cur_graph)
            if cur_modularity > highest_modularity:
                _communities = gn_modularity_graph.communities(cur_graph)
                highest_modularity = cur_modularity
                logger.info("Iter: {:>2}, Modularity: {:.4f}".format(iter, cur_modularity))
            if highest_modularity - cur_modularity > self.modulairty_decrease_threshold:
                logger.info("FINISH")
                break
        return _communities
