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


__all__ = ["GNModel"]
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GNBetweenessNode:
    """
    Graph node for calculating betweeness in Girvan-Newman algorithm
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

    def __init__(self, root_val, dict_graph):
        self.nodes = dict()  # TODO, maybe remove this self.node, cause we have hash
        self.root_node = self.construct_bfs_graph(root_val, dict_graph)

    def construct_bfs_graph(self, root_val, dict_graph):
        """
        Summary
            Variant BFS,
            which allows one node could have multiple parents
            which calculates weight and credit for each node

        Arguments:
            root {str} -- Value of root node
        
        Returns:
            GNNode -- Root node of this graph
        """
        root_node = GNBetweenessNode(root_val)
        q = [root_node]
        visited = set([root_val])
        while q:
            next_q = []
            for node in q:
                children = dict_graph[node.val]
                for c in children:
                    if c in visited:
                        continue
                    if c not in self.nodes:
                        self.nodes[c] = GNBetweenessNode(c)
                        next_q.append(self.nodes[c])
                    new_node = self.nodes[c]
                    node.add_child(new_node)
                    new_node.add_parent(node)
            # Update visited per layer, to allow multiple parent
            for node in next_q:
                visited.add(node.val)
            q = next_q
        return root_node

    @property
    def betweeness(self):
        q = deque([self.root_node])
        visited = set([self.root_node.val])
        betweeness = []
        while q:
            node = q.popleft()
            for child in node.children:
                frac = node.weight / child.weight
                value = frac * child.credit
                key = (min(node.val, child.val), max(node.val, child.val))
                betweeness.append([key, value])
                if child.val not in visited:
                    visited.add(child.val)
                    q.append(child)
        return betweeness


class GNModularityGraph:
    """
    Graph for calculating modularity in Girvan-Newman algorithm
    """

    def __init__(self, dict_graph, original_graph, num_edges):
        self.dict_graph = dict_graph
        self._communities = None
        self.original_graph = original_graph
        # m not change
        self.num_edges = num_edges

        # m changes
        # self.num_edges = sum(len(v) for k, v in user_to_neighbors_dict.items()) / 2

    @property
    def communities(self):
        """Get communities(connected graphs)
        
        Returns:
            List[List[str]] -- Each community is represented by the vertices of it
        """
        if self._communities is not None:
            return self._communities

        self._communities = []
        visited_nodes = set()
        for node in self.dict_graph:
            # Skip visited nodes
            if node in visited_nodes:
                continue
            # BFS initialization
            cur_community = [node]
            visited_nodes.add(node)
            q = deque([node])
            # BFS with queue
            while q:
                cur_node = q.popleft()
                for child_node in self.dict_graph[cur_node]:
                    if child_node in visited_nodes:
                        continue
                    visited_nodes.add(child_node)
                    q.append(child_node)
                    cur_community.append(child_node)
            self._communities.append(cur_community)
        return self._communities

    def degree(self, i):
        # k not change
        # return len(self.original_graph[i])

        # k change
        return len(self.dict_graph[i])

    def Aij(self, i, j):
        has_edge = j in self.original_graph[i]
        return int(has_edge)

    @property
    def modularity(self):
        res = 0
        m = self.num_edges
        for graph in self.communities:
            for v1 in graph:
                for v2 in graph:
                    ki, kj = self.degree(v1), self.degree(v2)
                    aij = self.Aij(v1, v2)
                    res += aij - ki * kj / (2 * m)
        return res / (2 * m)


class GNModel:
    """
    Girvan-newman model
    """

    def __init__(self, gndataset, modulairty_decrease_threshold=0.05):
        # why 2, explain
        self.modulairty_decrease_threshold = modulairty_decrease_threshold
        self.rdd_graph = gndataset.rdd.persist()
        self.dict_graph = self.rdd_graph.collectAsMap()
        self.original_graph = {k: v for k, v in self.dict_graph.items()}
        self.num_edges = sum(len(v) for k, v in self.dict_graph.items()) // 2

    @staticmethod
    def single_root_betweeness(root, dict_graph):
        betweeness_graph = GNBetweenessGraph(root, dict_graph)
        return betweeness_graph.betweeness

    @property
    def betweeness(self):
        # Avoid lambda function in rdd.map contains another rdd (self)
        tmp = self.dict_graph
        res = (
            self.rdd_graph.flatMap(lambda x: GNModel.single_root_betweeness(x[0], tmp))
            .reduceByKey(add)
            .map(lambda x: (x[0], x[1] / 2))
            .collect()
        )
        return res

    @property
    def highest_betweeness_edges(self):
        tmp = self.dict_graph
        res = (
            self.rdd_graph.flatMap(lambda x: GNModel.single_root_betweeness(x[0], tmp))
            .reduceByKey(add)
            .map(lambda x: (x[1], [x[0]]))
            .reduceByKey(add)
            .max(lambda x: x[0])
        )
        return res[1]

    def remove_edge(self, i, j):
        for k, v in [[i, j], [j, i]]:
            self.dict_graph[k].remove(v)

    @property
    def communities(self):
        highest_modularity = -1
        highest_m_graph = None
        cnt = 0
        num_remaining_edges = self.num_edges
        # Keep removing edges until no edge remaining
        while num_remaining_edges > 0:
            cnt += 1
            edges_to_remove = self.highest_betweeness_edges
            num_remaining_edges -= len(edges_to_remove)
            for i, j in edges_to_remove:
                self.remove_edge(i, j)
            cur_m_graph = GNModularityGraph(
                self.dict_graph, self.original_graph, self.num_edges
            )
            cur_m = cur_m_graph.modularity
            if cur_m > highest_modularity:
                highest_m_graph = cur_m_graph
                highest_modularity = cur_m
                logger.info(
                    "Iter: {:>2}, Num_edge_to_remove: {}, Modularity: {:.4f}".format(
                        cnt, len(edges_to_remove), cur_m
                    )
                )
            # Early quit
            if highest_modularity - cur_m > self.modulairty_decrease_threshold:
                break
        return highest_m_graph.communities
