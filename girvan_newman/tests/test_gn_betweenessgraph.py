import unittest
from girvan_newman.model import GNBetweenessGraph


DUMMY_GRAPH = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B"],
    "D": ["B", "E", "F", "G"],
    "E": ["D", "F"],
    "F": ["D", "E", "G"],
    "G": ["D", "F"]
}


class TestGNBetweenessGraph(unittest.TestCase):
    def test_construct_graph(self):
        root_node = GNBetweenessGraph('E', DUMMY_GRAPH)
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    unittest.main()