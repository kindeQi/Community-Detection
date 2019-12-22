from __future__ import division, print_function
import pandas
from collections import defaultdict
from girvan_newman import GNDataset, GNModel


def data_processing(input_file, num_common_prod_threshold):
    """Build graph as adjancency list from input.
    Only users with number of common porducts >= threshold would have edge in between
    
    Arguments:
        input_file {str} -- Path of inputfile
    
    Keyword Arguments:
        num_common_prod_threshold {int} -- Threshold
    
    Returns:
        graph -- defaultdict[key: int, value: List[int]]
    """
    # Read input from csv file
    video_csv = pandas.read_csv("./demo_datafile.csv")
    video_list = video_csv.to_numpy()[:, :-1].tolist()
    user_to_product = defaultdict(set)
    for user, prod, rating in video_list:
        user_to_product[user].add(prod)
    user_list = list(user_to_product.keys())

    # Find edges and construct the graph as adjancency list
    graph = defaultdict(list)
    for i in range(len(user_list)):
        u1 = user_list[i]
        prod1 = user_to_product[u1]
        for j in range(i + 1, len(user_list)):
            u2 = user_list[j]
            prod2 = user_to_product[u2]
            if len(set.intersection(prod1, prod2)) >= num_common_prod_threshold:
                graph[u1].append(u2)
                graph[u2].append(u1)
    return graph


if __name__ == "__main__":
    # Data processing
    graph = data_processing("./demo_datafile.csv", 7)
    gndataset = GNDataset(graph)

    # Find the community
    gnmodel = GNModel(gndataset)  # TODO, change api here, as using GNDataset
    res = gnmodel.communities
    res = [sorted(str(n) for n in c) for c in res]
    res = sorted(res, key=lambda x: (len(x), x[0]))
    for r in res:
        print(r)
