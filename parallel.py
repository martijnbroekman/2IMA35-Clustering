from pyspark import SparkContext, SparkConf
import sys
from sklearn.datasets import make_blobs, make_circles, make_moons
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.neighbors import kneighbors_graph
import numpy as np
from itertools import combinations
import findspark
findspark.init()

conf = SparkConf().setAppName('appName').setMaster("local")
sc = SparkContext(conf=conf)


def blobs(n, n_extra, steps, blobs):
    # generate 2d classification dataset
    X_all, y_all = make_blobs(n_samples=10**(n_extra), centers=blobs, n_features=2)
    index = 0
    edges = []

    # Create subsets with size n + steps * number of iteration
    for i in np.logspace(n, n_extra, steps).astype(int):
        X = X_all[:i]
        y = y_all[:i]
        edges.append(assign_neighbors(X, int(y.shape[0] / blobs * 1.1)))
        weights = assign_weights(X, edges[index], int(y.shape[0] / blobs * 1.1))
        index += 1

    return edges, X_all, y_all

def moons(n, n_extra, steps):
    # generate 2d classification dataset
    X_all, y_all = make_moons(n_samples=10 ** (n_extra), noise=0.1)
    index = 0
    edges = []

    # Create subsets with size n + steps * number of iteration
    for i in np.logspace(n, n_extra, steps).astype(int):
        X = X_all[:i]
        y = y_all[:i]
        edges.append(assign_neighbors(X, int(y.shape[0] / 2 * 1.1)))
        weights = assign_weights(X, edges[index], int(y.shape[0] / 2 * 1.1))
        index += 1

    return edges, X_all, y_all


def circles(n, n_extra, steps):
    # generate 2d classification dataset
    X, y = make_circles(n_samples=n, noise=0.05, factor=0.5)
    edges = []
    edges.append(assign_neighbors(X, int(y.shape[0] / 2 * 1.1)))
    weights = assign_weights(X, edges[0], int(y.shape[0] / 2 * 1.1))
    index = 1

    number_steps = int((n + n_extra)/steps)
    # Create subsets of size n + n_extra * number of iteration
    for i in range(number_steps):
        X2, y2 = make_circles(n_samples=n + i * steps, noise=0)
        X = np.concatenate((X, X2))
        y = np.concatenate((y, y2))
        edges.append(assign_neighbors(X, int(y.shape[0] / 2 * 1.1)))
        weights = assign_weights(X, edges[index], int(y.shape[0] / 2 * 1.1))
        index += 1

    return edges, X


def assign_neighbors(points, n):
    knn_graph = kneighbors_graph(points, n, include_self=False, mode='distance')
    weighted_edges = []

    for i in range(len(points)):
        for j in range(len(points)):
            weight = knn_graph[i, j]
            if weight > 0:
                weighted_edges.append((i, j, weight))

    return weighted_edges


def assign_weights(points, neighbors, n):
    weights = []
    point = -1
    minimum = sys.maxsize
    for i in range(len(neighbors)):
        neighbor_weight = neighbors[i][2]

        if neighbor_weight < minimum:
            minimum = neighbor_weight

        if i % n == 0:
            weights.append(minimum)
            point = point + 1
            minimum = sys.maxsize

    return weights

def combineValues(range, value):
    newRange = []
    for item in range:
        newRange.append(item)

    newRange.append(value)

    return newRange

def map_contraction(edges):
    c = edges[0]
    v = edges[0]
    S = []
    try:
        while v not in S:
            S.append(v)
            c = v
            v = nearestDict[v][0]
        # edges (edge, neighbors of the edge/distances)
        return (min(c, v), (edges[0], edges[1], min(c, v)))
    except:
        return (v, (edges[0], edges[1], v))

def reduce_contraction(neighbours1, neighbours2):
    neighbors_of_leader = []
    for neighbor in neighbours1[1]:
        leader_of_follower = leaderDict[neighbor[0]]
        if leader_of_follower != neighbours1[2]:
            neighbors_of_leader.append([leader_of_follower, neighbor[1]])
    for neighbor in neighbours2[1]:
        leader_of_follower = leaderDict[neighbor[0]]
        if leader_of_follower != neighbours2[2]:
            neighbors_of_leader.append([leader_of_follower, neighbor[1]])

    other_leaders_dict = {}

    for neighbor in neighbors_of_leader:
        leader_of_neighbor = leaderDict[neighbor[0]]

        if leader_of_neighbor in other_leaders_dict:
            if other_leaders_dict[leader_of_neighbor] > neighbor[1]:
                other_leaders_dict[leader_of_neighbor] = neighbor[1]
        else:
            other_leaders_dict[leader_of_neighbor] = neighbor[1]

    results = [(key, other_leaders_dict[key]) for key in other_leaders_dict.keys()]
    return (neighbours1[2], results, neighbours1[2])

def smallest(values):
    min = sys.maxsize
    vertex = 0
    for value in values:
        if value[1] < min:
            min = value[1]
            vertex = value[0]
    return vertex, min


def create_cluster(leader, clusters):
    total = []
    if leader not in clusters:
        return total

    for follower in clusters[leader]:
        total.append(follower)
        if follower is not leader:
            total.extend(create_cluster(follower, clusters))

    # Create distinct
    return list(set(total))

begin_n = 1 # 10^1
number_clusters = 5
steps = 50
total_n = 3 # 10^3
number_iterations = 100
counter_true_cluster = dict()
sum_recall = dict()
sum_precision = dict()
sum_F = dict()
error = dict()

for number_points in np.logspace(begin_n, total_n, steps).astype(int):
    counter_true_cluster[number_points] = 0
    sum_recall[number_points] = 0
    sum_precision[number_points] = 0
    sum_F[number_points] = 0
    error[number_points] = 0
for boot in range(number_iterations):
    print(boot)
    print(counter_true_cluster)
    print(sum_precision)
    print(sum_recall)
    print(sum_F)
    print(error)
    edges_all, coordinates, y_true = blobs(begin_n, total_n, steps, number_clusters)
    n = np.logspace(begin_n, total_n, steps).astype(int)
    index_n = 0
    for edges in edges_all:
        RDD_edges = sc.parallelize(edges)
        MAPPED_EDGES = RDD_edges.map(lambda x: (x[0], [x[1], x[2]]))
        MAPPED_EDGES = MAPPED_EDGES.combineByKey(
                lambda a: [a],
                combineValues,
                lambda a, b: a.extend(b))

        clustering = {}
        final = []
        try:
            while MAPPED_EDGES.count() > number_clusters:

                # RDD_nearest = MAPPED_EDGES.reduceByKey(lambda a, b: a if a[1] < b[1] else b).sortByKey()


                RDD_nearest = MAPPED_EDGES.map(lambda values: (values[0], smallest(values[1])))
                nearest = RDD_nearest.collect()
                nearestDict = {neighbor[0]: neighbor[1] for neighbor in nearest}
                sc.broadcast(nearestDict)
                RDD_contracted = MAPPED_EDGES.map(map_contraction)

                contractedValues = RDD_contracted.collect()
                leaderDict = {leaderNodePair[1][0]: leaderNodePair[0] for leaderNodePair in contractedValues}
                sc.broadcast(leaderDict)

                RDD = RDD_contracted.reduceByKey(reduce_contraction)

                MAPPED_EDGES = RDD.map(lambda pair: pair[1])

                for key in leaderDict.keys():
                    if leaderDict[key] in clustering:
                        if key not in clustering[leaderDict[key]]:
                            clustering[leaderDict[key]].append(key)
                    else:
                        clustering[leaderDict[key]] = [key]

                result = MAPPED_EDGES.collect()
                if MAPPED_EDGES.count() <= number_clusters:
                    final.extend(result)


            print(n[index_n])
            if len(final) == number_clusters:
                counter_true_cluster[n[index_n]] += 1

            clusters = []
            pairs = []
            p = 0
            for key in final:
                cluster = create_cluster(key[0], clustering)
                # print(cluster)
                clusters.append(cluster)
                # Create sets of all possible combinations
                pairs.extend(list(combinations(cluster, 2)))
                p += len(cluster) * (len(cluster) - 1) / 2

            # Look to only one true cluster and find all possible combinations and create sets of these
            pairs_true = []
            q = 0
            for ii in np.unique(y_true[:n[index_n]]):
                cluster_true = (np.where(y_true[:n[index_n]] == ii)[0])
                pairs_true.extend(list(combinations(cluster_true, 2)))
                q += len(cluster_true) * (len(cluster_true) - 1) / 2

            intersect = len(list(filter(lambda x:x in pairs, pairs_true))) # True positives
            not_in_pairs = len(list(set(pairs_true) - set(pairs))) # False negatives
            not_in_true = len(list(set(pairs) - set(pairs_true))) # False positives
            precision = intersect / p
            recall = intersect / q
            F = 2 * intersect / (2 * intersect + not_in_pairs + not_in_true)
            print(precision, recall, F)
            sum_precision[n[index_n]] += precision
            sum_recall[n[index_n]] += recall
            sum_F[n[index_n]] += F

            # x = []
            # y = []
            # ids = []
            # cluster_id = 0
            # for cluster in clusters:
            #     for index in cluster:
            #         x.append(coordinates[index][0])
            #         y.append(coordinates[index][1])
            #         ids.append(cluster_id)
            #     cluster_id = cluster_id + 1
            # print(y)
            #
            #
            # # scatter plot, dots colored by class value
            # df = DataFrame(dict(x=x, y=y, label=ids))
            # colors = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'black', 4: 'green', 5: 'brown', 6: 'white'}
            # fig, ax = pyplot.subplots()
            # grouped = df.groupby('label')
            # for key, group in grouped:
            #     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
            #
            # pyplot.show()

        except:
            error[n[index_n]] += 1

        index_n += 1

print(counter_true_cluster)
print(sum_precision)
print(sum_recall)
print(sum_F)
print(error)
print({k: v / number_iterations for k, v in sum_precision.items()})
print({k: v / number_iterations for k, v in sum_recall.items()})
print({k: v / number_iterations for k, v in sum_F.items()})


