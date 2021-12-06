from pyspark import SparkContext, SparkConf
import sys
from sklearn.datasets import make_blobs, make_circles
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.neighbors import kneighbors_graph


conf = SparkConf().setAppName('appName').setMaster("local")
sc = SparkContext(conf=conf)


def blobs():
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=100, centers=3, n_features=2)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()

def circles():
    # generate 2d classification dataset
    X, y = make_circles(n_samples=100, noise=0.05, factor=0.5)
    edges = assign_neighbors(X, 10)
    weights = assign_weights(X, edges, 10)

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
    while v not in S:
        S.append(v)
        c = v
        v = nearestDict[v][0]
    # edges (edge, neighbors of the edge/distances)
    return (min(c,v), (edges[0], edges[1], min(c,v)))

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

edges, coordinates = circles()
RDD_edges = sc.parallelize(edges)
MAPPED_EDGES = RDD_edges.map(lambda x: (x[0], [x[1], x[2]]))
MAPPED_EDGES = MAPPED_EDGES.combineByKey(
        lambda a: [a],
        combineValues,
        lambda a, b: a.extend(b))

clustering = {}
final = []

while MAPPED_EDGES.count() > 2:

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
            clustering[leaderDict[key]].append(key)
        else:
            clustering[leaderDict[key]] = [key]

    result = MAPPED_EDGES.collect()
    if MAPPED_EDGES.count() <= 2:
        final.extend(result)


clusters = []
for key in final:
    cluster = create_cluster(key[0], clustering)
    print(cluster)
    clusters.append(cluster)

x = []
y = []
ids = []
cluster_id = 0
for cluster in clusters:
    for index in cluster:
        x.append(coordinates[index][0])
        y.append(coordinates[index][1])
        ids.append(cluster_id)
    cluster_id = cluster_id + 1



# scatter plot, dots colored by class value
df = DataFrame(dict(x=x, y=y, label=ids))
colors = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'black', 4: 'green', 5: 'brown', 6: 'white'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

pyplot.show()