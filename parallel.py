from pyspark import SparkContext, SparkConf
import sys
from sklearn.datasets import make_blobs, make_circles
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


# Just some results for myself
# Total running tme 13 hours with only 1000 vertices, 5 clusters.
# Assigning the weights takes longest time namely |V|^3.
# So more vertices would be impossible.
# 1 iteration
# {10: 2, 12: 1, 13: 1, 14: 1, 15: 1, 17: 1, 19: 0, 21: 0, 23: 0, 25: 0, 28: 0, 30: 1, 33: 0, 37: 0, 40: 0, 44: 1, 49: 1, 54: 0, 59: 0, 65: 0, 71: 0, 79: 0, 86: 0, 95: 0, 104: 0, 115: 0, 126: 0, 138: 0, 152: 0, 167: 1, 184: 0, 202: 0, 222: 0, 244: 0, 268: 0, 294: 0, 323: 0, 355: 0, 390: 0, 429: 0, 471: 0, 517: 0, 568: 0, 625: 0, 686: 0, 754: 1, 828: 1, 910: 1, 1000: 0}
# {10: 1.2, 12: 0.5555555555555556, 13: 0.6363636363636364, 14: 0.7142857142857143, 15: 0.7058823529411765, 17: 0.75, 19: 0.3763440860215054, 21: 0.3867924528301887, 23: 0.40229885057471265, 25: 0.37962962962962965, 28: 0.5463917525773195, 30: 0.7325581395348837, 33: 0.7633587786259542, 37: 0.6057142857142858, 40: 0.5467980295566502, 44: 0.4774774774774775, 49: 0.5, 54: 0.37037037037037035, 59: 0.4103114930182599, 65: 0.7087667161961367, 71: 0.6724565756823822, 79: 0.6686626746506986, 86: 0.6809403862300588, 95: 0.6364874063989108, 104: 0.6246482836240855, 115: 0.6145124716553289, 126: 0.6162264150943396, 138: 0.5454545454545454, 152: 0.4319043452901721, 167: 0.38355477563299073, 184: 0.4040315512708151, 202: 0.4748595303270422, 222: 0.53079731480391, 244: 0.37058675713029277, 268: 0.5308787664227834, 294: 0.5189551849921834, 323: 0.5164936188622107, 355: 0.5204617941691748, 390: 0.5605553074289117, 429: 0.5680566726474046, 471: 0.37219836659088995, 517: 0.5485432805658471, 568: 0.4086710369487485, 625: 0.39469940519529984, 686: 0.4975644455440951, 754: 0.6677686827077771, 828: 0.6742259880379969, 910: 0.699499793604157, 1000: 0.3834296724470135}
# {10: 0.6666666666666666, 12: 0.3333333333333333, 13: 0.4117647058823529, 14: 0.5, 15: 0.5217391304347826, 17: 0.6428571428571429, 19: 1.0, 21: 1.0, 23: 0.660377358490566, 25: 0.6949152542372882, 28: 0.7066666666666667, 30: 0.7411764705882353, 33: 0.970873786407767, 37: 0.7851851851851852, 40: 0.7025316455696202, 44: 0.5698924731182796, 49: 0.584, 54: 0.9206349206349206, 59: 1.0, 65: 1.0, 71: 1.0, 79: 1.0, 86: 1.0, 95: 1.0, 104: 1.0, 115: 1.0, 126: 1.0, 138: 0.8758901322482198, 152: 0.6272765777213045, 167: 0.5404450724125751, 184: 0.6803423848878394, 202: 0.8096290837632032, 222: 0.9214884481701083, 244: 1.0, 268: 0.9443045060173524, 294: 0.9289878731343284, 323: 0.9204895913646878, 355: 0.9265834932821497, 390: 1.0, 429: 1.0, 471: 1.0, 517: 0.9808408928369782, 568: 0.9405255447149403, 625: 0.9049257794442447, 686: 0.6868609243249013, 754: 0.6677686827077771, 828: 0.6742259880379969, 910: 0.699499793604157, 1000: 1.0}
# {10: 0.8571428571428571, 12: 0.4166666666666667, 13: 0.5, 14: 0.5882352941176471, 15: 0.6, 17: 0.6923076923076923, 19: 0.546875, 21: 0.5578231292517006, 23: 0.5, 25: 0.49101796407185627, 28: 0.6162790697674418, 30: 0.7368421052631579, 33: 0.8547008547008547, 37: 0.6838709677419355, 40: 0.6149584487534626, 44: 0.5196078431372549, 49: 0.5387453874538746, 54: 0.5282331511839709, 59: 0.5818735719725818, 65: 0.8295652173913044, 71: 0.8041543026706232, 79: 0.8014354066985646, 86: 0.8101898101898102, 95: 0.7778702163061564, 104: 0.7689643228264634, 115: 0.7612359550561798, 126: 0.7625496147560121, 138: 0.6722623462814757, 152: 0.5115716753022452, 167: 0.44868035190615835, 184: 0.5069833938194215, 202: 0.5986196876135126, 222: 0.6735913914213122, 244: 0.5407709584269148, 268: 0.6796595658961575, 294: 0.6659144098963558, 323: 0.6617015380351947, 355: 0.6665324320437221, 390: 0.7184049226072646, 429: 0.7245358953619129, 471: 0.5424847830355389, 517: 0.7035951992007669, 568: 0.5697695278200032, 625: 0.5496564257357013, 686: 0.5770858741660091, 754: 0.6677686827077771, 828: 0.6742259880379969, 910: 0.699499793604157, 1000: 0.5543175487465181}

# 7 iterations
# {10: 0, 12: 0, 13: 2, 14: 2, 15: 5, 17: 1, 19: 2, 21: 1, 23: 0, 25: 0, 28: 0, 30: 0, 33: 0, 37: 0, 40: 0, 44: 1, 49: 2, 54: 3, 59: 2, 65: 3, 71: 2, 79: 1, 86: 1, 95: 0, 104: 0, 115: 0, 126: 1, 138: 2, 152: 3, 167: 2, 184: 3, 202: 7, 222: 5, 244: 5, 268: 2, 294: 4, 323: 1, 355: 1, 390: 0, 429: 0, 471: 0, 517: 0, 568: 0, 625: 1, 686: 3, 754: 1, 828: 4, 910: 5, 1000: 3}
# {10: 7.359890109890109, 12: 3.4022453569821987, 13: 3.58030303030303, 14: 3.70139775683254, 15: 3.7554657477025897, 17: 2.2067517086571966, 19: 3.209147394674625, 21: 3.35414362999596, 23: 2.9328526026993003, 25: 2.8385846572249966, 28: 3.318060486343289, 30: 3.549296435181985, 33: 4.15959063496417, 37: 3.7605836337190084, 40: 3.542652512458611, 44: 3.590285722942102, 49: 3.2864305974799777, 54: 3.3221298220474287, 59: 3.548787304513651, 65: 3.5316836497282127, 71: 2.990402890513842, 79: 3.7833058494814855, 86: 3.9446205044384683, 95: 2.910162293933882, 104: 3.5294208022321096, 115: 3.507302340629434, 126: 3.7737663449398653, 138: 4.075126482595787, 152: 4.037931412419775, 167: 3.6153693815238173, 184: 3.8295926015859245, 202: 4.238474281126946, 222: 3.6335574395324373, 244: 4.1704320750527515, 268: 3.251333587159813, 294: 3.2166534972101113, 323: 2.788916156297616, 355: 3.105177976334309, 390: 2.965643413500479, 429: 3.721200967176232, 471: 3.9406537035594087, 517: 3.4430778812540677, 568: 3.1307381314666882, 625: 3.5572639245201776, 686: 3.7035751084494675, 754: 3.4106380547506614, 828: 3.8364880668112464, 910: 4.3645621708428655, 1000: 3.8861419878587897}
# {10: 8.972527472527474, 12: 4.568406593406594, 13: 3.963582032003085, 14: 4.625185680332739, 15: 4.156302521008404, 17: 3.898078529657477, 19: 5.788704145360119, 21: 6.510004190236748, 23: 6.802884615384615, 25: 5.412699215785479, 28: 5.724396461238567, 30: 5.584534030110738, 33: 6.131499290349946, 37: 5.265209552684446, 40: 4.60111442089553, 44: 4.678073639015368, 49: 4.934096915600902, 54: 4.661776760662593, 59: 5.4529069414752325, 65: 4.884169476664108, 71: 5.036678403116408, 79: 6.3147313250590535, 86: 6.567612950585442, 95: 5.815503397920547, 104: 6.955004591368228, 115: 5.9436283614775824, 126: 5.844258617330079, 138: 5.946444444209389, 152: 5.1654415746281845, 167: 5.0645901898437256, 184: 4.806790508525979, 202: 4.48556878032835, 222: 4.155284058904183, 244: 5.433204359964947, 268: 6.120634817180814, 294: 5.573442107830609, 323: 5.1526569539996085, 355: 6.366727335420873, 390: 5.93588018847681, 429: 6.8260809117708, 471: 6.960881340424267, 517: 6.867580352858559, 568: 6.003705552354408, 625: 4.887033552367505, 686: 4.500622585564972, 754: 4.946357697438953, 828: 5.2870756882231, 910: 4.848801410897634, 1000: 5.6137286432160804}
# {10: 8.02015607580825, 12: 3.867529529294235, 13: 3.7367720429348337, 14: 3.831377967047554, 15: 3.581484102866961, 17: 2.6477693024457927, 19: 3.9497655128374114, 21: 4.283744982089758, 23: 4.0341415586024665, 25: 3.6507073312079434, 28: 4.14173678769108, 30: 4.312935586025676, 33: 4.913828530584996, 37: 4.330486166512028, 40: 3.9839479043307193, 44: 4.03018200973033, 49: 3.7456348715073315, 54: 3.6706853782300293, 59: 4.131324348803383, 65: 3.9735965538058964, 71: 3.5842332388925158, 79: 4.621183526449286, 86: 4.801010651308133, 95: 3.8126491748690463, 104: 4.61868494330399, 115: 4.351223657688877, 126: 4.449668319982744, 138: 4.691064486697139, 152: 4.4491587840507805, 167: 4.147167506312385, 184: 4.2044902444181975, 202: 4.350654711129867, 222: 3.829306370037706, 244: 4.584694137860003, 268: 4.046698067333367, 294: 3.827347838791593, 323: 3.496605169885138, 355: 4.091110728388522, 390: 3.842386936519798, 429: 4.751358361579146, 471: 4.983127670114781, 517: 4.566537728361842, 568: 4.106036335761386, 625: 4.090740972189215, 686: 4.032840075374837, 754: 3.966946169421621, 828: 4.217510581342476, 910: 4.567674785568539, 1000: 4.435574860192443}
# {10: 0, 12: 0, 13: 0, 14: 0, 15: 0, 17: 2, 19: 0, 21: 0, 23: 0, 25: 1, 28: 1, 30: 1, 33: 0, 37: 0, 40: 0, 44: 0, 49: 0, 54: 0, 59: 0, 65: 1, 71: 1, 79: 0, 86: 0, 95: 1, 104: 0, 115: 1, 126: 1, 138: 0, 152: 0, 167: 0, 184: 0, 202: 0, 222: 1, 244: 0, 268: 0, 294: 0, 323: 1, 355: 0, 390: 1, 429: 0, 471: 0, 517: 0, 568: 0, 625: 0, 686: 0, 754: 0, 828: 0, 910: 0, 1000: 0}
