import igraph
import pickle
import networkx as nx

from processing._influencers_model.db_csv import DatasetInfluencersModel
from settings import NX_SUBGRAPH_PATH, NX_SUBGRAPH_PATH_ML, INFLUENCE_POINTS


class InfluenceNode(object):
    def __init__(self, centrality=None, community=None, node_id=None):
            super(InfluenceNode, self).__init__()
            self.user_id = node_id
            self.centrality = sum(map(lambda x: centrality[x], centrality.keys()))
            self.community = community
            self.activity = 0
            self.user = None
            self.influence = 0

    def load_user_counts(self, counts):
        # s = open_session()
        # user_instance = s.query(User).filter(User.id==self.user_id).first()
        self.user = self.user_id
        # if self.user==None:
        #     print(self.user_id)
        # else:
        try:
            self.activity = counts[str(self.user_id)]
        except KeyError:
            self.activity = 0
        # self.activity = user_instance.retweets.count()
        # s.close()

    def normalize_activity(self, max_value):
        self.activity = self.activity / float(max_value)

    def normalize_centrality(self, max_value):
        self.centrality = self.centrality / float(max_value)

    def calculate_influence(self):
        self.influence = .5 * self.activity + .5 * self.centrality
        return self.influence


class Grafo(object):
    def __init__(self, graph=NX_SUBGRAPH_PATH_ML):
        super(Grafo, self).__init__()
        print("-Loading graph from {}..".format(graph))
        self.graph = igraph.Graph.Read_GraphML(graph)

    def get_influence_nodes(self):
        print("-Loading nodes as influence points..")
        nodes = []
        eigen = self.graph.eigenvector_centrality()
        betw = self.graph.betweenness()
        betw = list(map(lambda x: x / max(betw), betw))
        eccentricity = self.graph.eccentricity()
        eccentricity = list(map(lambda x: x / max(eccentricity), eccentricity))
        for i, node in enumerate(self.graph.vs):
            nodes.append(InfluenceNode(node_id=node['id'],
                                       centrality=dict(
                                           degree=node.degree(),
                                           pagerank=node.pagerank(),
                                           betweeness=betw[i],
                                           closeness=node.closeness(),
                                           eigen=eigen[i],
                                           eccentricity=eccentricity[i]
                                       ),
                                       community=0))
        return nodes


class InfluenceActions(object):

    @staticmethod
    def save_graph_as_graphml():
        gpickle_fn = NX_SUBGRAPH_PATH
        graph = nx.read_gpickle(gpickle_fn)
        nx.write_graphml(graph, NX_SUBGRAPH_PATH_ML)

    @staticmethod
    def build_influence_points():
        g = Grafo()
        influence_points = g.get_influence_nodes()
        print("-Calculating influence of each node..")
        DatasetInfluencersModel._load_df()
        df = DatasetInfluencersModel.df
        counts = df.user__id_str.groupby(df.user__id_str).count()
        for point in influence_points:
            point.load_user_counts(counts)
        max_activity = max(map(lambda x: x.activity, influence_points))
        max_centrality = max(map(lambda x: x.centrality, influence_points))
        for point in influence_points:
            point.normalize_activity(max_activity)
            point.normalize_centrality(max_centrality)
            influence = point.calculate_influence()
            # print(point.influence)
        with open(INFLUENCE_POINTS, 'wb') as save_file:
            pickle.dump(influence_points, save_file)

    @staticmethod
    def load_influencers_from_pickle(filename):
        print("Loading influence points from pickle {}".format(filename))
        with open(filename, 'r+') as f:
            influencers = pickle.load(f)
        return influencers
