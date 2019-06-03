import json
import os
import time

import networkx as nx
import pickle

from tweepy import TweepError

from preparation.twitter_api import API_HANDLER


# TMP_GRAPH_PATH = './preparation/temp/subgraph_new_tmp.gpickle'
# TMP_SEEN_USERS = './preparation/temp/seen_users.pickle'
#
# if os.path.exists(TMP_SEEN_USERS):
#     with open(TMP_SEEN_USERS, 'rb') as f:
#         SEEN_USERS = pickle.load(f)
# else:
#     SEEN_USERS = set()
from settings import CENTRAL_USER_DATA, NX_GRAPH_FOLDER

NOTAUTHORIZED_FNAME = "./temp/notauthorizedids.pickle"

if os.path.exists(NOTAUTHORIZED_FNAME):
    with open(NOTAUTHORIZED_FNAME, 'rb') as f:
        NOTAUTHORIZED = pickle.load(f)
else:
    NOTAUTHORIZED = set()


NOTFOUND_FNAME = "./temp/notfound.pickle"

if os.path.exists(NOTFOUND_FNAME):
    with open(NOTFOUND_FNAME, 'rb') as f:
        NOTFOUND = pickle.load(f)
else:
    NOTFOUND = set()


class GraphHandler(object):
    """Networkx graph handling based on other existing graph, nodes will be reused!"""
    def __init__(self, folder_path, central_uid, level=3):
        self.api = API_HANDLER
        self.level = level
        self.graph_path_folder = folder_path
        self.uid = central_uid

        # attempt to load full graph
        graph_path = self._get_file_path_full_graph(self.uid, self.level)
        print(graph_path)
        print(graph_path)
        print(graph_path)
        print(graph_path)
        if os.path.exists(graph_path):
            self.g = nx.read_gpickle(graph_path)
        else:
            print('WARNING: No pre-existing graph found! to load')

        # attempt to load subgraph
        graph_path = self._get_file_path_k_closure(self.uid)
        if os.path.exists(graph_path):
            self.subg = nx.read_gpickle(graph_path)
        else:
            print('WARNING: No pre-existing sub graph found! to load')

        # load dict of relevant users
        self.relevant_filepath = self._get_file_path_full_graph(self.uid, level=0) + 'relevantdit.json'
        if os.path.exists(self.relevant_filepath):
            with open(self.relevant_filepath, 'r') as f:
                self.relevant_users = json.load(f)
        else:
            print('Relevant users not found to load')
            self.relevant_users = {}

    def _get_file_path_full_graph(self, central_uid=None, level=None):
        central_uid = central_uid if central_uid is not None else self.uid
        level = level if level is not None else self.level

        return self.graph_path_folder + '{}_{}.gpickle'.format(central_uid, level)

    def _get_file_path_k_closure(self, central_uid=None):
        central_uid = central_uid if central_uid is not None else self.uid

        return self.graph_path_folder + '{}_subgraph.gpickle'.format(central_uid)

    def build_graph_to_level(self, level=None):
        level = level if level is not None else self.level

        graph = nx.DiGraph()
        self.g = graph
        outer_layer_ids = set([self.uid])

        fname_current = self._get_file_path_full_graph(level=level)
        fname_outer_layer = fname_current + 'outer_layer.pickle'
        nx.write_gpickle(self.g, fname_current)
        with open(fname_outer_layer, 'wb') as fl:
            pickle.dump(outer_layer_ids, fl)
        print('Done level 0 graph')
        # COMENZAR BIEN EL LOOP

        for l in range(1, level + 1):
            print('Now running level {}; EDGES: {}\tNODES: {}'.format(l, len(self.g.edges), len(self.g.nodes)))
            outer_layer_ids = self.extend_followed_graph(outer_layer_ids, l, save=True)
            print('Done running level {}; EDGES: {}\tNODES: {}'.format(l, len(self.g.edges), len(self.g.nodes)))
        self.g = graph

    def extend_followed_graph(self, outer_layer_ids, level, save=False):
        """
            Given a graph and the ids of its outer layer,
            it extends it with one extra step in the followed
            relation.

            This is meant to be applied up to a certain number
            of steps. The outer layer are the nodes that were
            seen for the first time in the previous step

            Level 0 are just a set of selected relevant users.
            After that, level N + 1 is the extension of level N
            by one more step in the followed relation
        """
        fname_current = self._get_file_path_full_graph(level=level)
        fname_outer_layer = fname_current + 'outer_layer.pickle'
        if os.path.exists(fname_current):  # load if already calculed
            self.g = nx.read_gpickle(fname_current)
            # with open('layer%d.pickle' % level, 'rb') as fl:
            with open(fname_outer_layer, 'rb') as fl:
                new_outer_layer = pickle.load(fl)
            # new_outer_layer = pickle.load(open(fname_outer_layer, 'rb'))
            return new_outer_layer
        # else:
        #     # start from previous
        #     fname_previous = self._get_file_path(level=level-1)
        #     fname_previous_outer_layer = fname_current + 'outer_layer.pickle'
        #     graph = nx.read_gpickle(fname_previous)
        #     new_outer_layer = pickle.load(fname_outer_layer)

        seen = set([x[0] for x in self.g.edges()])
        unvisited = outer_layer_ids - seen
        new_outer_layer = set()

        to_process = len(unvisited)
        print('Starting to run for level {}, {} users'.format(level, to_process))
        for u_id in unvisited:
            to_process -= 1
            percentage = 100 - int(to_process / len(unvisited) * 100)
            print('Avance: %{}'.format(percentage), end='\r')

            followed = self.get_followed_user_ids(u_id)
            followed = [f_id for f_id in followed if self._is_relevant_user(f_id)]
            self.g.add_edges_from([(u_id, f_id) for f_id in followed])

            new_nodes = [f_id for f_id in followed if self.g.out_degree(f_id) == 0]
            new_outer_layer.update(new_nodes)

        nx.write_gpickle(self.g, fname_current)
        with open(fname_outer_layer, 'wb') as fl:
            pickle.dump(new_outer_layer, fl)

        return new_outer_layer

    def get_followed_user_ids(self, user_id):

        print(user_id)
        if self.g.out_degree(user_id):
            followed = self.g.successors(user_id)
            print('fetched followed user_ids from loaded graph')
            return followed

        print('fetched from internet')
        retries = 0
        while True:
            try:
                # API_HANDLER.get_fresh_connection()
                followed = API_HANDLER.traer_seguidos(user_id=user_id)
                # GRAPH.add_edges_from([(user_id, f_id) for f_id in followed])
                return followed
            except TweepError as e:
                if e.response == 'Not authorized.':
                    NOTAUTHORIZED.add(user_id)
                    with open(NOTAUTHORIZED_FNAME, 'wb') as f:
                        pickle.dump(NOTAUTHORIZED, f)
                    return []
                else:
                    print("Error for user {}: {}".format(user_id, e.response))
                    retries += 1
                    if retries == 5:
                        print("Gave up retrying for user {}".format(user_id))
                        return []
                    else:
                        print("waiting...")
                        # time.sleep(10)

    def _is_relevant_user(self, user_id):
        user_id = str(user_id)
        if user_id in self.relevant_users.keys():
            return self.relevant_users[user_id]
        retries = 0
        while True:
            try:
                # API_HANDLER.get_fresh_connection()
                u = API_HANDLER.get_user(user_id=user_id)
                relevant = u.followers_count > 40 and u.friends_count > 40
                self.relevant_users[user_id] = relevant
                with open(self.relevant_filepath, 'w') as f:
                    json.dump(self.relevant_users, f)
                return relevant
            except TweepError as e:
                print(e)
                print("Error in is_relevant for %s" % user_id)
                retries += 1
                if retries == 5:
                    print("Gave up retrying for user %s" % user_id)
                    print("(marked as not relevant)")
                    return False
                else:
                    print("waiting...")
                    # time.sleep(10)

    def build_subgraph_k_degree_closure(self, K=50):
        """
            Partiendo de mi central_uid,
            voy agregando para cada usuario sus 50 seguidos más similares,
            incluyendo sólo usuarios relevantes ( >40 followers, >40 followed filtrados en build_graph_to_level)

            Creamos además un grafo auxiliar con los nodos ya visitados
            (útil para calcular relevancias y similaridades)
        """
        # try:
        #     graph = nx.read_gpickle('graph2.gpickle')
        # except IOError:
        #     pass
        self.subg = nx.DiGraph()

        visited = set([x for x in self.subg.nodes() if self.subg.out_degree(x)])

        # if self.subg.number_of_nodes():
        #     unvisited = set([x for x in self.subg.nodes() if self.subg.out_degree(x) == 0])
        # else:
        unvisited = [str(x) for x in self.g.nodes]

        # try:
        #     failed = set(json.load(open('failed.json')))
        # except IOError:
        failed = set()

        while unvisited:
            new_unvisited = set()
            for uid in unvisited:
                followed = self.get_followed_user_ids(user_id=uid)

                if followed is None:
                    failed.add(int(uid))
                    continue

                followed = followed  # All nodes in universe are assumed relevant
                scored = []
                for f in followed:
                    f_followed = self.get_followed_user_ids(user_id=f)
                    if f_followed is None:
                        failed.add(int(f))
                        continue

                    common = len(set(f_followed).intersection(set(followed)))
                    # print(type(followed), type(f_followed), end='\r')
                    # print(followed, f_followed)
                    total = len(list(followed)) + len(list(f_followed)) - common
                    score = common * 1.0 / total
                    scored.append((f, score))

                most_similar = sorted(scored, key=lambda u_s: -u_s[1])[:K]
                most_similar = [u for (u, s) in most_similar]

                self.subg.add_edges_from([(uid, f_id) for f_id in most_similar])
                print('added {} edges to graph'.format(len(most_similar)))
                nx.write_gpickle(self.subg, self._get_file_path_k_closure(self.uid))

                new_unvisited.update(most_similar)

                visited.add(uid)

            # import ipdb;ipdb.set_trace()
            new_unvisited = new_unvisited - visited
            unvisited = new_unvisited

            n_nodes = self.subg.number_of_nodes()
            n_edges = self.subg.number_of_edges()
            print("%d nodes, %d edges" % (n_nodes, n_edges))

            # save progress
            nx.write_gpickle(self.subg, self._get_file_path_k_closure(self.uid))

            # with open('failed.json', 'w') as f:
            #     json.dump(list(failed), f)

        return self.subg

    def update_edges(self):
        to_process = self.g.number_of_nodes()
        percentage = 0
        for u_id in self.g.nodes():

            # advance notification
            to_process -= 1
            percentage = 100 - int(to_process / self.g.number_of_nodes() * 100)
            print(
                'Avance: %{}; {}/{}\t\t\tTrayendo a {}'.format(percentage, to_process, self.g.number_of_nodes(), u_id),
                end='\r'
            )

            retries = 0
            fetched = False
            while retries <= 4 and not fetched:
                try:
                    follows = self.api.traer_seguidos(user_id=u_id)
                    fetched = True
                except TweepError as e:
                    if e.reason == 'Not authorized.':
                        NOTAUTHORIZED.add(u_id)
                        with open(NOTAUTHORIZED_FNAME, 'wb') as f:
                            pickle.dump(NOTAUTHORIZED, f)
                        print('Not authorized length: {}'.format(len(NOTAUTHORIZED)))
                        retries = 5
                    elif 'Sorry, that page does not exist.' in e.reason:
                        NOTFOUND.add(u_id)
                        with open(NOTFOUND_FNAME, 'wb') as f:
                            pickle.dump(NOTFOUND, f)
                        print('Not found length: {}'.format(len(NOTFOUND)))
                        retries = 5
                    else:
                        print(
                            "Error for user %d: %s" % (u_id, e.reason))
                        retries += 1
                        if retries == 5:
                            print(
                                "Gave up retrying for user %d" % u_id)
                        else:
                            print(
                                "waiting...",
                                time.sleep(10))


            # self._partial_save(graph=self.g)
            print('Fetched users: {}'.format(len(follows)))
            follows_filtered = set(follows).intersection(self.g.nodes())
            # print(follows_filtered)
            self.g.add_edges_from([(u_id, f_id) for f_id in follows_filtered])
        nx.write_gpickle(self.g, UPDATED_GAPH_PATH)

    def _partial_save(self, graph):
        if os.path.exists(TMP_GRAPH_PATH):
            with open(TMP_GRAPH_PATH, 'rb') as f:
                graph = nx.read_gpickle(self.g, UPDATED_GAPH_PATH)
        else:
            graph = nx.DiGraph()

    @staticmethod
    def build_graph():
        central_uid = CENTRAL_USER_DATA['id']
        graph = GraphHandler(NX_GRAPH_FOLDER, central_uid=central_uid)
        graph.build_graph_to_level()

    @staticmethod
    def build_k_closure_graph():
        central_uid = '137548920'
        graph = GraphHandler(NX_GRAPH_FOLDER, central_uid=central_uid)
        graph.build_subgraph_k_degree_closure()

    @staticmethod
    def build_k_closure_graph_2():
        central_uid = '137548920'
        graph = GraphHandler(NX_GRAPH_FOLDER, central_uid=central_uid)
        graph.build_k_closure_graph_from_scratch()

    def build_k_closure_graph_from_scratch(self, K=50):
        """
            Partiendo de mi usuario,
            voy agregando para cada usuario sus 50 seguidos más similares,
            incluyendo sólo usuarios relevantes ( >40 followers, >40 followed )

            Creamos además un grafo auxiliar con los nodos ya visitados
            (útil para calcular relevancias y similaridades)
        """
        # try:
        #     graph = nx.read_gpickle('graph2.gpickle')
        # except IOError:
        self.subg = nx.DiGraph()

        visited = set([x for x in self.subg.nodes() if self.subg.out_degree(x)])
        #
        # if graph.number_of_nodes():
        #     unvisited = set([x for x in graph.nodes() if graph.out_degree(x) == 0])
        # else:
        unvisited = [str(CENTRAL_USER_DATA['id'])]

        try:
            failed = set(json.load(open('failed.json')))
        except IOError:
            failed = set()

        while unvisited:
            new_unvisited = set()
            for uid in unvisited:
                followed = self.get_followed_user_ids(user_id=uid)

                if followed is None:
                    failed.add(int(uid))
                    continue

                # r_followed = [f for f in followed if is_relevant(f)]
                r_followed = followed  # All nodes in universe are assumed relevant
                scored = []
                for f in r_followed:
                    f_followed = self.get_followed_user_ids(user_id=f)
                    if f_followed is None:
                        failed.add(int(f))
                        continue

                    common = len(set(f_followed).intersection(set(followed)))
                    total = len(list(followed)) + len(list(f_followed)) - common
                    score = common * 1.0 / total
                    scored.append((f, score))

                most_similar = sorted(scored, key=lambda u_s: -u_s[1])[:K]
                most_similar = [u for (u, s) in most_similar]

                self.subg.add_edges_from([(uid, f_id) for f_id in most_similar])
                nx.write_gpickle(self.subg, 'graph2.gpickle')

                new_unvisited.update(most_similar)
                print('new_unvisited', new_unvisited)

                visited.add(uid)

            new_unvisited = new_unvisited - visited
            unvisited = new_unvisited

            n_nodes = self.subg.number_of_nodes()
            n_edges = self.subg.number_of_edges()
            print("%d nodes, %d edges" % (n_nodes, n_edges))

            # save progress
            nx.write_gpickle(self.subg, 'graph2.gpickle')

            # with open('failed.json', 'w') as f:
            #     json.dump(list(failed), f)

        return self.subg
