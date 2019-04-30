import os
import time

import networkx as nx
import pickle

from tweepy import TweepError

from twitter_api import API_HANDLER



GRAPH_PATH = './graphs/subgraph.gpickle'
UPDATED_GAPH_PATH = './graphs/subgraph_new.gpickle'
TMP_GRAPH_PATH = './temp/subgraph_new_tmp.gpickle'
TMP_SEEN_USERS = './temp/seen_users.pickle'

if os.path.exists(TMP_SEEN_USERS):
    with open(TMP_SEEN_USERS, 'rb') as f:
        SEEN_USERS = pickle.load(f)
else:
    SEEN_USERS = set()

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


class GraphUpdate(object):
    """Networkx graph handling based on other existing graph, nodes will be reused!"""
    def __init__(self, fpath):
        self.api = API_HANDLER
        if os.path.exists(fpath):
            old_graph = nx.read_gpickle(fpath)
            self.g = nx.DiGraph()
            self.g.add_nodes_from(old_graph.nodes())
        else:
            raise Exception('No pre-existing graph found!')

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
    #
    # def _partial_save(self, graph):
    #     if os.path.exists(TMP_GRAPH_PATH):
    #         with open(TMP_GRAPH_PATH, 'rb') as f:
    #             graph = nx.read_gpickle(self.g, UPDATED_GAPH_PATH)
    #     else:
    #         graph = nx.DiGraph()


if __name__ == '__main__':
    graph = GraphUpdate(GRAPH_PATH)
    graph.update_edges()

#
#
# def get_followed_user_ids(user=None, user_id=None):
#     if user is not None:
#         user_id = user.id
#
#     if GRAPH.out_degree(user_id):
#         followed = GRAPH.successors(user_id)
#         return followed
#     else:
#         retries = 0
#         while True:
#
#             try:
#                 TW = API_HANDLER.get_connection()
#                 followed = TW.friends_ids(user_id=user_id)
#                 GRAPH.add_edges_from([(user_id, f_id) for f_id in followed])
#                 return followed
#             except Exception as e:
#                 # print e
#                 if e.message == u'Not authorized.':
#                     NOTAUTHORIZED.add(user_id)
#                     with open(NOTAUTHORIZED_FNAME, 'wb') as f:
#                         pickle.dump(NOTAUTHORIZED, f)
#                     return []
#                 else:
#                     print
#                     "Error for user %d: %s" % (user_id, e.message)
#                     retries += 1
#                     if retries == 5:
#                         print
#                         "Gave up retrying for user %d" % user_id
#                         return []
#                     else:
#                         print
#                         "waiting..."
#                         time.sleep(10)
#
#
# def get_friends_graph():
#     my_id = USER_DATA['id']
#
#     # Seed: users I'm following
#     my_followed = get_followed_user_ids(my_id)
#
#     fname = 'graph.gpickle'
#     if os.path.exists(fname):  # resume
#         graph = nx.read_gpickle(fname)
#     else:
#         graph = nx.DiGraph()
#
#     seen = set([x[0] for x in graph.edges()])
#
#     unvisited = list(set(my_followed) - seen)
#     for u_id in unvisited:
#         followed = get_followed_user_ids(u_id)
#         nx.write_gpickle(graph, fname)
#
#     return graph
#
#
# if __name__ == '__main__':
#     compute_extended_graphs()
#     # graph = get_friends_graph()
#
#     # fname = 'graph.gpickle'
#     # graph = nx.read_gpickle(fname)
#
#     # for uid in graph.nodes():
#     #     fetch_timeline(user_id=uid)
#     # import matplotlib.pyplot as plt
#     # nx.draw(graph)
#     # plt.show()
#
#     # user_ids = graph.nodes()
#
#     # for u_id in user_ids:
#     #     fetch_timeline(screen_name=None, user_id=None, days=30)
#
#     # Among those, I collect all the following relationships within the set
