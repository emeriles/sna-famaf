import pickle

from processing.db_csv import Dataset
from processing.utils import load_nx_graph
import networkx as nx


class ActiveAndCentral(object):

    @staticmethod
    def get_most_central_users(N=1000):
        g = load_nx_graph()
        # g = gt.load_graph(GT_GRAPH_PATH)
        katzc = nx.katz_centrality_numpy(g)
        # katzc = gt.katz(g)
        # katzc_array = katzc.get_array()
        # katzc_sorted = sorted(enumerate(katzc_array), key=lambda v: v[1])
        katzc_sorted = sorted(katzc.items(), key=lambda x: x[1])
        # most_central = [id for (id, c) in katzc_sorted][:N]
        # most_central_twids = [get_twitter_id(g,id) for id in most_central]
        most_central_twids = [k for k, v in katzc_sorted][:N]

        return most_central_twids

    @staticmethod
    def get_most_active_users(N=1000, just_ids=True):
        return Dataset.get_most_active_users(N=N, just_ids=just_ids)

    @staticmethod
    def get_most_central_and_active():
        most_central_ids = ActiveAndCentral.get_most_central_users()
        most_active_ids = ActiveAndCentral.get_most_active_users()
        active_and_central_ids = list(set(most_active_ids).intersection(set(most_central_ids)))
        print('Got {} from most active and central intersection.'.format(len(active_and_central_ids)))
        pickle.dump(active_and_central_ids, open('active_and_central.pickle', 'w'))
        return
