import sys
import datetime

import networkx as nx
import time

from db import DBHandler

GRAPH_PATH = './graphs/subgraph.gpickle'

class Action(object):

    @staticmethod
    def move_graph_users_to_db(graph_path=GRAPH_PATH):
        start_time = time.time()

        graph = nx.read_gpickle(graph_path)
        u_ids = list(graph.nodes())
        db_handler = DBHandler()
        db_handler.save_users_ids(u_ids)

        elapsed_time = time.time() - start_time
        print("Done moving graph users. Took %.1f secs to save %d users" % (elapsed_time, len(u_ids)))

        return 0

    @staticmethod
    def get_and_save_tweets_for_all_users(dia=None):
        db_handler = DBHandler()
        db_handler.save_users_timelines(dia=dia)

    @staticmethod
    def get_and_save_today_tweets_for_all_users():
        db_handler = DBHandler()
        db_handler.save_users_timelines(dia=datetime.date.today())

    @staticmethod
    def get_and_save_yesterdays_tweets_for_all_users():
        db_handler = DBHandler()
        day = datetime.date.today() - datetime.timedelta(days=1)
        db_handler.save_users_timelines(dia=day)


if __name__ == '__main__':
    # Action.move_graph_users_to_db() # just for empty db. users will be consumed from db.
    # Action.get_and_save_tweets_for_all_users()
    args = sys.argv
    if len(args) > 1:
        if args[1] == 'get_today':
            Action.get_and_save_today_tweets_for_all_users()
        if args[1] == 'get_yesterday':
            Action.get_and_save_yesterdays_tweets_for_all_users()

