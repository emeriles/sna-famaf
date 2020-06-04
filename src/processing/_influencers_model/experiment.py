import pickle

from processing._influencers_model.db_csv import DatasetInfluencersModel
from processing._influencers_model.influence import InfluenceActions
from settings import MIN_INFLUENCERS, MAX_INFLUENCERS, STEP_INFLUENCERS, AVG_RANDOM_REPETITIONS_NEEDED, \
    INFLUENCE_POINTS, XY_CACHE_FOLDER, XY_CACHE_FOLDER_FT


class Experiments(object):

    @staticmethod
    def _experiment(d, influence_points, with_influencers=True, communities_distributed=False, folder=XY_CACHE_FOLDER, full=False,
                    fasttext=False):

        if fasttext:
            folder = XY_CACHE_FOLDER_FT
        experiments = list()
        experiments.append('social')
        # if full:
        #     experiments.append('tw_lda_60')
        #     experiments.append('fasttext')
        #     experiments.append('random')
        #     experiments.append('lda_20')
        # if fasttext:
        #     experiments = ['fasttext']
        d.load_tweets_filtered()
        # d.load_tw_lda(num_topics=60)
        for experiment in experiments:
            for number in range(MIN_INFLUENCERS, MAX_INFLUENCERS, STEP_INFLUENCERS):
                need_to_repeat = True
                repetitions = 0
                while need_to_repeat:
                    need_to_repeat = repetitions < AVG_RANDOM_REPETITIONS_NEEDED and "random" in experiment
                    if experiment != "random":
                        if communities_distributed:
                            d.get_influencers_id_list_by_community(influence_points,
                                                                   number)
                        else:
                            d.get_influencers_id_list(number_of_influencers=number)
                    else:
                        # d.get_random_id_list(influence_points,
                        #                     number_of_influencers=number)
                        d.get_influencers_id_list(number_of_influencers=number)
                        # d.get_random_from_influencers(influence_points,
                        #                              number_of_influencers=number)
                    if 'lda' in experiment and not 'tw' in experiment:
                        d.load_lda(num_topics=experiment.split("lda_")[-1],
                                   save_file=experiment + ".pickle")
                    # if 'lda' in experiment and 'tw' in experiment:
                    #    d.load_tw_lda(num_topics=experiment.split("tw_lda_")[-1])
                    # if 'fasttext' in experiment:
                    #     pass
                    #     d.load_fasttext()
                    time_window = d.delta_minutes
                    suffix = "_{}i_{}_{}m".format(number, experiment, time_window)
                    if repetitions != 0:
                        suffix += "_{}".format(str(repetitions))
                    x_train, y_train = d.extract_features(dataset="train",
                                                          with_influencers=with_influencers, fasttext=fasttext)
                    with open('{}/_infl_X_train{}.pickle'.format(folder, suffix), 'wb') as save_file:
                        pickle.dump(x_train, save_file)
                    with open('{}/_infl_y_train{}.pickle'.format(folder, suffix), 'wb') as save_file:
                        pickle.dump(y_train, save_file)
                    x_test, y_test = d.extract_features(dataset="test",
                                                        with_influencers=with_influencers, fasttext=fasttext)
                    with open('{}/_infl_X_test{}.pickle'.format(folder, suffix), 'wb') as save_file:
                        pickle.dump(x_test, save_file)
                    with open('{}/_infl_y_test{}.pickle'.format(folder, suffix), 'wb') as save_file:
                        pickle.dump(y_test, save_file)
                    repetitions += 1

    @staticmethod
    def experiment(delta_minutes, fasttext=False):
        import sys
        import os
        # FOLDER = sys.argv[1]
        # os.mkdir(FOLDER)
        d = DatasetInfluencersModel(delta_minutes_filter=delta_minutes, fasttext=fasttext)
        # INFLUENCE_FILE = "influence_points_{}.pickle".format('50_10_40')
        # INFLUENCE_FILE = "influence_points_{}.pickle".format('new')
        # INFLUENCE_FILE = "influence_points_{}.pickle".format('nx_35_15_25_25_infomap')
        # INFLUENCE_FILE = "influence_points_{}.pickle".format('nx_50_10_40_0_infomap')
        influence_points = InfluenceActions.load_influencers_from_pickle(INFLUENCE_POINTS)
        Experiments._experiment(d,
                                influence_points,
                                with_influencers=True,
                                communities_distributed=False,
                                folder=XY_CACHE_FOLDER,
                                full=False,
                                fasttext=fasttext)
        # only_one_community()
