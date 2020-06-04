import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do all kinds of stuff with the project')

    action_choices = [
        'test_all_clfs',
        'try_some_users',
        'try_some_users_cp',
        'compute_scores',
        'compute_scores_ft',
        'compute_scores_cp',
        'create_and_save_csv_cutted',
        # 'reset_sqlite_db',
        'active_and_central',
        'build_users_graph',
        'build_k_degree_subgraph',
        'build_k_degree_subgraph_2',
        'save_graph_as_graphml',
        'build_influence_points',
        'get_users_followed_data',
        'experiment_influencers',
        'evaluate_influencers',
        'develop',
        'build_ftext_features',
        'cross_prediction',
    ]
    parser.add_argument('action', metavar='ACTION', type=str,
                        help='action to be performed. One of {}'.format(action_choices), choices=action_choices)
    data_choices = ['small', 'full']
    parser.add_argument('--data', metavar='DATA', type=str,
                        help='data to be loaded. Small sample or full', choices=data_choices)
    parser.add_argument('--seconds', required=False, action='store_true',
                        help='If present, it says to take delta_minutes as seconds (just for compute_scores[_ft] and experiment_influencers[_ft]')
    delta_minutes_choices = [0, 75, 150, 225, 500]  # quartiles to 5 hours
    parser.add_argument('delta_minutes', metavar='DELTA_MINUTES', type=int,
                        help='delta minutes to consider, zero means use the COMPLETE dataset')
                        # choices=delta_minutes_choices)

    args = parser.parse_args()

    delta_minutes = args.delta_minutes if args.delta_minutes != 0 else None
    as_seconds = args.seconds
    print('delta minutes as seconds? : {}'.format(as_seconds))

    if args.data == 'full':
        os.environ.setdefault('DATASET_SIZE_TYPE', 'FULL')
    else:
        os.environ.setdefault('DATASET_SIZE_TYPE', 'SMALL')

    from modeling._1_one_user_learn_neighbours.compute_scores import compute_scores
    from modeling._1_one_user_learn_neighbours.model import OneUserModel
    from modeling._1_one_user_learn_neighbours.try_some_users import try_some_users
    from processing.preprocess_csv import PreprocessCSV
    # from processing.dbmodels import reset_sqlite_db
    from preparation.get_active_and_central import ActiveAndCentral
    from preparation.twitter_users import GraphHandler
    from processing._influencers_model.influence import InfluenceActions
    from processing._influencers_model.experiment import Experiments
    from processing._influencers_model.evaluate import evaluate
    from preparation.fasttext_integration import FTextActions

    print('RUNNING {}'.format(args.action))
    if args.action == 'test_all_clfs':
        OneUserModel.test_all_clfs(uid=76133133, time_delta_filter=delta_minutes)
    if args.action == 'try_some_users':
        try_some_users(delta_minutes=delta_minutes)
    if args.action == 'try_some_users_cp':
        try_some_users(delta_minutes=delta_minutes, cherry_pick_users=True)
    if args.action == 'compute_scores':
        compute_scores(delta_minutes=delta_minutes, as_seconds=as_seconds)
    if args.action == 'compute_scores_ft':
        compute_scores(delta_minutes=delta_minutes, fasttext=True, as_seconds=as_seconds)
    if args.action == 'compute_scores_cp':
        compute_scores(delta_minutes=delta_minutes, cherry_pick_users=True)

    if args.action == 'create_and_save_csv_cutted':
        PreprocessCSV.create_and_save_csv_cutted()
    # if args.action == 'reset_sqlite_db':
    #     reset_sqlite_db()
    if args.action == 'active_and_central':
        ActiveAndCentral.get_most_central_and_active()
    if args.action == 'build_users_graph':
        GraphHandler.build_graph()
    if args.action == 'build_k_degree_subgraph':
        GraphHandler.build_k_closure_graph()
    if args.action == 'build_k_degree_subgraph_2':
        GraphHandler.build_k_closure_graph_2()
    if args.action == 'save_graph_as_graphml':
        InfluenceActions.save_graph_as_graphml()

    if args.action == 'build_influence_points':
        InfluenceActions.build_influence_points()
    if args.action == 'get_users_followed_data':
        GraphHandler.get_users_followed_data()

    if args.action == 'experiment_influencers':
        Experiments.experiment(delta_minutes=delta_minutes)
    if args.action == 'evaluate_influencers':
        evaluate(delta_minutes=delta_minutes)

    if args.action == 'build_ftext_features':
        FTextActions.build_ftext_features()

    if args.action == 'cross_prediction':
        OneUserModel.cross_prediction()
