import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do all kinds of stuff with the project')

    action_choices = [
        'test_all_clfs',
        'try_some_users',
        'compute_scores',
        'create_and_save_csv_cutted',
        'reset_sqlite_db'
    ]
    parser.add_argument('action', metavar='ACTION', type=str,
                        help='action to be performed. One of {}'.format(action_choices), choices=action_choices)
    data_choices = ['small', 'full']
    parser.add_argument('--data', metavar='DATA', type=str,
                        help='data to be loaded. Small sample or full', choices=data_choices)
    delta_minutes_choices = [0, 75, 150, 225, 500]  # quartiles to 5 hours
    parser.add_argument('delta_minutes', metavar='DELTA_MINUTES', type=int,
                        help='delta minutes to consider, zero means use the COMPLETE dataset',
                        choices=delta_minutes_choices)

    args = parser.parse_args()

    delta_minutes = args.delta_minutes if args.delta_minutes != 0 else None

    if args.data == 'full':
        os.environ.setdefault('DATASET_SIZE_TYPE', 'FULL')
    else:
        os.environ.setdefault('DATASET_SIZE_TYPE', 'SMALL')

    from modeling._1_one_user_learn_neighbours.compute_scores import compute_scores
    from modeling._1_one_user_learn_neighbours.model import OneUserModel
    from modeling._1_one_user_learn_neighbours.try_some_users import try_some_users
    from processing.preprocess_csv import PreprocessCSV
    from processing.dbmodels import reset_sqlite_db

    print('RUNNING {}'.format(args.action))
    if args.action == 'test_all_clfs':
        OneUserModel.test_all_clfs(uid=42976687)
    if args.action == 'try_some_users':
        try_some_users()
    if args.action == 'compute_scores':
        compute_scores(delta_minutes=delta_minutes)
    if args.action == 'create_and_save_csv_cutted':
        PreprocessCSV.create_and_save_csv_cutted()
    if args.action == 'reset_sqlite_db':
        reset_sqlite_db()
