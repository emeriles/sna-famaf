from __future__ import print_function

from os.path import join

from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

from processing._1_user_model.db_csv import DatasetOneUserModel
from processing.utils import get_test_users_ids
from settings import MODELS_FOLDER_1_


class OneUserModel(object):

    @staticmethod
    def evaluate_model(clf, X_train, X_test, y_train, y_test):
        y_true, y_pred = y_train, clf.predict(X_train)

        print("Detailed classification report:\n")
        print("Scores on training set.\n")
        print(classification_report(y_true, y_pred, digits=4))

        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred, digits=4))

    # @staticmethod
    # def sub_sample_negs_arr(X, y):
    #     npos = int(sum(y))
    #     neg_inds = [i for i in range(len(y)) if y[i] == 0]
    #     pos_inds = [i for i in range(len(y)) if y[i]]
    #     sample_neg_inds = sample(neg_inds, npos)
    #     inds = pos_inds + sample_neg_inds
    #
    #     Xs = X[inds,:]
    #     ys = y[inds]
    #
    #     return Xs, ys

    @staticmethod
    def model_select_rdf(dataset, cv=3, n_jobs=6):
        print('\t\t\tSTART RDF')
        X_train, X_test, y_train, y_test = dataset

        w1 = sum(y_train)/len(y_train)
        w0 = 1 - w1
        sample_weight = np.array([w0 if x==0 else w1 for x in y_train])

        # Set the parameters by cross-validation
        params = dict(
            max_depth=[5, 20, None],
            n_estimators=[10, 30, 100, 500],
            class_weight=['balanced_subsample', 'balanced', None],
            # sample_weight=[sample_weight, None],
            max_features=[50, 300, None, 'auto'],
            min_samples_leaf=[1, 3]
        )

        scores = [
            # 'recall',
            'f1',
            # 'precision',
        ]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                RandomForestClassifier(),
                param_grid=params,  # parameters to tune via cross validation
                refit=True,  # fit using all data, on the best detected classifier
                n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
                scoring=score,  # what score are we optimizing?
                cv=cv,  # what type of cross validation to use
            )

            clf.fit(X_train, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_params_)

            print("Detailed classification report:")
            print()
            print("Scores on training set.")
            y_true, y_pred = y_train, clf.predict(X_train)
            print(classification_report(y_true, y_pred, digits=4))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=4))
            print()

        print('\t\t\tEND RDF')
        return clf

    @staticmethod
    def model_select_svc(dataset, cv=3, n_jobs=6, max_iter=-1):
        print('\t\t\tSTART SVC')
        X_train, X_test, y_train, y_test = dataset

        # Set the parameters by cross-validation
        parameters = [
            {
             'kernel': ['rbf', 'poly'],
             'gamma': [0.1, 1, 10, 100],
             'C': [0.01, 0.1, 1],
             'class_weight': ['balanced', None]
            }
        ]

        scores = [
            # 'precision',
            # 'recall',
            'f1'
        ]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print('X_train.shape, X_test.shape, True in y_train, True in y_test :')
            print(X_train.shape, X_test.shape, True in y_train, True in y_test)

            clf = GridSearchCV(
                SVC(max_iter=max_iter),
                param_grid=parameters,  # parameters to tune via cross validation
                refit=True,  # fit using all data, on the best detected classifier
                n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
                scoring=score,  # what score are we optimizing?
                cv=cv,  # what type of cross validation to use
            )

            clf.fit(X_train, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_params_)

            print("Detailed classification report:")
            print()
            print("Scores on training set.")
            y_true, y_pred = y_train, clf.predict(X_train)
            print(classification_report(y_true, y_pred, digits=4))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=4))
            print()

        print('\t\t\tEND SVC')
        return clf

    @staticmethod
    def model_select_svc2(dataset, cv=3, n_jobs=6):
        print('\t\t\tSTART SVC2')
        # Parameter grid es subconjunto de la de
        # model_select_svc, con kernel y gamma limitados
        # a los valores que siempre funcionaban mejor
        X_train, X_test, y_train, y_test = dataset

        # Set the parameters by cross-validation
        parameters = [
            {
             'kernel': ['rbf'],
             'gamma': [0.1],
             'C': [0.01, 0.1, 1],
             'class_weight': ['balanced', None]
            }
        ]

        scores = [
            # 'precision',
            # 'recall',
            'f1'
        ]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                SVC(),
                param_grid=parameters,  # parameters to tune via cross validation
                refit=True,  # fit using all data, on the best detected classifier
                n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
                scoring=score,  # what score are we optimizing?
                cv=cv,  # what type of cross validation to use
            )

            clf.fit(X_train, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_params_)

            print("Detailed classification report:")
            print()
            print("Scores on training set.")
            y_true, y_pred = y_train, clf.predict(X_train)
            print(classification_report(y_true, y_pred, digits=4))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=4))
            print()

        print('\t\t\tEND SVC 2')
        return clf

    @staticmethod
    def model_select_sgd(dataset, cv=3, n_jobs=6):
        print('\t\t\tSTART SGD')
        X_train, X_test, y_train, y_test = dataset

        # Set the parameters by cross-validation
        parameters = [
            {
                'alpha': (0.01, 0.001, 0.00001),
                'penalty': ('l1', 'l2', 'elasticnet'),
                'loss': ('hinge', 'log'),
                'n_iter': (10, 50, 80),
            }
        ]

        scores = [
            # 'precision',
            'recall',
            # 'f1'
        ]

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                SGDClassifier(),
                param_grid=parameters,  # parameters to tune via cross validation
                refit=True,  # fit using all data, on the best detected classifier
                n_jobs=n_jobs,  # number of cores to use for parallelization; -1 for "all cores"
                scoring=score,  # what score are we optimizing?
                cv=cv,  # what type of cross validation to use
            )

            clf.fit(X_train, y_train)

            print("Best parameters set found on training set:")
            print()
            print(clf.best_params_)

            print("Detailed classification report:")
            print()
            print("Scores on training set.")
            y_true, y_pred = y_train, clf.predict(X_train)
            print(classification_report(y_true, y_pred, digits=4))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred, digits=4))
            print()

        print('\t\t\tEND SGD')
        return clf

    @staticmethod
    def test_all_clfs(uid, time_delta_filter, save=True):
        # from create_clesa_datasets import *
        # uid=37226353
        X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l = DatasetOneUserModel.\
                                                    load_or_create_dataset(uid, delta_minutes_filter=time_delta_filter)
        dataset = X_valid, X_train, y_valid, y_train           ## THIS SAME AS  ... mirar mas abajo.
        clf1 = OneUserModel.model_select_rdf(dataset)
        clf2 = OneUserModel.model_select_svc(dataset)
        clf3 = OneUserModel.model_select_svc2(dataset)
        clf4 = OneUserModel.model_select_sgd(dataset)
        if save:
            OneUserModel.save_model(clf1, uid, 'rdf', time_delta_filter)
            OneUserModel.save_model(clf2, uid, 'svc', time_delta_filter)
            OneUserModel.save_model(clf3, uid, 'svc2', time_delta_filter)
            OneUserModel.save_model(clf4, uid, 'sgd', time_delta_filter)

    @staticmethod
    def load_model(uid, model_type, time_delta_filter, as_seconds=False, fasttext=False):
        time_delta_filter = str(time_delta_filter) + 'secs' if as_seconds else str(time_delta_filter)
        ft = '_ft' if fasttext else ''
        filename = "{}_{}_{}{}.pickle".format(model_type, uid, time_delta_filter, ft)
        model_path = join(MODELS_FOLDER_1_, filename)
        try:
            clf = joblib.load(model_path)
            print('LOADED MODEL FROM {model_path}'.format(model_path=model_path))
        except FileNotFoundError:
            return None
        return clf

    @staticmethod
    def save_model(clf, uid, model_type, time_delta_filter, as_seconds=False, fasttext=False):
        time_delta_filter = str(time_delta_filter) + 'secs' if as_seconds else str(time_delta_filter)
        ft = '_ft' if fasttext else ''
        filename = "{}_{}_{}{}.pickle".format(model_type, uid, time_delta_filter, ft)
        model_path = join(MODELS_FOLDER_1_, filename)
        joblib.dump(clf, model_path)

    @staticmethod
    def load_or_build_model(uid, model_type, time_delta_filter, save=True, as_seconds=False, fasttext=False):
        print('Load or build model. For {}, model type: {}, time_delta: {}'.format(uid, model_type, time_delta_filter))
        clf = OneUserModel.load_model(uid, model_type, time_delta_filter, as_seconds=as_seconds, fasttext=fasttext)
        dataset = DatasetOneUserModel.load_or_create_dataset(uid, time_delta_filter, fasttext=fasttext, as_seconds=as_seconds)
        X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l = dataset
        dataset = X_train, X_test, y_train, y_test
        # dataset = X_valid, X_train, y_valid, y_train   ###### SAME AS THIS                      <<<<<<<<<<<<<<<<<<<<-----------------------------------
        if not clf:
            clf = getattr(OneUserModel, 'model_select_{}'.format(model_type))(dataset)
            if save:
                OneUserModel.save_model(clf, uid, model_type, time_delta_filter, fasttext=fasttext, as_seconds=as_seconds)
        return clf

    @staticmethod
    def _cross_pred(uid, time_clf, time_to_pred, d):
        clf = OneUserModel.load_or_build_model(uid, 'svc', time_clf)
        # this is just legacy treatment! dataset now comes with labels!
        X_train, X_valid, X_testv, y_train, y_valid, y_testv = DatasetOneUserModel.\
            load_or_create_dataset(uid,delta_minutes_filter=time_to_pred)

        result = {}
        y_true, y_pred = y_train, clf.predict(X_train)
        print('Classification report on TRAIN with clf {} predicting on {}'.format(time_clf, time_to_pred))
        print(classification_report(y_true, y_pred, digits=4))
        result['f1s_train'] = f1_score(y_true, y_pred)
        result['precisions_train'] = precision_score(y_true, y_pred)
        result['recalls_train'] = recall_score(y_true, y_pred)
        result['pos_cases_train'] = int(np.sum(y_true))

        # y_true, y_pred = y_valid, clf.predict(X_valid)
        # lock.acquire()
        # f1s_valid[uid] = f1_score(y_true, y_pred)
        # precisions_valid[uid] = precision_score(y_true, y_pred)
        # recalls_valid[uid] = recall_score(y_true, y_pred)
        # pos_cases_valid[uid] = int(np.sum(y_true))
        # lock.release()

        y_true, y_pred = y_testv, clf.predict(X_testv)
        print('Classification report on TEST with clf {} predicting on {}'.format(time_clf, time_to_pred))
        result['f1s_testv'] = f1_score(y_true, y_pred)
        result['precisions_testv'] = precision_score(y_true, y_pred)
        result['recalls_testv'] = recall_score(y_true, y_pred)
        result['pos_cases_testv'] = int(np.sum(y_true))
        d[uid] = result

    @staticmethod
    def cross_prediction():
        time_delta_clf = 2
        time_delta_pred = None  # equals to hole dataset
        uids = get_test_users_ids()
        d = {}
        for uid in uids:
            time_delta_pred = None
            try:
                OneUserModel._cross_pred(uid, time_delta_clf, time_delta_pred, d)
            except ValueError:
                print('\t\tValue error for user {}.'.format(uid))
            # test here for dataset health
            time_delta_pred = 0
            OneUserModel.check_y_vector_health(uid, time_delta_clf, time_delta_pred)
        import pickle
        with open('scores_2_mins_predicting', 'wb') as f:
            pickle.dump(d, f)
        print('There were {} users. There are {} predictions'.format(len(uids), len(d)))
        print('Success!')

    @staticmethod
    def check_y_vector_health(uid, time_delta_1, time_delta_2):  # maybe extend to X?
        assert(time_delta_2 is None or time_delta_1 < time_delta_2)  # this is to check X. -> 0s could turn to 1s. but 1s can not turn to 0s

        def check_equals(arr1, arr2):
            for m1, m2 in zip(arr1, arr2):
                if issparse(m1):
                    m1, m2 = m1.todense(), m2.todense()
                assert (np.array_equal(m1, m2))

        print('Checking health on datasets')
        dataset_1 = DatasetOneUserModel. \
            load_or_create_dataset(uid, delta_minutes_filter=time_delta_1)
        dataset_2 = DatasetOneUserModel. \
            load_or_create_dataset(uid, delta_minutes_filter=time_delta_2)

        # each dataset unpacks to this:
        # X_train, X_valid, X_testv, y_train, y_valid, y_testv, X_train_l, X_test_l, X_valid_l

        print('Checking healht on Ys')
        ys_1, ys_2 = dataset_1[3:6], dataset_2[3:6]
        check_equals(ys_1, ys_2)
        print('Good health on targets (ys)')

        print('Checking healht on labels')
        lbs_1, lbs_2 = dataset_1[6:9], dataset_2[6:9]
        check_equals(lbs_1, lbs_2)
        print('Good health on labels')
