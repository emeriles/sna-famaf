from __future__ import print_function

from os.path import join

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

from processing._1_user_model.db_csv import DatasetOneUserModel
from settings import MODELS_FOLDER_1_


class OneUserModel(object):

    @staticmethod
    def evaluate_model(clf, X_train, X_test, y_train, y_test):
        y_true, y_pred = y_train, clf.predict(X_train)

        print("Detailed classification report:\n")
        print("Scores on training set.\n")
        print(classification_report(y_true, y_pred))

        y_true, y_pred = y_test, clf.predict(X_test)
        print("Scores on test set.\n")
        print(classification_report(y_true, y_pred))

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
            print(classification_report(y_true, y_pred))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
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
            print(classification_report(y_true, y_pred))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
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
            print(classification_report(y_true, y_pred))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
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
            print(classification_report(y_true, y_pred))
            print()


            print("Scores on test set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
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
    def load_model(uid, model_type, time_delta_filter):
        model_path = join(MODELS_FOLDER_1_, "{}_{}_{}.pickle".format(model_type, uid, time_delta_filter))
        try:
            clf = joblib.load(model_path)
            print('LOADED MODEL FROM {model_path}'.format(model_path=model_path))
        except FileNotFoundError:
            return None
        return clf

    @staticmethod
    def save_model(clf, uid, model_type, time_delta_filter):
        model_path = join(MODELS_FOLDER_1_, "{}_{}_{}.pickle".format(model_type, uid, time_delta_filter))
        joblib.dump(clf, model_path)

    @staticmethod
    def load_or_build_model(uid, model_type, time_delta_filter, save=True):
        print('Load or build model. For {}, model type: {}, time_delta: {}'.format(uid, model_type, time_delta_filter))
        clf = OneUserModel.load_model(uid, model_type, time_delta_filter)
        dataset = DatasetOneUserModel.load_or_create_dataset(uid, time_delta_filter)
        X_train, X_test, X_valid, y_train, y_test, y_valid, X_train_l, X_test_l, X_valid_l = dataset
        dataset = X_train, X_test, y_train, y_test
        # dataset = X_valid, X_train, y_valid, y_train   ###### SAME AS THIS                      <<<<<<<<<<<<<<<<<<<<-----------------------------------
        if not clf:
            clf = getattr(OneUserModel, 'model_select_{}'.format(model_type))(dataset)
            if save:
                OneUserModel.save_model(clf, uid, model_type, time_delta_filter)
        return clf
