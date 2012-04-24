"""MI-ClusterRegress"""
import numpy as np
from math import log10

class MIClusterRegress(object):
    """Implements the MI-ClusterRegress algorithm described in:

    Kiri L. Wagstaff, Terran Lane, and Alex Roper. Multiple-Instance Regression
    with Structured Data. Proceedings of the 4th International Workshop on
    Mining Complex Data, December 2008.  
    """

    def __init__(self, cluster, regress, verbose=True):
        """
        MI-ClusterRegress

        @param cluster : an implemented Clusterer object
        @param regress : returns an instance of a RegressionModel object
        @param verbose : print output during training
        """
        self.cluster = cluster
        self.regress = regress
        self.regression_models = None
        self.selected_index = None
        self.verbose = verbose

    def _status_msg(self, message):
        """
        Print a status message if verbose output is enabled.

        @param message : message to print
        """
        if self.verbose: print message

    def _exemplars(self, bags):
        """
        Compute an array of bag exemplars w.r.t. each cluster.

        @param bags : sequence of array-like bags, each with 
                      shape [n_instances, n_features]
        @return : sequence of array exemplar sets, one for each
                  cluster with shape [n_bags, n_features]
        """

        # Sequence of matrices holding exemplars for each cluster
        cluster_examplars = [np.dot(self.cluster.relevance(bag), bag)
                             for bag in bags]

        # Convert to sequence of matrices holding bag exemplars per cluster
        bag_exemplars = map(np.vstack, zip(*cluster_examplars))
        return bag_exemplars

    def _select(self, bags, y):
        """
        Select a cluster/model index to use for predicting new bags
        (Override this method to implement different selection criteria.)

        @param bags : sequence of array-like bags, each with 
                      shape [n_instances, n_features]
        @param y : bag labels, array-like, shape [n_bags]
        @return : index of cluster/model to use for prediction
        """
        exemplars_models = zip(_exemplars(bags), self.regression_models)
        predictions = [model.predict(ex_set)
                       for ex_set, model in exemplars_models]
        rmses = [rmse(p, y) for p in predictions]
        return rmses.index(min(rmses))

    def fit(self, bags, y):
        """
        Fit the model according to the given training data

        @param bags : sequence of array-like bags, each with 
                      shape [n_instances, n_features]
        @param y : bag labels, array-like, shape [n_bags]
        @return : self
        """
        bags = map(np.asarray, bags)
        X = np.vstack(bags)

        self._status_msg('Clustering instances...')
        self.cluster.fit(X)

        self._status_msg('Computing exemplars...')
        exemplar_sets = self._exemplars(bags)

        self._status_msg('Computing regression models...')
        k = len(exemplar_sets)
        fstr = '    %%0%dd of %d...' % (int(log10(k)) + 1, k)
        self.regression_models = list()
        for i, ex_set in enumerate(exemplar_sets, 1):
            self._status_msg(fstr % i)
            model = self.regress()
            model.fit(ex_set, y)
            regression_models.append(model)

        self._status_msg('Selecting predictor...')
        self.selected_index = _select(bags, y)

        return self

    def predict(self, bags):
        """
        Apply fit regression function to each bag

        @param bags : sequence of array-like bags, each with 
                      shape [n_instances, n_features]
        @return y : array, shape [n_bags]
        """
        bags = map(np.asarray, bags)
        exemplars, model = zip(
            _exemplars(bags), self.regression_models)[self.selected_index]
        return model.predict(exemplars)

class Clusterer(object):
    """Interface for an object that can cluster data"""

    def fit(self, X):
        """
        Computes clusters of data X.

        @param X : array-like, shape [n_instances, n_features]
        """
        pass

    def relevance(self, bag):
        """
        Returns the relevance matrix, i.e. the relevance of the bag instances to
        each cluster. Relevance should be normalized so that the relevances of
        instances within each cluster (each row) sum to one.

        @param bag : array-like, shape [n_instances, n_features]
                     bag for which relevance should be computed
        @return : array, shape [n_clusters, n_instances]
        """
        pass

    @staticmethod
    def normalize(relevance):
        """
        A convenience method that ensures the relevance matrix is properly
        normalized.

        @param relevance : array, shape [n_clusters, n_instances]
        @return : normalized copy of relevance matrix
        """
        relevance = np.array(relevance)
        sums = np.sum(relevance, axis=1)
        for row, s in enumerate(sums):
            if s == 1:
                continue
            elif s == 0:
                n = relevance.shape[1]
                relevance[row, :] = 1.0 / n
            else:
                relevance[row, :] /= s

        return relevance

class RegressionModel(object):
    """Interface for an instance-based regression technique"""

    def fit(self, X, y):
        """
        Fit the model according to the given training data

        @param X : array-like, shape [n_instances, n_features]
        @param y : array-like, shape [n_instances]
        """
        pass

    def predict(self, X):
        """
        Apply fit regression function to each instance in X

        @param X : array-like, shape [n_instances, n_features]
        @return y : array, shape [n_instances]
        """
        pass
