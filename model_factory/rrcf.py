import rrcf
import numpy as np
from .base import Base

class rrcf_model(Base):
    """Tree based outlier algorithm (RRCF)
    """
    def __init__(self, params=None):
        self.tree_size = tree_size
        self.num_trees = num_trees
        self.forest = []

    def fit(self, X=None):

        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        n,d = X.shape
        while len(self.forest) < self.num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n, size=(n // self.tree_size, self.tree_size), replace=False)
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            self.forest.extend(trees)
#             print(len(forest))

    def decision_function(self, X=None):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        test_codisp = []
        for ix, point in enumerate(X):
        #     print('data index {}, data point {}'.format(ix, point))
            index = str(ix)+'_test'
            ix_codisp = []
            for tree in self.forest:
                tree.insert_point(np.array(point), index = index)
                ix_codisp.append(tree.codisp(index))
                tree.forget_point(index)
            test_codisp.append((sum(ix_codisp)/len(ix_codisp)).round(2))
        return np.array(test_codisp)
