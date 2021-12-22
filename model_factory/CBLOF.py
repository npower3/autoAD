
from pyod.models.cblof import CBLOF
from .base import Base

class CBLOF_model(Base):
    """Clustering Based Local Outlier Factor (CBLOF)
    """
    def __init__(self, params=None):
        self.params = params

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
        self.clusterer = CBLOF(**self.params).fit(X)
        return self

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
        self.scores = CBLOF.decision_function(self.clusterer, X)
        return self.scores