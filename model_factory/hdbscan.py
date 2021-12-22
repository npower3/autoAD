import hdbscan
from .base import Base

class hdbscan_model(Base):
    """Clustering & Density Based HDBSCAN 
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
        self.clusterer = hdbscan.HDBSCAN(**self.params).fit(X)
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
        self.scores = hdbscan.approximate_predict_scores(self.clusterer, X)
        return self.scores