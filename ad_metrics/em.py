import numpy as np
from sklearn.metrics import auc
import statistics

random_state=42
np.random.seed(2)

class em_metric:
    def _init_(self, X,static_variable_obj):
        self.X = X
        self.static_variable_obj = static_variable_obj
        self.algorithm = static_variable_obj.algorithms_run
        self.cv = static_variable_obj.cv
        self.clf = static_variable_obj.clf
    def metric_trial(self):
        a_curve = []
        iterations_max = data_dim= self.X.shape[1]
        i = 1
        while i <= iterations_max:
            if data_dim >= 5:
                self.level_set_dim = 5
                features = np.random.choice(data_dim, self.level_set_dim, replace=False)
                i = i + 1
                x_selection = self.X[:, features]
            else:
                features = np.random.choice(data_dim, data_dim, replace=False)
                i = iterations_max + 1
                x_selection = self.X[:, features]
                self.level_set_dim = x_selection.shape[1]
            self.lim_inf = x_selection.min(axis=0)
            self.lim_sup = x_selection.max(axis=0)
            self.n_sim = self.n_generated = 10000
            self.U = np.random.uniform(self.lim_inf, self.lim_sup, size=(self.n_sim,self.level_set_dim))
            self.vol_tot_cube = (self.lim_sup - self.lim_inf).prod()
            
            for train, test in self.cv.split(x_selection):
                x_train = x_selection[train]
                x_test = x_selection[test]
                clf_train = self.clf.fit(x_train)
                score_U = np.nan_to_num(clf_train.decision_function(self.U))
                score_test = np.nan_to_num(clf_train.decision_function(x_test))
                t_max = 0.9
                t = np.arange(0, 100 / self.vol_tot_cube, 0.01 / self.vol_tot_cube)
                EM_t = np.zeros(t.shape[0])
                n_samples = score_test.shape[0]
                s_X_unique = np.unique(score_test)
                EM_t[0] = 1.
                for u in s_X_unique:
                    EM_t = np.maximum(EM_t, 1. / n_samples * (score_test < u).sum() -
                                      t * (score_U < u).sum() / self.n_generated
                                      * self.vol_tot_cube)
                amax = np.argmax(EM_t <= t_max) + 1
                if amax == 1:
                    amax = -1
                auemc = auc(t[:amax], EM_t[:amax])
                a_curve.append(auemc)
        med_a_curve = statistics.median(a_curve)
        return med_a_curve