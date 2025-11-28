import time
import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MissForestImputer:
    """
    随机森林填补方法
    """

    def __init__(self, n_estimators=10, max_iter=3, max_features='sqrt', seed=42):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_features = max_features
        self.seed = seed

    def impute(self, incomp_data, verbose=True):
        """
        执行随机森林填补
        """
        recov = np.copy(incomp_data)
        m_mask = np.isnan(incomp_data)

        if verbose:
            print(f"(IMPUTATION) MISS FOREST\n\tMatrix: {incomp_data.shape}\n\tn_estimators: {self.n_estimators}")

        # Convert numpy array to pandas DataFrame if needed
        if isinstance(incomp_data, np.ndarray):
            incomp_data = pd.DataFrame(incomp_data)

        # Define custom estimators
        clf = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features,
                                     random_state=self.seed)
        rgr = RandomForestRegressor(n_estimators=self.n_estimators, max_features=self.max_features,
                                    random_state=self.seed)

        # Initialize MissForest
        mf_imputer = MissForest(clf=clf, rgr=rgr, max_iter=self.max_iter)
        recov_data = mf_imputer.fit_transform(incomp_data)
        recov_data = np.array(recov_data)

        recov[m_mask] = recov_data[m_mask]
        return recov


# 为了向后兼容，保留函数
def miss_forest_impute(incomp_data, n_estimators=10, max_iter=3, max_features='sqrt', seed=42, logs=True, verbose=True):
    imputer = MissForestImputer(n_estimators, max_iter, max_features, seed)
    return imputer.impute(incomp_data, verbose)