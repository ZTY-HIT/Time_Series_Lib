import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostImputer:
    """
    XGBoost填补方法
    """

    def __init__(self, n_estimators=10, seed=42):
        self.n_estimators = n_estimators
        self.seed = seed

    def impute(self, incomp_data, verbose=True):
        """
        执行XGBoost填补
        """
        if verbose:
            print(f"(IMPUTATION) XGBOOST\n\tMatrix: {incomp_data.shape}")

        if isinstance(incomp_data, np.ndarray):
            incomp_data = pd.DataFrame(incomp_data)

        recov_data = incomp_data.copy()

        for column in recov_data.columns:
            model = XGBRegressor(n_estimators=self.n_estimators, random_state=self.seed)

            non_missing = recov_data.loc[incomp_data[column].notna()]
            missing = recov_data.loc[incomp_data[column].isna()]

            # 如果该列全部缺失，用均值填充
            if len(non_missing) == 0:
                recov_data[column] = recov_data[column].fillna(0)
                continue

            X_train = non_missing.drop(columns=[column])
            y_train = non_missing[column]
            X_missing = missing.drop(columns=[column])

            # 如果缺失数据为空，跳过
            if len(X_missing) == 0:
                continue

            # 确保训练数据没有NaN
            if X_train.isna().any().any() or y_train.isna().any():
                # 如果有NaN，用简单方法填充该列
                recov_data[column] = recov_data[column].fillna(recov_data[column].mean())
                continue

            # 训练模型并预测
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_missing)
                recov_data.loc[recov_data[column].isna(), column] = predictions
            except Exception as e:
                if verbose:
                    print(f"XGBoost填补列 {column} 失败: {e}，使用均值填充")
                recov_data[column] = recov_data[column].fillna(recov_data[column].mean())

        return np.array(recov_data)


# 为了向后兼容，保留函数
def xgboost_impute(incomp_data, n_estimators=10, seed=42, logs=True, verbose=True):
    imputer = XGBoostImputer(n_estimators, seed)
    return imputer.impute(incomp_data, verbose)