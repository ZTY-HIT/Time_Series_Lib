import numpy as np
import pandas as pd


class MeanImputer:
    """
    均值填补方法
    """

    def __init__(self):
        pass

    def impute(self, raw_data):
        """
        执行均值填补

        Parameters:
        - raw_data: numpy.ndarray, 包含NaN值的原始数据

        Returns:
        - numpy.ndarray, 填补后的数据
        """
        df = pd.DataFrame(raw_data)
        imputed_df = df.fillna(df.mean())  # 列均值填充
        return imputed_df.values


# 为了向后兼容，保留函数
def mean_impute(raw_data):
    imputer = MeanImputer()
    return imputer.impute(raw_data)