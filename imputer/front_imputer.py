import numpy as np
import pandas as pd


class FrontImputer:
    """
    前向填充填补方法
    """

    def __init__(self):
        pass

    def impute(self, raw_data):
        """
        执行前向填充

        Parameters:
        - raw_data: numpy.ndarray, 包含NaN值的原始数据

        Returns:
        - numpy.ndarray, 填补后的数据
        """
        df = pd.DataFrame(raw_data)
        imputed_df = df.ffill()  # 前向填充
        return imputed_df.values


# 为了向后兼容，保留函数
def front_impute(raw_data):
    imputer = FrontImputer()
    return imputer.impute(raw_data)