from abc import ABC, abstractmethod
import pandas as pd

class BaseForecaster(ABC):
    """预测模型统一基类"""
    
    def __init__(self):
        self.model_fit = None
        self.is_fitted = False
    
    @abstractmethod
    def forecast(self, df, forecast_steps=10, plot=True):
        """执行预测"""
        pass
    
    def get_model_name(self):
        """获取模型名称"""
        return self.__class__.__name__
    
    def get_params(self):
        """获取模型参数"""
        return {}