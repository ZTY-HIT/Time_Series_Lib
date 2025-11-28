import os
import yaml

def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def get_imputer_class(imputer_name):
    """根据填补器名称返回对应的类"""
    from imputer import (
        MeanImputer, FrontImputer, KNNImputer, MissForestImputer,
        TRMFImputer, XGBoostImputer, IIMImputer
    )

    imputer_map = {
        "mean": MeanImputer,
        "front": FrontImputer,
        "knn": KNNImputer,
        "miss_forest": MissForestImputer,
        "trmf": TRMFImputer,
        "xgboost": XGBoostImputer,
        "ilm": IIMImputer  # 注意：配置中是"ilm"，但类名是IIMImputer
    }

    return imputer_map.get(imputer_name)


def get_forecast_class(model_name):
    """根据预测模型名称返回对应的类"""
    from statistical_models import (
        ARIMAForecaster, AutoARIMAForecaster, AutoSARIMAForecaster, SARIMAForecaster
    )
    
    model_map = {
        "arima": ARIMAForecaster,
        "auto_arima": AutoARIMAForecaster,
        "auto_sarima": AutoSARIMAForecaster,
        "sarima": SARIMAForecaster
    }
    
    return model_map.get(model_name)

__all__ = ["load_config", "get_imputer_class", "get_forecast_class"]