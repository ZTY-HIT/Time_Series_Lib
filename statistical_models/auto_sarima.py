import pandas as pd
import numpy as np
import warnings
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")

class AutoSARIMAForecaster(BaseForecaster):
    """自动 SARIMA 预测模型统一接口"""
    
    def __init__(self, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1, seasonal=True, m=None):
        super().__init__()
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal = seasonal
        self.m = m
        self.best_order = None
        self.best_seasonal_order = None
        self.best_aic = None
    
    def forecast(self, df, forecast_steps=10, plot=True):
        from utils import DataLoader, Visualizer
        
        data_loader = DataLoader()
        visualizer = Visualizer()
        
        print(f"🔍 AutoSARIMA模型数据检查:")
        print(f"  输入数据形状: {df.shape}")
        print(f"  数据列名: {list(df.columns)}")
        
        # 数据准备
        ts, pred_truth, freq = data_loader.prepare_forecast_data(df, forecast_steps)

        print(f"📊 AutoSARIMA训练数据详情:")
        print(f"  训练序列长度: {len(ts)}")
        print(f"  训练数据范围: {ts.index[0]} 到 {ts.index[-1]}")
        print(f"  训练数据统计 - 均值: {ts.mean():.2f}, 标准差: {ts.std():.2f}")
        print(f"  缺失值数量: {ts.isna().sum()}")
        print(f"  时间频率: {freq}")

        # 若 m 未指定，则自动推断季节长度
        m = self.m
        if self.seasonal and m is None:
            if freq.upper().startswith("M"):
                m = 12   # 月频 → 年季节性
            elif freq.upper().startswith("W"):
                m = 52   # 周频 → 年季节性
            elif freq.upper().startswith("D"):
                m = 7    # 日频 → 周季节性
            elif freq.upper().startswith("H"):
                m = 24   # 小时频 → 日季节性
            else:
                m = 1    # 默认无季节
            print(f"🔄 自动推断季节长度 m = {m}")

        print(f"🔎 参数搜索范围:")
        print(f"  order: p=[0,{self.max_p}], d=[0,{self.max_d}], q=[0,{self.max_q}]")
        print(f"  seasonal: P=[0,{self.max_P}], D=[0,{self.max_D}], Q=[0,{self.max_Q}], m={m}")
        total_combinations = (self.max_p+1) * (self.max_d+1) * (self.max_q+1) * (self.max_P+1) * (self.max_D+1) * (self.max_Q+1)
        print(f"  总参数组合数: {total_combinations}")

        # 参数搜索
        print(f"🔎 开始搜索最优SARIMA参数...")
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        tested_models = 0
        successful_models = 0

        for p, d, q in product(range(self.max_p + 1), range(self.max_d + 1), range(self.max_q + 1)):
            for P, D, Q in product(range(self.max_P + 1), range(self.max_D + 1), range(self.max_Q + 1)):
                tested_models += 1
                seasonal_order = (P, D, Q, m) if self.seasonal else (0, 0, 0, 0)
                try:
                    model = SARIMAX(
                        ts,
                        order=(p, d, q),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False)
                    successful_models += 1
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                        best_seasonal_order = seasonal_order
                        best_model = result
                        print(f"  🎯 发现更优参数: order=({p},{d},{q}), seasonal={seasonal_order}, AIC: {best_aic:.2f}")
                except Exception:
                    continue

        self.best_order = best_order
        self.best_seasonal_order = best_seasonal_order
        self.best_aic = best_aic
        self.model_fit = best_model
        self.is_fitted = True

        print(f"✅ 参数搜索完成:")
        print(f"  测试模型数: {tested_models}")
        print(f"  成功拟合数: {successful_models}")
        print(f"  成功率: {successful_models/tested_models*100:.1f}%")
        print(f"  最优SARIMA参数: order={best_order}, seasonal_order={best_seasonal_order}, AIC={best_aic:.2f}")

        # 预测
        print(f"🔮 正在进行 {forecast_steps} 步预测...")
        forecast = best_model.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq=freq)[1:]
        forecast_series = pd.Series(forecast, index=forecast_index, name='forecast')

        print(f"📈 预测结果统计:")
        print(f"  预测值范围: {forecast_series.min():.2f} 到 {forecast_series.max():.2f}")
        print(f"  预测均值: {forecast_series.mean():.2f}")
        print(f"  预测时间范围: {forecast_series.index[0]} 到 {forecast_series.index[-1]}")

        # 绘制结果
        if plot:
            print(f"🎨 绘制预测结果图...")
            visualizer.plot_forecast_results(pred_truth, forecast_series, "Auto-SARIMA", df.columns[-1])
        
        return forecast_series
    
    def get_params(self):
        return {
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "max_P": self.max_P,
            "max_D": self.max_D,
            "max_Q": self.max_Q,
            "seasonal": self.seasonal,
            "m": self.m,
            "best_order": self.best_order,
            "best_seasonal_order": self.best_seasonal_order,
            "best_aic": self.best_aic
        }