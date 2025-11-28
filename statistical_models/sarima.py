import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from .base_forecaster import BaseForecaster

warnings.filterwarnings("ignore")

class SARIMAForecaster(BaseForecaster):
    """SARIMA 预测模型统一接口"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
    
    def forecast(self, df, forecast_steps=10, plot=True):
        from utils import DataLoader, Visualizer
        
        data_loader = DataLoader()
        visualizer = Visualizer()
        
        print(f"🔍 SARIMA模型数据检查:")
        print(f"  输入数据形状: {df.shape}")
        print(f"  数据列名: {list(df.columns)}")
        
        # 数据准备
        ts, pred_truth, freq = data_loader.prepare_forecast_data(df, forecast_steps)
        
        print(f"📊 SARIMA训练数据详情:")
        print(f"  训练序列长度: {len(ts)}")
        print(f"  训练数据范围: {ts.index[0]} 到 {ts.index[-1]}")
        print(f"  训练数据统计 - 均值: {ts.mean():.2f}, 标准差: {ts.std():.2f}")
        print(f"  缺失值数量: {ts.isna().sum()}")
        print(f"  时间频率: {freq}")
        print(f"  模型参数: order={self.order}, seasonal_order={self.seasonal_order}")
        
        # 拟合模型
        print(f"🔄 正在训练 SARIMA 模型...")
        try:
            model = SARIMAX(
                ts,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = model.fit(disp=False)
            self.is_fitted = True
            
            print(f"✅ SARIMA模型训练成功")
            print(f"  模型AIC: {self.model_fit.aic:.2f}")
            print(f"  模型BIC: {self.model_fit.bic:.2f}")
            print(f"  参数: order={self.order}, seasonal_order={self.seasonal_order}")
        except Exception as e:
            print(f"❌ SARIMA模型训练失败: {e}")
            raise

        # 预测
        print(f"🔮 正在进行 {forecast_steps} 步预测...")
        forecast = self.model_fit.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq=freq)[1:]
        forecast_series = pd.Series(forecast, index=forecast_index, name='forecast')

        print(f"📈 预测结果统计:")
        print(f"  预测值范围: {forecast_series.min():.2f} 到 {forecast_series.max():.2f}")
        print(f"  预测均值: {forecast_series.mean():.2f}")
        print(f"  预测时间范围: {forecast_series.index[0]} 到 {forecast_series.index[-1]}")

        # 绘制结果
        if plot:
            print(f"🎨 绘制预测结果图...")
            visualizer.plot_forecast_results(pred_truth, forecast_series, "SARIMA", df.columns[-1])
        
        return forecast_series
    
    def get_params(self):
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }