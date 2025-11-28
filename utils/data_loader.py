import pandas as pd
import numpy as np
import os
from config import load_config

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.config = load_config()
        self.paths = self.config['paths']
    
    def load_original_data(self, dataset_name):
        """加载原始完整数据"""
        file_path = os.path.join(self.paths['data_raw'], f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"原始数据文件不存在: {file_path}")
        return pd.read_csv(file_path)
    
    def load_missing_data(self, dataset_name, missing_pattern, missing_rate):
        """加载指定缺失模式的数据"""
        pattern_name = f"{missing_pattern}_rate{missing_rate}_data.csv"
        file_path = os.path.join(
            self.paths['data_missing'], 
            dataset_name, 
            pattern_name
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"缺失数据文件不存在: {file_path}")
        return pd.read_csv(file_path)
    
    def validate_timestamp_column(self, df):
        """
        验证第一列是否为时间戳格式
        
        Parameters:
        - df: DataFrame, 输入数据
        
        Returns:
        - bool, 是否为有效时间戳
        """
        first_col = df.iloc[:, 0]
        
        # 尝试转换为时间戳
        try:
            pd.to_datetime(first_col)
            return True
        except:
            return False
    
    def prepare_forecast_data(self, df, forecast_steps):
        """
        准备预测数据
    
        Parameters:
        - df: DataFrame, 第一列为时间戳，最后一列为目标变量
        - forecast_steps: int, 预测步数
    
        Returns:
        - ts: Series, 训练时间序列
        - pred_truth: Series, 测试真实值
        - freq: str, 时间频率
        """
        # 验证第一列是否为时间戳
        if not self.validate_timestamp_column(df):
            raise ValueError("第一列不是时间戳格式")
    
        # 首先获取目标列名（在修改df之前）
        time_col = df.columns[0]
        target_col = df.columns[-1]  # 在修改df之前获取目标列名
    
        print(f"🔍 数据列信息: 时间列='{time_col}', 目标列='{target_col}'")
    
        # 确保时间列是 datetime 类型并按时间排序
        df_copy = df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        df_copy = df_copy.sort_values(by=time_col)  # 确保按时间排序
    
        time_series = df_copy.set_index(time_col)
    
        # 提取数据
        data = time_series[target_col].astype(float)
    
        # 检查数据有效性
        if data.isna().all():
            raise ValueError("目标列全部为缺失值，无法进行预测")
    
        # 划分训练测试 - 确保时间连续性
        if len(data) <= forecast_steps:
            raise ValueError(f"数据长度({len(data)})不足以进行{forecast_steps}步预测")
    
        ts = data.iloc[:-forecast_steps]
        pred_truth = data.iloc[-forecast_steps:]
    
        # 打印调试信息
        print(f"📊 数据分割信息:")
        print(f"  总数据点: {len(data)}")
        print(f"  训练数据: {len(ts)} (从 {ts.index[0]} 到 {ts.index[-1]})")
        print(f"  测试数据: {len(pred_truth)} (从 {pred_truth.index[0]} 到 {pred_truth.index[-1]})")
        print(f"  目标列统计 - 均值: {data.mean():.2f}, 标准差: {data.std():.2f}")
    
        # 推断频率
        try:
            freq = pd.infer_freq(ts.index)
            if freq is None and len(ts.index) > 1:
                # 计算时间间隔
                time_diffs = pd.Series(ts.index).diff().dropna()
                if len(time_diffs) > 0:
                    mode_diff = time_diffs.mode()
                    if len(mode_diff) > 0:
                        freq = pd.tseries.frequencies.to_offset(mode_diff[0])
            if freq is None:
                freq = "D"  # 默认日频率
        except Exception as e:
            print(f"⚠️  频率推断失败: {e}, 使用默认频率 'D'")
            freq = "D"
    
        print(f"  推断频率: {freq}")
    
        return ts, pred_truth, freq
    def validate_data_integrity(self, df):
        """
        验证数据完整性
        """
        time_col = df.columns[0]
    
        # 检查时间列是否唯一且单调递增
        times = pd.to_datetime(df[time_col])
        if times.duplicated().any():
            print("⚠️  警告: 时间列存在重复值")
            return False
    
        if not times.is_monotonic_increasing:
            print("⚠️  警告: 时间列不是单调递增")
            return False
    
        return True