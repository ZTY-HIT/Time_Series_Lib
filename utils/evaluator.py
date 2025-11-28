import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import psutil
import os

class Evaluator:
    """评估指标计算器 - 实现实验设计所有指标"""
    
    @staticmethod
    def calculate_imputation_metrics(original, imputed, mask=None):
        """
        计算填补质量指标 - 对应实验设计5.1节
        """
        original_array = np.array(original)
        imputed_array = np.array(imputed)
        
        if mask is None:
            mask = np.isnan(original_array)
        
        # 缺失位置索引集合 Ω
        missing_indices = np.where(mask)
        
        if len(missing_indices[0]) == 0:
            return {"error": "No missing values to evaluate"}
        
        original_missing = original_array[missing_indices]
        imputed_missing = imputed_array[missing_indices]
        
        # 1. 均方根误差 RMSE
        rmse = np.sqrt(np.mean((imputed_missing - original_missing) ** 2))
        
        # 2. 平均绝对误差 MAE
        mae = np.mean(np.abs(imputed_missing - original_missing))
        
        # 3. 平均偏差 Bias
        bias = np.mean(imputed_missing - original_missing)
        
        # 4. 拟合优度 R²
        ss_res = np.sum((imputed_missing - original_missing) ** 2)
        ss_tot = np.sum((original_missing - np.mean(original_missing)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            "RMSE_imp": rmse,
            "MAE_imp": mae, 
            "Bias_imp": bias,
            "R2_imp": r2
        }
    
    @staticmethod
    def calculate_forecast_metrics(true_values, predicted_values):
        """
        计算预测性能指标 - 对应实验设计5.2节
        """
        true_array = np.array(true_values).flatten()
        pred_array = np.array(predicted_values).flatten()
        
        valid_mask = ~(np.isnan(true_array) | np.isnan(pred_array))
        true_valid = true_array[valid_mask]
        pred_valid = pred_array[valid_mask]
        
        if len(true_valid) == 0:
            return {"error": "No valid values to evaluate"}
        
        n = len(true_valid)
        
        # 1. 预测 RMSE
        rmse = np.sqrt(np.sum((pred_valid - true_valid) ** 2) / n)
        
        # 2. 预测 MAE
        mae = np.sum(np.abs(pred_valid - true_valid)) / n
        
        # 3. 平均绝对百分比误差 MAPE
        epsilon = 1e-8
        mape = 100 * np.sum(np.abs((pred_valid - true_valid) / (np.abs(true_valid) + epsilon))) / n
        
        return {
            "RMSE_pred": rmse,
            "MAE_pred": mae,
            "MAPE_pred": mape
        }
    
    @staticmethod
    def calculate_skill_score(baseline_rmse, model_rmse):
        """计算相对提升度 Skill Score - 对应实验设计5.2节"""
        if baseline_rmse <= 0:
            return 0.0
        
        skill_score = 100 * (1 - model_rmse / baseline_rmse)
        
        # 🆕 新增：解释 Skill Score 含义
        if skill_score > 0:
            interpretation = f"比 baseline (mean) 提升 {skill_score:.1f}%"
        elif skill_score < 0:
            interpretation = f"比 baseline (mean) 差 {abs(skill_score):.1f}%"
        else:
            interpretation = "与 baseline (mean) 持平"
        
        print(f"📈 Skill Score: {skill_score:.1f}% ({interpretation})")
        return skill_score
    
    @staticmethod
    def calculate_computational_metrics(start_time, end_time, process=None):
        """
        计算计算成本指标 - 对应实验设计5.3节
        """
        time_cost = end_time - start_time
        
        if process is None:
            # 获取当前进程内存使用
            process = psutil.Process(os.getpid())
        
        memory_usage = process.memory_info().rss / 1024 / 1024  # 转换为MB
        
        return {
            "Time_imp": time_cost,
            "Mem_imp": memory_usage
        }
    
    @staticmethod
    def calculate_robustness_metrics(rmse_list, missing_rates):
        """
        计算鲁棒性指标 - 对应实验设计5.4节
        """
        rmse_array = np.array(rmse_list)
        missing_rates_array = np.array(missing_rates)
        
        # 1. 种子方差（稳定性）
        seed_variance = np.var(rmse_array) if len(rmse_array) > 1 else 0
        
        # 2. 缺失率敏感度 Slope
        if len(missing_rates_array) >= 2:
            slope = np.polyfit(missing_rates_array, rmse_array, 1)[0]
        else:
            slope = 0
        
        return {
            "sigma_pred": np.sqrt(seed_variance),
            "Slope": slope
        }
    
    @staticmethod
    def calculate_comprehensive_score(metrics_dict, weights=None):
        """
        计算综合评分 - 对应实验设计第6节
        """
        if weights is None:
            weights = {
                'imputation': 0.25,
                'forecast': 0.45, 
                'cost': 0.15,
                'robustness': 0.15
            }
        
        # 这里需要多个实验结果的聚合，先预留接口
        # 实际实现需要在多个实验运行后计算
        return {
            "Q_imp": 0,  # 填补质量子分数
            "Q_down": 0, # 下游预测子分数  
            "Q_cost": 0, # 计算成本子分数
            "Q_rob": 0,  # 鲁棒性子分数
            "S_total": 0 # 最终综合评分
        }
    
    @staticmethod
    def normalize_metric(value, min_val, max_val, higher_better=False):
        """指标归一化"""
        if higher_better:
            return (value - min_val) / (max_val - min_val + 1e-8)
        else:
            return (max_val - value) / (max_val - min_val + 1e-8)