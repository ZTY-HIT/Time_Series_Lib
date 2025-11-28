import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

class Visualizer:
    """可视化工具类"""

    # 尝试加载常见中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    

    def plot_forecast_results(self, true_values, predicted_values, model_name, target_name, save_path=None):
        """
        绘制预测结果图
    
        Parameters:
        - true_values: Series, 真实值（必须按时间排序）
        - predicted_values: Series, 预测值（必须按时间排序）
        - model_name: str, 模型名称
        - target_name: str, 目标变量名称
        - save_path: str, 保存路径
        """
        plt.figure(figsize=(14, 7))
    
        # 确保数据按时间排序
        true_sorted = true_values.sort_index()
        pred_sorted = predicted_values.sort_index()
    
        print(f"📈 绘图数据信息:")
        print(f"  真实值时间范围: {true_sorted.index[0]} 到 {true_sorted.index[-1]}")
        print(f"  预测值时间范围: {pred_sorted.index[0]} 到 {pred_sorted.index[-1]}")
        print(f"  真实值数量: {len(true_sorted)}, 预测值数量: {len(pred_sorted)}")
    
        # 合并所有时间点用于X轴
        all_times = true_sorted.index.union(pred_sorted.index)
    
        # 绘制真实值
        plt.plot(true_sorted.index, true_sorted.values, 'b-', label='True Values', linewidth=2, marker='o', markersize=4)
    
        # 绘制预测值
        plt.plot(pred_sorted.index, pred_sorted.values, 'r--', label='Predictions', linewidth=2, marker='s', markersize=4)
    
        # 添加预测起始线
        forecast_start = pred_sorted.index[0]
        plt.axvline(x=forecast_start, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
        plt.title(f'{model_name} Forecast - {target_name}')
        plt.xlabel('Time')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        # 智能设置X轴标签
        if len(all_times) > 20:
            # 如果时间点太多，显示部分标签
            n_ticks = min(10, len(all_times))
            tick_indices = np.linspace(0, len(all_times)-1, n_ticks, dtype=int)
            plt.xticks([all_times[i] for i in tick_indices], 
                    [all_times[i].strftime('%Y-%m-%d') for i in tick_indices], 
                    rotation=45)
        else:
            # 显示所有标签
            plt.xticks(all_times, [t.strftime('%Y-%m-%d') for t in all_times], rotation=45)
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {save_path}")
        plt.show()

    
    def plot_imputation_comparison(self, original, imputed, mask, method_name, save_path=None):
        """
        绘制填补对比图
        
        Parameters:
        - original: ndarray, 原始数据
        - imputed: ndarray, 填补数据
        - mask: ndarray, 缺失位置掩码
        - method_name: str, 填补方法名称
        - save_path: str, 保存路径
        """
        # 随机选择一个变量进行可视化
        n_vars = original.shape[1]
        var_idx = np.random.randint(0, n_vars)
        
        plt.figure(figsize=(12, 6))
        
        time_idx = np.arange(len(original))
        original_var = original[:, var_idx]
        imputed_var = imputed[:, var_idx]
        mask_var = mask[:, var_idx]
        
        # 绘制原始数据（完整部分）
        plt.plot(time_idx, original_var, 'b-', label='Original', alpha=0.7, linewidth=1)
        
        # 标记缺失位置
        missing_indices = np.where(mask_var)[0]
        plt.scatter(missing_indices, imputed_var[missing_indices], 
                   color='red', s=30, label='Imputed Values', zorder=5)
        
        plt.title(f'Imputation Results - {method_name} (Variable {var_idx})')
        plt.xlabel('Time Index')
        plt.ylabel(f'Variable {var_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    def plot_forecast_results_safe(self, true_values, predicted_values, model_name, target_name, save_path=None):
        """
        安全的预测结果图绘制（避免时间索引问题）
        """
        plt.figure(figsize=(12, 6))
    
        # 完全使用数值索引，避免时间索引问题
        true_x = np.arange(len(true_values))
        pred_x = np.arange(len(true_values), len(true_values) + len(predicted_values))
    
        # 绘制真实值
        plt.plot(true_x, true_values.values, 'b-', label='True Values', linewidth=2, marker='o', markersize=4)
    
        # 绘制预测值
        plt.plot(pred_x, predicted_values.values, 'r--', label='Predictions', linewidth=2, marker='s', markersize=4)
    
        # 添加分隔线
        plt.axvline(x=len(true_values)-0.5, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
        plt.title(f'{model_name} Forecast - {target_name}')
        plt.xlabel('Time Steps')
        plt.ylabel(target_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()