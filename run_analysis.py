#!/usr/bin/env python3
"""
分析汇总脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

from utils import Evaluator, Visualizer
from config import load_config


# 尝试加载常见中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def run_analysis():
    """运行分析汇总"""
    config = load_config()
    evaluator = Evaluator()
    visualizer = Visualizer()
    
    print("📊 开始分析实验结果...")
    
    # 加载实验结果
    imputation_results = load_imputation_results()
    forecast_results = load_forecast_results()
    
    if imputation_results is not None:
        analyze_imputation_results(imputation_results, visualizer)
    
    if forecast_results is not None:
        analyze_forecast_results(forecast_results, visualizer, evaluator)
    
    if imputation_results is not None and forecast_results is not None:
        analyze_correlations(imputation_results, forecast_results, visualizer)
    
    print("✅ 分析完成！结果保存在 results/ 目录")

def load_imputation_results():
    """加载填补结果"""
    config = load_config()
    file_path = Path(config['paths']['outputs_impute']) / 'imputation_metrics.csv'
    
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        print("⚠️  未找到填补结果文件")
        return None

def load_forecast_results():
    """加载预测结果"""
    config = load_config()
    file_path = Path(config['paths']['outputs_forecast']) / 'forecast_metrics.csv'
    
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        print("⚠️  未找到预测结果文件")
        return None

def analyze_imputation_results(results, visualizer):
    """分析填补结果"""
    print("🔍 分析填补结果...")
    
    # 按填补方法分组分析
    method_performance = results.groupby('imputer').agg({
        'RMSE_imp': 'mean',
        'MAE_imp': 'mean', 
        'R2_imp': 'mean',
        'Time_imp': 'mean'
    }).round(4)
    
    # 保存填补方法性能排名
    method_performance.to_csv('results/imputation_method_ranking.csv')
    
    # 尝试加载常见中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 可视化填补方法比较
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(data=results, x='imputer', y='RMSE_imp')
    plt.title('填补方法 RMSE 比较')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(data=results, x='imputer', y='R2_imp')
    plt.title('填补方法 R² 比较')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=results, x='imputer', y='RMSE_imp')
    plt.title('填补方法 RMSE 分布')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=results, x='Time_imp', y='RMSE_imp', hue='imputer')
    plt.title('时间成本 vs 填补质量')
    
    plt.tight_layout()
    plt.savefig('results/imputation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_forecast_results(results, visualizer, evaluator):
    """分析预测结果"""
    print("🔍 分析预测结果...")
    
    # 按预测模型分组分析
    model_performance = results.groupby('model').agg({
        'RMSE_pred': 'mean',
        'MAE_pred': 'mean',
        'MAPE_pred': 'mean',
        'skill_score': 'mean'
    }).round(4)
    
    model_performance.to_csv('results/forecast_model_ranking.csv')
    
    # 尝试加载常见中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 可视化预测模型比较
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    sns.barplot(data=results, x='model', y='RMSE_pred')
    plt.title('预测模型 RMSE 比较')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 2)
    sns.barplot(data=results, x='model', y='MAPE_pred')
    plt.title('预测模型 MAPE 比较')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 3)
    sns.barplot(data=results, x='model', y='skill_score')
    plt.title('预测模型 Skill Score 比较')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 4)
    sns.boxplot(data=results, x='model', y='RMSE_pred')
    plt.title('预测模型 RMSE 分布')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 5)
    # 填补方法对预测性能的影响
    if 'imputer' in results.columns:
        pivot_data = results.pivot_table(
            values='RMSE_pred', 
            index='model', 
            columns='imputer', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('模型 vs 填补方法 热力图')
    
    plt.tight_layout()
    plt.savefig('results/forecast_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations(imputation_results, forecast_results, visualizer):
    """分析填补与预测的关联性"""
    print("🔍 分析填补与预测的关联性...")
    
    # 合并数据
    merged_data = pd.merge(
        forecast_results, 
        imputation_results, 
        on=['dataset', 'pattern', 'rate', 'imputer'],
        suffixes=('_forecast', '_imputation')
    )
    
    # 计算相关性
    correlation_analysis = merged_data[[
        'RMSE_imp', 'R2_imp', 'RMSE_pred', 'MAPE_pred'
    ]].corr()
    
    correlation_analysis.to_csv('results/correlation_analysis.csv')
    
    # 尝试加载常见中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 可视化相关性
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_analysis, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('填补质量与预测性能相关性')
    plt.tight_layout()
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 填补质量 vs 预测性能散点图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=merged_data, x='RMSE_imp', y='RMSE_pred', hue='model')
    plt.xlabel('填补 RMSE')
    plt.ylabel('预测 RMSE')
    plt.title('填补质量 vs 预测性能')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=merged_data, x='R2_imp', y='RMSE_pred', hue='model')
    plt.xlabel('填补 R²')
    plt.ylabel('预测 RMSE')
    plt.title('填补拟合度 vs 预测性能')
    
    plt.tight_layout()
    plt.savefig('results/imputation_forecast_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_analysis()