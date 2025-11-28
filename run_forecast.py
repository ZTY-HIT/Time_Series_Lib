#!/usr/bin/env python3
"""
预测实验主脚本
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import psutil

from utils import DataLoader, Evaluator, ExperimentLogger, Visualizer, ResultsManager 
from config import load_config, get_forecast_class

def run_forecast_experiment():
    """运行预测实验"""
    parser = argparse.ArgumentParser(description='运行预测实验')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--pattern', type=str, help='缺失模式')
    parser.add_argument('--rate', type=float, help='缺失率')
    parser.add_argument('--imputer', type=str, help='填补方法')
    parser.add_argument('--model', type=str, help='预测模型')
    parser.add_argument('--forecast_steps', type=int, default=10, help='预测步数')
    parser.add_argument('--all', action='store_true', help='运行所有组合')
    
    args = parser.parse_args()
    config = load_config()
    logger = ExperimentLogger()
    data_loader = DataLoader()
    evaluator = Evaluator()
    visualizer = Visualizer()
    results_manager = ResultsManager()
    
    logger.log_experiment_start(config)
    
    # 获取实验组合
    if args.all:
        experiments = generate_all_forecast_experiments(config)
    else:
        experiments = [{
            'dataset': args.dataset or config['datasets']['default'],
            'pattern': args.pattern or config['missing_patterns']['default'],
            'rate': args.rate or config['missing_rates']['default'],
            'imputer': args.imputer or config['imputers']['default'],
            'model': args.model or config['forecast_models']['default'],
            'forecast_steps': args.forecast_steps
        }]
    
    results = []
    
    for exp in experiments:
        try:
            print(f"\n🔧 运行预测实验: {exp}")
            
            # 加载填补结果
            imputed_data = load_imputation_result(exp)
            if imputed_data is None:
                print(f"⚠️  未找到填补结果: {exp}，跳过")
                continue
            
            # 加载原始数据用于计算指标
            original_data = data_loader.load_original_data(exp['dataset'])
            
            # 验证时间戳
            if not data_loader.validate_timestamp_column(original_data):
                raise ValueError(f"数据集 {exp['dataset']} 第一列不是有效时间戳")
            
            # 获取预测模型
            forecast_class = get_forecast_class(exp['model'])
            if not forecast_class:
                raise ValueError(f"未知的预测模型: {exp['model']}")
            
            # 实例化模型（这里可以根据需要传递参数）
            forecast_class = get_forecast_class(exp['model'])
            if not forecast_class:
                raise ValueError(f"未知的预测模型: {exp['model']}")
            
            # 执行预测
            start_time = time.time()
            process = psutil.Process()
            
            # 使用填补后的数据进行预测
            forecaster = forecast_class()
            # 使用填补后的数据进行预测
            forecast_result = forecaster.forecast(
                imputed_data, 
                forecast_steps=exp['forecast_steps'],
                plot=False  # 在主脚本中统一绘制
            )
            
            end_time = time.time()
            
            # 准备真实值用于评估
            ts, pred_truth, freq = data_loader.prepare_forecast_data(
                original_data, exp['forecast_steps']
            )
            
            # 计算预测指标
            forecast_metrics = evaluator.calculate_forecast_metrics(
                pred_truth.values, forecast_result.values
            )
            
            computational_metrics = evaluator.calculate_computational_metrics(
                start_time, end_time, process
            )
            
            # 🆕 修改：Skill Score 现在在 ResultsManager 中统一计算
            skill_score = 0.0  # 临时值，后面会重新计算
            
            # 保存结果到字典
            result = {
                'dataset': exp['dataset'],
                'pattern': exp['pattern'],
                'rate': exp['rate'],
                'imputer': exp['imputer'],
                'model': exp['model'],
                'forecast_steps': exp['forecast_steps'],
                **forecast_metrics,
                'skill_score': skill_score,
                **computational_metrics
            }
            results.append(result)

             # 🆕 修改：使用新的实时更新方法保存结果
            results_manager.update_forecast_results(result)
            
            # 记录日志
            logger.log_forecast_result(
                exp['dataset'], exp['model'], forecast_metrics, skill_score
            )
            
            # 保存预测结果和图表
            save_forecast_result(forecast_result, exp, pred_truth, visualizer)
            
        except Exception as e:
            logger.log_error(f"预测实验失败 {exp}: {str(e)}")
            continue
    
    # 🆕 修改：所有实验完成后统一计算 Skill Score
    if results:
        print("\n📊 正在计算所有实验的 Skill Score...")
        updated_df = results_manager.calculate_skill_scores()
    
        # 🆕 修改：使用通用方法实时更新 forecast_metrics.csv
        if updated_df is not None:
            output_dir = Path(config['paths']['outputs_forecast'])
        
            # 为每个结果更新 forecast_metrics.csv
            for _, row in updated_df.iterrows():
                result_dict = row.to_dict()
                # 使用专门的 forecast metrics 方法
                results_manager.update_forecast_metrics(result_dict, output_dir)
        
            print(f"💾 预测指标已实时更新")
    
        # 保持原有的 JSON 保存功能
        logger.save_results(results, 'forecast_results.json')

def generate_all_forecast_experiments(config):
    """生成所有预测实验组合"""
    experiments = []
    
    # 首先检查有哪些填补结果可用
    impute_output_dir = Path(config['paths']['outputs_impute'])
    if not impute_output_dir.exists():
        print("⚠️  未找到填补结果，请先运行填补实验")
        return experiments
    
    for dataset_dir in impute_output_dir.iterdir():
        if dataset_dir.is_dir():
            dataset = dataset_dir.name
            for imputed_file in dataset_dir.glob("*_imputed.csv"):
                # 解析文件名获取参数
                filename = imputed_file.stem
                parts = filename.split('_')
                pattern = parts[0]
                rate = float(parts[1].replace('rate', ''))
                imputer = parts[2]
                
                for model in config['forecast_models']['available']:
                    experiments.append({
                        'dataset': dataset,
                        'pattern': pattern,
                        'rate': rate,
                        'imputer': imputer,
                        'model': model,
                        'forecast_steps': 10
                    })
    
    return experiments

def load_imputation_result(exp_config):
    """加载填补结果"""
    config = load_config()
    filename = f"{exp_config['pattern']}_rate{exp_config['rate']}_{exp_config['imputer']}_imputed.csv"
    file_path = Path(config['paths']['outputs_impute']) / exp_config['dataset'] / filename
    
    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"📁 加载填补数据: {file_path}")
        print(f"  数据形状: {df.shape}")
        print(f"  数据列名: {list(df.columns)}")
        print(f"  前5行时间列: {df.iloc[:5, 0].tolist()}")
        return df
    else:
        print(f"❌ 填补结果文件不存在: {file_path}")
        return None

def save_forecast_result(forecast_result, exp_config, true_values, visualizer):
    """保存预测结果"""
    config = load_config()
    output_dir = Path(config['paths']['outputs_forecast']) / exp_config['dataset']
    output_dir.mkdir(exist_ok=True)
    
    # 保存预测数据
    filename = f"{exp_config['pattern']}_rate{exp_config['rate']}_{exp_config['imputer']}_{exp_config['model']}_forecast.csv"
    forecast_result.to_csv(output_dir / filename)
    
    # 保存预测图表（使用安全的方法）
    plot_filename = f"{exp_config['pattern']}_rate{exp_config['rate']}_{exp_config['imputer']}_{exp_config['model']}_plot.png"
    plot_path = output_dir / plot_filename
    
    try:
        # 首先尝试常规绘图
        visualizer.plot_forecast_results(
            true_values, forecast_result, 
            f"{exp_config['model']} ({exp_config['imputer']})", 
            exp_config['dataset'],
            save_path=plot_path
        )
    except Exception as e:
        print(f"⚠️  常规绘图失败，使用安全绘图: {e}")
        # 如果常规绘图失败，使用安全绘图
        if hasattr(visualizer, 'plot_forecast_results_safe'):
            visualizer.plot_forecast_results_safe(
                true_values, forecast_result, 
                f"{exp_config['model']} ({exp_config['imputer']})", 
                exp_config['dataset'],
                save_path=plot_path
            )
        else:
            print(f"❌ 安全绘图方法也不可用，跳过绘图")

if __name__ == "__main__":
    run_forecast_experiment()