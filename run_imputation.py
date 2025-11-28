#!/usr/bin/env python3
"""
填补实验主脚本 - 支持增量保存
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import psutil

from utils import DataLoader, Evaluator, ExperimentLogger
from config import load_config, get_imputer_class

def run_imputation_experiment():
    """运行填补实验 - 支持增量保存"""
    parser = argparse.ArgumentParser(description='运行填补实验')
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--pattern', type=str, help='缺失模式')
    parser.add_argument('--rate', type=float, help='缺失率')
    parser.add_argument('--imputer', type=str, help='填补方法')
    parser.add_argument('--all', action='store_true', help='运行所有组合')
    
    args = parser.parse_args()
    config = load_config()
    logger = ExperimentLogger()
    data_loader = DataLoader()
    evaluator = Evaluator()
    
    logger.log_experiment_start(config)
    
    # 获取实验组合
    if args.all:
        experiments = generate_all_experiments(config)
    else:
        experiments = [{
            'dataset': args.dataset or config['datasets']['default'],
            'pattern': args.pattern or config['missing_patterns']['default'],
            'rate': args.rate or config['missing_rates']['default'],
            'imputer': args.imputer or config['imputers']['default']
        }]
    
    # 设置输出目录
    output_dir = Path(config['paths']['outputs_impute'])
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / 'imputation_metrics.csv'
    
    # 加载已有的结果（如果存在）
    existing_results = []
    if results_file.exists():
        try:
            existing_results_df = pd.read_csv(results_file)
            existing_results = existing_results_df.to_dict('records')
            print(f"📁 加载已有结果: {len(existing_results)} 条记录")
        except Exception as e:
            print(f"⚠️  加载已有结果失败: {e}，将重新开始")
    
    # 过滤掉已经完成的实验
    completed_experiments = set()
    for r in existing_results:
        key = (r['dataset'], r['pattern'], r['rate'], r['imputer'])
        completed_experiments.add(key)
    
    remaining_experiments = []
    for exp in experiments:
        key = (exp['dataset'], exp['pattern'], exp['rate'], exp['imputer'])
        if key not in completed_experiments:
            remaining_experiments.append(exp)
    
    print(f"🔧 总实验数: {len(experiments)}")
    print(f"✅ 已完成: {len(completed_experiments)}")
    print(f"⏳ 待完成: {len(remaining_experiments)}")
    
    if not remaining_experiments:
        print("🎉 所有实验已完成！")
        return
    
    results = existing_results.copy()
    
    # 开始运行剩余实验
    for i, exp in enumerate(remaining_experiments):
        try:
            print(f"\n[{i+1}/{len(remaining_experiments)}] 运行实验: {exp}")
            
            # 加载数据
            missing_data = data_loader.load_missing_data(exp['dataset'], exp['pattern'], exp['rate'])
            original_data = data_loader.load_original_data(exp['dataset'])
            
            # 验证时间戳
            if not data_loader.validate_timestamp_column(original_data):
                raise ValueError(f"数据集 {exp['dataset']} 第一列不是有效时间戳")
            
            # 提取数值数据（排除时间戳列）
            missing_values = missing_data.iloc[:, 1:].values  # 第一列是时间戳
            original_values = original_data.iloc[:, 1:].values
            
            # 获取填补器
            imputer_class = get_imputer_class(exp['imputer'])
            if not imputer_class:
                raise ValueError(f"未知的填补方法: {exp['imputer']}")
            
            # 执行填补
            start_time = time.time()
            process = psutil.Process()
            
            # 创建填补器实例并执行
            imputer_instance = imputer_class()
            imputed_values = imputer_instance.impute(missing_values)
            
            end_time = time.time()
            
            # 计算指标
            mask = np.isnan(missing_values)
            imputation_metrics = evaluator.calculate_imputation_metrics(
                original_values, imputed_values, mask
            )
            computational_metrics = evaluator.calculate_computational_metrics(
                start_time, end_time, process
            )
            
            # 保存结果
            result = {
                'dataset': exp['dataset'],
                'pattern': exp['pattern'],
                'rate': exp['rate'],
                'imputer': exp['imputer'],
                **imputation_metrics,
                **computational_metrics
            }
            results.append(result)
            
            # 记录日志
            logger.log_imputation_result(
                exp['dataset'], exp['pattern'], exp['rate'], exp['imputer'], imputation_metrics
            )
            
            # 保存填补结果
            save_imputation_result(imputed_values, exp, missing_data)
            
            print(f"✅ 实验完成: {exp['imputer']} - RMSE: {imputation_metrics.get('RMSE_imp', 'N/A'):.4f}")
            
            # 增量保存：每完成一个实验就保存一次
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file, index=False)
            print(f"💾 已保存 {len(results)} 条结果到 {results_file}")
            
        except Exception as e:
            logger.log_error(f"实验失败 {exp}: {str(e)}")
            print(f"❌ 实验失败: {exp} - {str(e)}")
            
            # 即使失败也保存已有结果
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file, index=False)
            print(f"💾 失败后保存 {len(results)} 条结果")
            continue
    
    print(f"\n🎉 所有实验完成！共完成 {len(results)}/{len(experiments)} 个实验")
    print(f"📊 最终结果保存至: {results_file}")

def generate_all_experiments(config):
    """生成所有实验组合"""
    experiments = []
    for dataset in config['datasets']['available']:
        for pattern in config['missing_patterns']['available']:
            for rate in config['missing_rates']['available']:
                for imputer in config['imputers']['available']:
                    experiments.append({
                        'dataset': dataset,
                        'pattern': pattern,
                        'rate': rate,
                        'imputer': imputer
                    })
    return experiments

def save_imputation_result(imputed_data, exp_config, original_df):
    """保存填补结果"""
    config = load_config()
    output_dir = Path(config['paths']['outputs_impute']) / exp_config['dataset']
    output_dir.mkdir(exist_ok=True)
    
    # 重建DataFrame（保持时间戳列）
    result_df = original_df.copy()
    result_df.iloc[:, 1:] = imputed_data  # 第一列是时间戳
    
    filename = f"{exp_config['pattern']}_rate{exp_config['rate']}_{exp_config['imputer']}_imputed.csv"
    result_df.to_csv(output_dir / filename, index=False)

if __name__ == "__main__":
    run_imputation_experiment()