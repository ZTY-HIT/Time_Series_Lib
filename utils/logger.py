import logging
import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.setup_logging()
    
    
    def setup_logging(self):
        """设置日志配置"""
        import sys
        import io
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.log")
            
        # 为处理器指定UTF-8编码
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 为 stdout 创建一个使用 UTF-8 编码的文本包装器，避免调用不存在的 setEncoding
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        stream_handler = logging.StreamHandler(stream)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_experiment_start(self, config):
        """记录实验开始"""
        self.logger.info("🚀 开始实验")
        self.logger.info(f"实验配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    def log_imputation_result(self, dataset, pattern, rate, imputer, metrics):
        """记录填补结果"""
        self.logger.info(
            f"📊 填补结果 - 数据集: {dataset}, 模式: {pattern}, "
            f"缺失率: {rate}, 方法: {imputer}, RMSE: {metrics.get('RMSE_imp', 'N/A'):.4f}"
        )
    
    def log_forecast_result(self, dataset, model, metrics, skill_score=None):
        """记录预测结果"""
        msg = f"📈 预测结果 - 数据集: {dataset}, 模型: {model}, RMSE: {metrics.get('RMSE_pred', 'N/A'):.4f}"
        if skill_score is not None:
            msg += f", Skill Score: {skill_score:.2f}%"
        self.logger.info(msg)
    
    def log_error(self, error_msg):
        """记录错误信息"""
        self.logger.error(f"❌ 错误: {error_msg}")
    
    def save_results(self, results, filename):
        """保存结果到文件"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            if filename.endswith('.json'):
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(results))
        
        self.logger.info(f"💾 结果已保存: {filepath}")
class ResultsManager:
    """🆕 新增：实验结果管理器 - 实时更新和覆盖写入"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def update_forecast_results(self, result_dict, filename="forecast_results.csv"):
        """
        更新预测结果CSV文件
        
        Parameters:
        - result_dict: 单次实验结果字典
        - filename: 结果文件名
        """
        file_path = self.results_dir / filename
        
        # 添加时间戳
        result_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建唯一标识（用于覆盖相同配置的结果）
        config_id = self._generate_config_id(result_dict)
        result_dict['config_id'] = config_id
        
        if file_path.exists():
            # 读取现有结果
            existing_df = pd.read_csv(file_path)
            
            # 检查是否存在相同配置的结果
            mask = existing_df['config_id'] == config_id
            if mask.any():
                # 覆盖更新现有结果
                for key, value in result_dict.items():
                    existing_df.loc[mask, key] = value
                print(f"📝 更新现有实验结果: {config_id}")
            else:
                # 添加新结果
                new_df = pd.DataFrame([result_dict])
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"✅ 添加新实验结果: {config_id}")
        else:
            # 创建新文件
            existing_df = pd.DataFrame([result_dict])
            print(f"🆕 创建实验结果文件，添加: {config_id}")
        
        # 保存文件
        existing_df.to_csv(file_path, index=False)
        print(f"💾 实验结果已保存: {file_path}")
        
        return existing_df
    
    def update_csv_results(self, result_dict, file_path, config_keys=None):
        """
        🆕 通用方法：更新任何 CSV 结果文件，实现实时记录和覆盖功能
        
        Parameters:
        - result_dict: 单次实验结果字典
        - file_path: CSV 文件路径
        - config_keys: 用于生成唯一标识的键列表（默认使用所有标准配置键）
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确定配置键
        if config_keys is None:
            config_keys = ['dataset', 'pattern', 'rate', 'imputer', 'model', 'forecast_steps']
        
        # 创建唯一标识
        config_id = self._generate_config_id(result_dict, config_keys)
        
        # 添加时间戳（可选）
        if 'timestamp' not in result_dict:
            result_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if file_path.exists():
            # 读取现有结果
            existing_df = pd.read_csv(file_path)
            
            # 检查是否存在相同配置的结果
            if 'config_id' in existing_df.columns:
                mask = existing_df['config_id'] == config_id
            else:
                # 如果没有 config_id 列，动态创建匹配条件
                mask = pd.Series([True] * len(existing_df))
                for key in config_keys:
                    if key in existing_df.columns and key in result_dict:
                        mask = mask & (existing_df[key] == result_dict[key])
            
            if mask.any() and mask.sum() > 0:
                # 覆盖更新现有结果
                for key, value in result_dict.items():
                    if key in existing_df.columns:
                        existing_df.loc[mask, key] = value
                    else:
                        # 如果列不存在，添加新列
                        existing_df[key] = None
                        existing_df.loc[mask, key] = value
                print(f"📝 更新现有记录: {config_id} -> {file_path.name}")
            else:
                # 添加新结果
                new_df = pd.DataFrame([result_dict])
                existing_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"✅ 添加新记录: {config_id} -> {file_path.name}")
        else:
            # 创建新文件
            existing_df = pd.DataFrame([result_dict])
            print(f"🆕 创建文件: {file_path.name}，添加: {config_id}")
        
        # 确保保存 config_id 用于后续更新
        existing_df['config_id'] = config_id
        
        # 保存文件
        existing_df.to_csv(file_path, index=False)
        print(f"💾 结果已保存: {file_path}")
        
        return existing_df
    
    def _generate_config_id(self, result_dict, config_keys=None):
        """生成配置唯一标识"""
        if config_keys is None:
            config_keys = ['dataset', 'pattern', 'rate', 'imputer', 'model', 'forecast_steps']
        
        id_parts = []
        for key in config_keys:
            if key in result_dict:
                id_parts.append(str(result_dict[key]))
            else:
                id_parts.append('')
        return '_'.join(id_parts).replace(' ', '_')
    
    def calculate_skill_scores(self, results_file="forecast_results.csv"):
        """
        计算所有结果的 Skill Score（基于 mean 方法作为 baseline）
        """
        file_path = self.results_dir / results_file
        
        if not file_path.exists():
            print("⚠️  结果文件不存在，无法计算 Skill Score")
            return None
        
        df = pd.read_csv(file_path)
        
        if len(df) == 0:
            print("⚠️  结果文件为空，无法计算 Skill Score")
            return None
            
        # 计算每个配置的 baseline RMSE（使用 mean 方法）
        skill_scores = []
        
        for _, row in df.iterrows():
            # 找到相同配置的 baseline 结果（使用 mean 方法）
            baseline_mask = (
                (df['dataset'] == row['dataset']) &
                (df['pattern'] == row['pattern']) &
                (df['rate'] == row['rate']) &
                (df['model'] == row['model']) &
                (df['forecast_steps'] == row['forecast_steps']) &
                (df['imputer'] == 'mean')  # baseline 方法
            )
            
            baseline_results = df[baseline_mask]
            
            if len(baseline_results) > 0 and 'RMSE_pred' in baseline_results.columns:
                baseline_rmse = baseline_results.iloc[0]['RMSE_pred']
                current_rmse = row['RMSE_pred'] if 'RMSE_pred' in row else 0
                
                # 计算 Skill Score
                if baseline_rmse > 0 and current_rmse > 0:
                    skill_score = 100 * (1 - current_rmse / baseline_rmse)
                else:
                    skill_score = 0.0
            else:
                skill_score = 0.0  # 没有 baseline 数据
            
            skill_scores.append(skill_score)
        
        # 更新 Skill Score 列
        df['skill_score'] = skill_scores
        
        # 保存更新后的结果
        df.to_csv(file_path, index=False)
        print(f"📊 已更新 Skill Score: {len(skill_scores)} 条记录")
        
        return df
    def update_forecast_metrics(self, result_dict, output_dir="forecast_outputs"):
        """
        专门更新 forecast_metrics.csv 文件
        """
        output_path = Path(output_dir) / "forecast_metrics.csv"
        return self.update_csv_results(result_dict, output_path)