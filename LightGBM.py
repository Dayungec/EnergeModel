import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import torch
import os
import warnings

import time

from EnergeModel.Tools import Config, DataReader
from EnergeModel.Tools.RasterProcessor import RasterProcessor
import Tools.Regress

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ----------------------------
# 2. LightGBM模型类
# ----------------------------
class LightGBMSusceptibilityModel:
    def __init__(self, input_dim: int, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def build_model(self, **lgb_params):
        """构建LightGBM模型"""
        default_params = {
            'n_estimators': 700,
            'learning_rate': 0.005,
            'max_depth': 3,
            'num_leaves': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbose': -1,
            'force_row_wise': True
        }

        # 更新用户自定义参数
        default_params.update(lgb_params)

        self.model = lgb.LGBMClassifier(**default_params)
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        if self.model is None:
            raise ValueError("请先调用 build_model() 方法构建模型")

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        else:
            eval_set = None
            callbacks = [lgb.log_evaluation(period=100)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        return self.model

    def predict_proba(self, X, calibrated=True):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, importance_type='gain'):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.feature_importances_


# ----------------------------
# 3. 基于LightGBM的易发性评价模型
# ----------------------------
class LightGBMModel:
    def __init__(self, input_dim: int, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.lightgbm_model = LightGBMSusceptibilityModel(input_dim, random_state)
        self.feature_names = None

    def forward(self, x):
        """为了保持接口一致，实际不直接使用"""
        pass

    def training_step(self, batch, batch_idx):
        """训练步骤 - 为了保持接口兼容"""
        pass

    def configure_optimizers(self):
        """优化器配置 - 为了保持接口兼容"""
        pass

    def __call__(self, X):
        """使模型对象可被直接调用"""
        return self.predict_proba(X)

    def predict_proba(self, x, calibrated=True):
        """预测概率"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return self.lightgbm_model.predict_proba(x, calibrated)

    def fit(self, X_pos, X_neg, feature_names):
        """训练LightGBM模型"""
        self.feature_names = feature_names

        # 准备数据
        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])


        # 构建并训练模型
        self.lightgbm_model.build_model()
        self.lightgbm_model.fit(X_train, y_train)

        return self

    def get_feature_importance(self):
        return self.lightgbm_model.get_feature_importance()


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    start_time = time.time()
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    neg_path = Config.BASE_DIR + Config.NEG_FILE
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    output_dir = "result/results_lightgbm"

    # 运行训练评估流程
    print("🚀 开始训练...")
    df_pos = DataReader.load_data(pos_path)

    model = LightGBMModel(input_dim=df_pos.shape[1])
    results = Tools.Regress.train_and_evaluate(
        model,
        pos_path=pos_path,
        neg_path=neg_path,
        bg_path=bg_path,
        test_path=test_path,  # 传入测试集路径
        output_dir=output_dir
    )

    # 打印最终结果（增加测试集指标）
    print("\n⭐ 最终评估结果 ⭐")
    print(f"● AD_AUC分数: {results['ad_auc_score']:.4f}")
    print(f"● ROC_AUC分数: {results['auc_score']:.4f}")
    print(f"● 预测精度: {results['train_accuracy']:.4f}")
    print(f"● 预测密度: {results['train_density']:.4f}")
    print(
        f"● 正样本概率统计 - 均值: {results['pos_prob_mean']:.3f} ± {results['pos_prob_std']:.3f} | 中位数: {results['pos_median']:.3f}")
    print(
        f"● 负样本概率统计 - 均值: {results['neg_prob_mean']:.3f} ± {results['neg_prob_std']:.3f} | 中位数: {results['neg_median']:.3f}")

    # 打印测试集结果
    if 'test_accuracy' in results:
        print(f"\n🧪 测试集评估结果（纯灾害样本）:")
        print(f"● AD_AUC分数: {results['ad_auc_score_test']:.4f}")
        print(f"● ROC_AUC分数: {results['auc_score_test']:.4f}")
        print(f"● 分类准确率: {results['test_accuracy']:.2%}")  # 百分比格式更直观
        print(f"● 测试样本数: {results['test_size']}")

    print(f"\n📊 风险区域分析结果:")
    print("=" * 80)
    print(f"{'风险区域':<12} {'阈值范围':<20} {'灾害点数量':<10} {'灾害点比例':<12} {'背景样本比例':<12}")
    print("-" * 80)

    for risk_name, risk_info in results['risk_zones'].items():
        print(f"{risk_name:<12} {risk_info['阈值范围']:<20} {risk_info['灾害点数量']:<10} "
              f"{risk_info['灾害点比例']:<12.1%} {risk_info['背景样本比例']:<12.1%}")

    print(f"\n📊 Top 5重要特征:")
    for i, row in results['feature_importance'].head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print(f"● 结果保存路径: {os.path.abspath(output_dir)}")

    if Config.EXPORT_TIFF:
        feature_names = df_pos.columns.tolist()  # 假设df_pos已从pos_path加载
        # 构建特征映射字典（自动匹配entropies目录下的同名tif）
        feature_mapping = {
            feature: os.path.join(Config.BASE_DIR+"entropies", f"{feature}.tif")
            for feature in feature_names
            if os.path.exists(os.path.join(Config.BASE_DIR+"entropies", f"{feature}.tif"))
        }
        # 初始化处理器
        processor = RasterProcessor(model, feature_mapping)
        # 输出路径
        prob_tif_path = os.path.join(output_dir, "susceptibility_probability.tif")
        # 执行预测
        processor.predict_to_raster(prob_tif_path)
        print(f"✅ 空间概率分布已保存至: {os.path.abspath(prob_tif_path)}")
        risk_thresholds = {
            zone_name: {
                '下限阈值': float(zone_info['阈值范围'].split(' - ')[0]),
                '上限阈值': float(zone_info['阈值范围'].split(' - ')[1])
            }
            for zone_name, zone_info in results['risk_zones'].items()
        }

        zone_colors = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue']
        zones_tif_path = os.path.join(output_dir, "susceptibility_zones.tif")
        zones = processor.generate_susceptibility_zones(
            prob_tif_path=prob_tif_path,
            risk_thresholds=risk_thresholds,
            output_tif_path=zones_tif_path,
            colors=zone_colors
        )

    end_time = time.time()
    total_time = end_time - start_time

    # 格式化显示运行时间
    time_str = f"{total_time:.2f}秒"
    print(f"\n🎉 程序执行完成！总运行时间: {time_str}")
    print("=" * 60)