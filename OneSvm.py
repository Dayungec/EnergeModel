import time
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import os
import warnings
import Tools.Positive
from EnergeModel.Tools import Config, DataReader
from EnergeModel.Tools.RasterProcessor import RasterProcessor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# ----------------------------
# 2. 单分类SVM模型类
# ----------------------------
class OneClassSVMSusceptibilityModel:
    """
    单分类SVM模型（仅使用正样本进行训练）
    适用于异常检测和新颖性检测任务
    """

    def __init__(self, input_dim: int, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def build_model(self, **ocsvm_params):
        """构建单分类SVM模型"""
        # 单分类SVM的默认参数
        default_params = {
            'nu': 0.1,  # 异常值比例的上界
            'kernel': 'rbf',  # 核函数类型
            'gamma': 'scale'
        }

        # 更新用户自定义参数
        default_params.update(ocsvm_params)

        self.model = OneClassSVM(**default_params)
        return self.model

    def fit(self, X_train, X_val=None):
        """
        训练模型 - 仅使用正样本
        单分类SVM只需要正样本进行训练
        """
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 单分类SVM训练（只需要正样本）
        self.model.fit(X_train_scaled)

        if X_val is not None:
            # 对于单分类，我们可以计算在验证集上的异常检测性能
            X_val_scaled = self.scaler.transform(X_val)
            # 预测结果：+1表示正常样本，-1表示异常样本
            val_pred = self.model.predict(X_val_scaled)
            # 计算正常样本的比例（可以作为性能参考）
            normal_ratio = np.sum(val_pred == 1) / len(val_pred)
            print(f"验证集正常样本比例: {normal_ratio:.4f}")

        return self.model

    def predict_proba(self, X):
        """
        预测样本为正常样本的概率
        将OneClassSVM的决策函数值转换为概率估计
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        # 数据标准化
        X_scaled = self.scaler.transform(X)

        # 使用decision_function获取到决策边界的距离
        # 距离越大，表示越可能是正常样本
        distances = self.model.decision_function(X_scaled)

        # 将距离转换为概率（使用sigmoid函数进行标准化）
        # 注意：这只是近似概率，单分类SVM不直接提供概率估计
        max_distance = np.max(np.abs(distances))
        if max_distance > 0:
            normalized_distances = distances / (2 * max_distance) + 0.5
        else:
            normalized_distances = 0.5 * np.ones_like(distances)

        probs = np.clip(normalized_distances, 0.001, 0.999)
        return probs

    def predict(self, X):
        """预测样本标签：+1正常，-1异常"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        """获取特征重要性（仅适用于线性核）"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        if self.model.kernel != 'linear':
            print("警告: 特征重要性仅适用于线性核One-Class SVM")
            return np.zeros(self.input_dim)

        return np.abs(self.model.coef_[0])


# ----------------------------
# 3. 基于单分类SVM的易发性评价模型
# ----------------------------
class OneClassSVMModel:
    def __init__(self, input_dim: int, random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        self.ocsvm_model = OneClassSVMSusceptibilityModel(input_dim, random_state)
        self.feature_names = None

    def forward(self, x):
        """为了保持接口一致"""
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

    def predict_proba(self, x):
        """预测概率"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return self.ocsvm_model.predict_proba(x)

    def fit(self, X_pos, X_bg,feature_names, **ocsvm_params):
        """
        训练单分类SVM模型 - 只需要正样本
        与传统二分类方法不同，单分类方法不需要负样本
        """
        self.feature_names = feature_names

        # 单分类SVM只需要正样本进行训练
        X_train = X_pos  # 仅使用正样本

        # 构建并训练模型
        self.ocsvm_model.build_model(**ocsvm_params)
        self.ocsvm_model.fit(X_train)

        print(f"✅ 单分类SVM训练完成，使用正样本数量: {len(X_train)}")
        return self

    def get_feature_importance(self):
        return self.ocsvm_model.get_feature_importance()


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    start_time = time.time()

    # 文件路径配置 - 移除了neg_path
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"  # 背景样本用于评估预测密度
    test_path = Config.BASE_DIR + "test.xlsx"
    output_dir = "result/results_oneclass_svm"

    # 运行训练评估流程
    print("🚀 开始训练单分类SVM模型...")
    df_pos = DataReader.load_data(pos_path)

    # 创建单分类SVM模型
    model = OneClassSVMModel(input_dim=df_pos.shape[1])

    # 修改训练评估调用，适应单分类模式
    # 注意：这里需要调整Tools.Regress.train_and_evaluate函数以支持单分类
    results = Tools.Positive.train_and_evaluate(
        model,
        pos_path=pos_path,
        bg_path=bg_path,
        test_path=test_path,  # 传入测试集路径
        output_dir=output_dir
    )

    # 打印最终结果
    print("\n⭐ 单分类SVM最终评估结果 ⭐")
    print(f"● AD_AUC分数: {results['ad_auc_score']:.4f}")
    print(f"● 预测精度: {results['train_accuracy']:.4f}")
    print(f"● 预测密度: {results['train_density']:.4f}")
    print(
        f"● 正样本概率统计 - 均值: {results['pos_prob_mean']:.3f} ± {results['pos_prob_std']:.3f} | 中位数: {results['pos_median']:.3f}")

    # 单分类模型没有负样本，调整输出
    if 'bg_prob_mean' in results:
        print(
            f"● 背景样本概率统计 - 均值: {results['bg_prob_mean']:.3f} ± {results['bg_prob_std']:.3f} | 中位数: {results['bg_median']:.3f}")

    # 打印测试集结果
    if 'test_accuracy' in results:
        print(f"\n🧪 测试集评估结果（纯灾害样本）:")
        print(f"● AD_AUC分数: {results['ad_auc_score_test']:.4f}")
        print(f"● 分类准确率: {results['test_accuracy']:.2%}")
        print(f"● 测试样本数: {results['test_size']}")

    print(f"\n📊 风险区域分析结果:")
    print("=" * 80)
    print(f"{'风险区域':<12} {'阈值范围':<20} {'灾害点数量':<10} {'灾害点比例':<12} {'背景样本比例':<12}")
    print("-" * 80)

    for risk_name, risk_info in results['risk_zones'].items():
        print(f"{risk_name:<12} {risk_info['阈值范围']:<20} {risk_info['灾害点数量']:<10} "
              f"{risk_info['灾害点比例']:<12.1%} {risk_info['背景样本比例']:<12.1%}")

    # 特征重要性（如果可用）
    if 'feature_importance' in results:
        print(f"\n📊 Top 5重要特征:")
        for i, row in results['feature_importance'].head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    print(f"● 结果保存路径: {os.path.abspath(output_dir)}")

    # 空间预测输出
    if Config.EXPORT_TIFF:
        feature_names = df_pos.columns.tolist()
        feature_mapping = {
            feature: os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif")
            for feature in feature_names
            if os.path.exists(os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif"))
        }

        processor = RasterProcessor(model, feature_mapping)
        prob_tif_path = os.path.join(output_dir, "susceptibility_probability.tif")
        processor.predict_to_raster(prob_tif_path)
        print(f"✅ 空间概率分布已保存至: {os.path.abspath(prob_tif_path)}")

        # 风险分区
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

    time_str = f"{total_time:.2f}秒"
    print(f"\n🎉 单分类SVM程序执行完成！总运行时间: {time_str}")
    print("=" * 60)