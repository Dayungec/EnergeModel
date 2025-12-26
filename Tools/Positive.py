import pandas as pd
from typing import Tuple
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.metrics import auc  # 保留auc用于计算曲线下面积

# ----------------------------
# 1. 数据加载（增加测试数据加载）
# ----------------------------
from EnergeModel.Tools import Config, DataReader
from EnergeModel.Tools.ShapAnalyse import ShapAnalyse

def load_data(pos_path: str, bg_path: str, test_path: str = None) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """加载正负样本数据、背景样本数据和测试数据"""
    df_pos = DataReader.load_data(pos_path)
    pos_columns_order = df_pos.columns.tolist()

    df_bg = DataReader.load_data(bg_path)

    # 验证背景数据是否包含所有必需的列
    missing_in_bg = set(pos_columns_order) - set(df_bg.columns)
    if missing_in_bg:
        raise ValueError(f"背景数据缺失以下列: {missing_in_bg}")

    df_bg = df_bg.reindex(columns=pos_columns_order)

    df_test = None
    if test_path and os.path.exists(test_path):
        df_test = DataReader.load_data(test_path)

        # 验证测试数据是否包含所有必需的列
        missing_in_test = set(pos_columns_order) - set(df_test.columns)
        if missing_in_test:
            print(f"警告: 测试数据缺失以下列，将填充NaN: {missing_in_test}")

        df_test = df_test.reindex(columns=pos_columns_order)
        df_test = df_test.astype(np.float32)

    df_pos = df_pos.astype(np.float32)
    df_bg = df_bg.astype(np.float32)

    return df_pos, df_bg, df_test, pos_columns_order


def calculate_pq_curve(positive_probs, background_probs, n_thresholds=100):
    """
    正确计算P-Q曲线和AD-AUC面积
    P = 正样本中预测为正的比例
    Q = 背景样本中预测为正的比例
    """
    # 生成阈值从0到1（从高到低）
    thresholds = np.linspace(1, 0, n_thresholds)  # 从1到0，确保Q从0到1
    p_values = []  # 预测精度
    q_values = []  # 预测密度

    for t in thresholds:
        # 计算预测精度P：正样本中预测为正的比例
        pos_predicted = (positive_probs >= t).astype(int)
        p = np.mean(pos_predicted) if len(positive_probs) > 0 else 0

        # 计算预测密度Q：背景样本中预测为正的比例
        bg_predicted = (background_probs >= t).astype(int)
        q = np.mean(bg_predicted) if len(background_probs) > 0 else 0

        p_values.append(p)
        q_values.append(q)

    # 将列表转换为numpy数组
    q_values = np.array(q_values)
    p_values = np.array(p_values)

    # 计算曲线下面积（使用梯形积分）
    # 注意：Q是横轴，P是纵轴
    ad_auc = auc(q_values, p_values)

    return thresholds, q_values, p_values, ad_auc

# 使用基于实际分布的百分位数阈值
def get_realistic_thresholds(df_bg):
    """根据实际概率分布设置合理阈值"""

    pos_probs = df_bg["原始概率"].values
    pos_sorted = np.sort(pos_probs)[::-1]  # 降序

    # 使用百分位数，确保每个区域都有合理的灾害点分布
    thresholds = [
        np.percentile(pos_sorted, 80),  # 高风险区
        np.percentile(pos_sorted, 60),  # 中高风险区
        np.percentile(pos_sorted, 40),  # 中风险区
        np.percentile(pos_sorted, 20),  # 中低风险区
    ]

    print(f"实际使用的阈值: {[f'{t:.3f}' for t in thresholds]}")
    return thresholds

def calculate_risk_zones(model, df_bg, df_pos, feature_names,
                                  thresholds=[0.8, 0.6, 0.4,0.2]):
    """结合固定阈值和灾害点比例的风险区域划分"""

    pos_path = Config.BASE_DIR + "positive_all.xlsx"
    df_pos = DataReader.load_data(pos_path)

    # 预测概率
    X_bg = df_bg[feature_names].values.astype(np.float32)
    df_bg["预测概率"] = model.predict_proba(X_bg)

    X_pos = df_pos[feature_names].values.astype(np.float32)
    df_pos["预测概率"] = model.predict_proba(X_pos)

    pos_probs = df_pos["预测概率"].values

    # 确保阈值递减
    for i in range(1, len(thresholds)):
        if thresholds[i] > thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] - 0.001

    # 定义风险区域
    risk_names = ['高风险区', '中高风险区', '中风险区', '中低风险区', '低风险区']

    risk_results = {}

    # 计算前4个风险区域
    for i in range(len(risk_names) - 1):
        risk_name = risk_names[i]

        if i == 0:
            lower_threshold = thresholds[0]
            upper_threshold = 1.0
            pos_in_zone = df_pos[df_pos["预测概率"] >= lower_threshold]
            bg_in_zone = df_bg[df_bg["预测概率"] >= lower_threshold]
        else:
            lower_threshold = thresholds[i]
            upper_threshold = thresholds[i - 1]
            pos_in_zone = df_pos[(df_pos["预测概率"] >= lower_threshold) &
                                 (df_pos["预测概率"] < upper_threshold)]
            bg_in_zone = df_bg[(df_bg["预测概率"] >= lower_threshold) &
                               (df_bg["预测概率"] < upper_threshold)]

        # 计算比例
        n_pos_in_zone = len(pos_in_zone)
        n_bg_in_zone = len(bg_in_zone)
        total_pos = len(df_pos)
        total_bg = len(df_bg)

        disaster_ratio = n_pos_in_zone / total_pos if total_pos > 0 else 0
        bg_ratio = n_bg_in_zone / total_bg if total_bg > 0 else 0

        risk_results[risk_name] = {
            '阈值范围': f"{lower_threshold:.4f} - {upper_threshold:.4f}",
            '下限阈值': lower_threshold,
            '上限阈值': upper_threshold,
            '灾害点数量': n_pos_in_zone,
            '灾害点比例': disaster_ratio,
            '背景样本数量': n_bg_in_zone,
            '背景样本比例': bg_ratio
        }

    # 计算极低风险区
    risk_name = '极低风险区'
    lower_threshold = 0.0
    upper_threshold = thresholds[-1]

    pos_in_zone = df_pos[df_pos["预测概率"] < upper_threshold]
    bg_in_zone = df_bg[df_bg["预测概率"] < upper_threshold]

    n_pos_in_zone = len(pos_in_zone)
    n_bg_in_zone = len(bg_in_zone)

    disaster_ratio = n_pos_in_zone / total_pos if total_pos > 0 else 0
    bg_ratio = n_bg_in_zone / total_bg if total_bg > 0 else 0

    risk_results[risk_name] = {
        '阈值范围': f"{lower_threshold:.4f} - {upper_threshold:.4f}",
        '下限阈值': lower_threshold,
        '上限阈值': upper_threshold,
        '灾害点数量': n_pos_in_zone,
        '灾害点比例': disaster_ratio,
        '背景样本数量': n_bg_in_zone,
        '背景样本比例': bg_ratio
    }

    return risk_results, df_bg, df_pos

def evaluate_test_set(model, df_test,df_bg, feature_names):
    if df_test is None:
        return None

    # 提取特征数据
    X_test = df_test[feature_names].values.astype(np.float32)
    y_test = np.ones(len(X_test))

    # 获取预测概率
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    positive_probs = df_test["原始概率"].values
    background_probs = df_bg["原始概率"].values
    # 计算P-Q曲线
    _, _, _, ad_auc = calculate_pq_curve(
        positive_probs, background_probs
    )

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)  # 等价于accuracy_score

    return {
        "ad_auc_score_test": ad_auc,
        'test_accuracy': float(accuracy),  # 转换为Python原生float类型
        'test_size': len(X_test)
    }

def train_and_evaluate(
        model,
        pos_path: str,
        bg_path: str,
        test_path: str = None,  # 新增测试集路径
        output_dir: str = "results"
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # 数据加载（包含测试数据）
    df_pos, df_bg, df_test, feature_names = load_data(pos_path, bg_path, test_path)

    # 转换为numpy数组用于训练
    X_pos = df_pos.values.astype(np.float32)
    X_bg = df_bg.values.astype(np.float32)
    X_test = df_test.values.astype(np.float32)

    print(f"正样本数量: {len(X_pos)}")
    print(f"背景样本数量: {len(df_bg)}")
    if df_test is not None:
        print(f"测试样本数量: {len(df_test)}")
    print(f"特征维度: {X_pos.shape[1]}")
    print(f"特征名: {feature_names}")

    # 创建并训练模型
    print("🚀 开始训练模型...")
    model.fit(X_pos,X_bg, feature_names)

    # 评估模型（增加测试集评估）
    results = evaluate_model(model, X_pos, X_bg,X_test,df_pos, df_bg, df_test, feature_names, output_dir)

    if os.path.exists(Config.BASE_DIR + "test2.xlsx"):
        df_test2 = DataReader.load_data(Config.BASE_DIR + "test2.xlsx")
        X_test2 = df_test2.values.astype(np.float32)
        df_test2["原始概率"] = model.predict_proba(X_test2)
        df_test2.to_excel(f"{output_dir}/test2_prob.xlsx", index=False)

    return results


def evaluate_model(
        model,
        X_pos: np.ndarray,
        X_bg: np.ndarray,
        X_test: np.ndarray,
        df_pos: pd.DataFrame,
        df_bg: pd.DataFrame,
        df_test: pd.DataFrame,  # 新增测试集
        feature_names: list,
        output_dir: str,
        calcu_zone=True
) -> dict:
    """模型评估（增加测试集评估）"""
    # 预测概率
    df_pos["原始概率"] = model.predict_proba(X_pos)
    df_bg["原始概率"] = model.predict_proba(X_bg)
    df_test["原始概率"] = model.predict_proba(X_test)

    # 计算训练集AUC
    positive_probs = df_pos["原始概率"].values
    background_probs = df_bg["原始概率"].values
    # 计算P-Q曲线
    thresholds, q_values, p_values, ad_auc = calculate_pq_curve(
        positive_probs, background_probs
    )

    print(f"📈 P-Q曲线结果:")
    print(f"   - AD-AUC面积: {ad_auc:.4f}")
    print(f"   - P值范围: {p_values.min():.3f} ~ {p_values.max():.3f}")
    print(f"   - Q值范围: {q_values.min():.3f} ~ {q_values.max():.3f}")

    # 保存P-Q曲线数据到Excel
    pq_data = pd.DataFrame({
        'Threshold': thresholds,
        'Prediction_Density_Q': q_values,
        'Prediction_Accuracy_P': p_values
    })
    pq_data.to_excel(f"{output_dir}/pq_curve_data.xlsx", index=False)


    y_pred = (df_pos["原始概率"] >= 0.5).astype(int)
    y_test = np.ones(len(df_pos["原始概率"]))
    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    m_pred = (df_bg["原始概率"] >= 0.5).astype(int)
    m_test = np.ones(len(df_bg["原始概率"]))
    density=np.mean(m_pred == m_test)

    if calcu_zone:
        thresholds=get_realistic_thresholds(df_bg)
        # 划分风险区域
        risk_results, df_bg_with_prob, df_pos_with_prob = calculate_risk_zones(model, df_bg, df_pos, feature_names,thresholds)

    # 特征重要性
    feature_importance = model.get_feature_importance()
    feature_importance_df =None
    if feature_importance is not None:
       feature_importance_df = pd.DataFrame({
          'feature': feature_names,
          'importance': feature_importance
        }).sort_values('importance', ascending=False)

    # 测试集评估（新增）
    test_metrics = evaluate_test_set(model, df_test,df_bg, feature_names)

    # 保存结果
    if calcu_zone:
        df_pos_with_prob.to_excel(f"{output_dir}/positive_with_prob.xlsx", index=False)
        df_bg_with_prob.to_excel(f"{output_dir}/background_with_prob.xlsx", index=False)
    if feature_importance_df is not None:
        feature_importance_df.to_excel(f"{output_dir}/feature_importance.xlsx", index=False)

    # 保存测试集结果
    if calcu_zone:
        if df_test is not None:
            df_test_result = df_test.copy()
            X_test = df_test[feature_names].values.astype(np.float32)
            df_test_result["预测概率"] = model.predict_proba(X_test)
            df_test_result.to_excel(f"{output_dir}/test_set_predictions.xlsx", index=False)

    # 保存风险区域结果
    if calcu_zone:
        risk_df = pd.DataFrame(risk_results).T
        risk_df.to_excel(f"{output_dir}/risk_zone_analysis.xlsx")

    # 计算基本指标
    if calcu_zone:
        metrics = {
            "ad_auc_score": ad_auc,
            "pos_prob_mean": df_pos["原始概率"].mean(),
            "bg_prob_mean": df_bg["原始概率"].mean(),
            "pos_prob_std": df_pos["原始概率"].std(),
            "bg_prob_std": df_bg["原始概率"].std(),
            "pos_median": np.median(df_pos["原始概率"]),
            "bg_median": np.median(df_bg["原始概率"]),
            'train_accuracy': float(accuracy),
            'train_size': len(df_pos["原始概率"]),
            'train_density': float(density),
            "feature_importance": feature_importance_df,
            "risk_zones": risk_results,
        }
    else:
        metrics = {
            "ad_auc_score": ad_auc,
            "pos_prob_mean": df_pos["原始概率"].mean(),
            "bg_prob_mean": df_bg["原始概率"].mean(),
            "pos_prob_std": df_pos["原始概率"].std(),
            "bg_prob_std": df_bg["原始概率"].std(),
            "pos_median": np.median(df_pos["原始概率"]),
            "bg_median": np.median(df_bg["原始概率"]),
            'train_accuracy': float(accuracy),
            'train_size': len(df_pos["原始概率"]),
            'train_density': float(density),
            "feature_importance": feature_importance_df,
        }
    # 合并测试集指标
    if test_metrics:
        metrics.update(test_metrics)
    if Config.SHAP_ANA == 1 :
        # SHAP分析
        try:
            shap_analyzer = ShapAnalyse(model, feature_names)
            shap_analyzer.Analyse(df_pos, df_bg, feature_importance_df, metrics, output_dir)
        except Exception as e:
            print(f"SHAP分析跳过: {e}")

    # 可视化（增加测试集结果展示）
    if calcu_zone:
        create_visualizations(model,df_pos, df_bg, metrics, feature_importance_df, risk_results, output_dir, df_test)
    return metrics


# ----------------------------
# 7. 可视化函数（增加测试集展示）
# ----------------------------
def create_visualizations(model,df_pos,df_bg, metrics, feature_importance_df, risk_results, output_dir,
                                  df_test=None):
    """创建PU版本的可视化（增加测试集结果）"""
    if df_test is not None:
        plt.figure(figsize=(25, 15))
        n_subplots = 7
    else:
        plt.figure(figsize=(20, 15))
        n_subplots = 6

    # 1. 概率分布图
    plt.subplot(2, 4, 1)
    plt.hist(df_pos["原始概率"], bins=50, alpha=0.5, label=f"训练正样本 (n={len(df_pos)})", density=True)
    plt.hist(df_bg["原始概率"], bins=50, alpha=0.5, label=f"训练背景样本 (n={len(df_bg)})", density=True)


    # 添加风险区域阈值线
    colors = ['red', 'orange', 'yellow', 'lightblue', 'blue']
    risk_names = list(risk_results.keys())

    for i, risk_name in enumerate(risk_names):
        lower_threshold = risk_results[risk_name]['下限阈值']
        plt.axvline(lower_threshold, color=colors[i], linestyle='--', alpha=0.7, label=f'{risk_name}下限')

    plt.xlabel("预测概率", fontsize=12)
    plt.ylabel("密度", fontsize=12)
    title = f"概率分布 (AUC={metrics['ad_auc_score']:.3f})"
    plt.title(title, fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)


    # 2. 特征重要性
    if feature_importance_df is not None:
        plt.subplot(2, 4, 2)
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title('PU Top特征重要性')
        plt.gca().invert_yaxis()

        # 3. P-Q曲线（不标记最优点）
    plt.subplot(2, 4, 3)

    # 重新计算P-Q曲线数据
    positive_probs = df_pos["原始概率"].values
    background_probs = df_bg["原始概率"].values
    thresholds, q_values, p_values, ad_auc = calculate_pq_curve(positive_probs, background_probs)

    plt.plot(q_values, p_values, 'b-', linewidth=2, label=f'P-Q Curve (AD-AUC = {ad_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='随机基线')

    plt.xlabel('预测密度 (Q)', fontsize=12)
    plt.ylabel('预测精度 (P)', fontsize=12)
    plt.title('P-Q曲线: 精度 vs 密度', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    # 添加解释性文本（不涉及最优阈值）
    plt.text(0.05, 0.95, '左上角: 高精度低密度\n右下角: 低精度高密度',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 4. 风险区域灾害点比例
    plt.subplot(2, 4, 4)
    disaster_ratios = [risk_results[name]['灾害点比例'] for name in risk_names]
    bars = plt.bar(risk_names, disaster_ratios, color=colors)
    plt.ylabel('灾害点比例')
    plt.title('各风险区域灾害点比例')
    plt.xticks(rotation=45)
    for bar, ratio in zip(bars, disaster_ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{ratio:.1%}', ha='center', va='bottom')

    # 6. 风险区域分布饼图
    plt.subplot(2, 4, 6)
    bg_ratios = [max(risk_results[name]['背景样本比例'], 0) for name in risk_names]
    valid_indices = [i for i, ratio in enumerate(bg_ratios) if ratio > 0]
    if valid_indices:
        valid_ratios = [bg_ratios[i] for i in valid_indices]
        valid_labels = [risk_names[i] for i in valid_indices]
        valid_colors = [colors[i] for i in valid_indices]
        plt.pie(valid_ratios, labels=valid_labels, autopct='%1.1f%%', colors=valid_colors)
        plt.title('风险区域分布比例')
    else:
        plt.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('风险区域分布比例（无数据）')

    # 7. 概率箱线图
    plt.subplot(2, 4, 7)
    prob_data = [df_pos["原始概率"], df_bg["原始概率"]]
    if df_test is not None and '预测概率' in df_test.columns:
        test_pos_probs = df_test[df_test.iloc[:, -1] == 1]["预测概率"] if len(df_test[df_test.iloc[:, -1] == 1]) > 0 else []
        test_neg_probs = df_test[df_test.iloc[:, -1] == 0]["预测概率"] if len(df_test[df_test.iloc[:, -1] == 0]) > 0 else []
        if len(test_pos_probs) > 0 and len(test_neg_probs) > 0:
            prob_data.extend([test_pos_probs, test_neg_probs])
            labels = ['训练正样本', '训练背景样本', '测试正样本', '测试负样本']
        else:
            labels = ['训练正样本', '训练背景样本']
    else:
        labels = ['训练正样本', '训练背景样本']

    plt.boxplot(prob_data, labels=labels)
    plt.ylabel('易发性概率')
    plt.title('概率分布箱线图')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

