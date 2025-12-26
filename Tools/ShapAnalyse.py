import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Union, Optional
from catboost import CatBoostClassifier


class ShapAnalyse:
    def __init__(self, model, feature_names: list):
        """
        SHAP分析器初始化
        参数:
            model: 已训练的CatBoost模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """数据预处理"""
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].values
        return X.astype(np.float32) if X.dtype != np.float32 else X

    def create_explainer(self, background_data: Optional[np.ndarray] = None):
        """创建SHAP解释器"""
        if background_data is None:
            self.explainer = shap.Explainer(self.model)
        else:
            self.explainer = shap.Explainer(
                self.model,
                shap.sample(background_data, 100)  # 使用100个样本作为背景
            )

    def analyze_global(self, X: Union[pd.DataFrame, np.ndarray], output_dir: str) -> shap.Explanation:
        """
        修改返回类型为Explanation对象
        返回: SHAP Explanation对象
        """
        os.makedirs(output_dir, exist_ok=True)
        X_data = self._prepare_data(X)

        if self.explainer is None:
            self.create_explainer()

        shap_values = self.explainer(X_data)  # 已经是Explanation对象

        # 保存原始值备用
        pd.DataFrame(shap_values.values, columns=self.feature_names).to_csv(
            f"{output_dir}/shap_values.csv", index=False)

        return shap_values  # 直接返回Explanation对象

    def analyze_feature_dependence(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            shap_values: Union[shap.Explanation, np.ndarray],
            features_to_plot: list,
            output_dir: str
    ):
        """绘制指定特征的依赖图（兼容Explanation对象和数值数组）"""
        os.makedirs(output_dir, exist_ok=True)

        # 准备特征数据
        X_data = self._prepare_data(X)

        # 转换SHAP值为数值数组
        if isinstance(shap_values, shap.Explanation):
            shap_values = shap_values.values
        elif not isinstance(shap_values, np.ndarray):
            raise TypeError("shap_values必须是Explanation对象或numpy数组")

        # 绘制每个特征的依赖图
        for feat in features_to_plot:
            if feat not in self.feature_names:
                continue

            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    ind=feat,
                    shap_values=shap_values,
                    features=X_data,
                    feature_names=self.feature_names,
                    interaction_index=None,
                    show=False
                )
                plt.title(f"'{feat}'特征依赖关系", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/dependence_{feat}.png", dpi=150)
                plt.close()
            except Exception as e:
                print(f"特征'{feat}'依赖图生成失败: {str(e)}")

    def analyze_samples(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            shap_explanation: shap.Explanation,  # 修改参数类型
            sample_indices: list,
            output_dir: str,
            max_display: int = 10
    ):
        """样本级解释分析（使用Explanation对象）"""
        os.makedirs(output_dir, exist_ok=True)
        X_data = self._prepare_data(X)

        for idx in sample_indices:
            if idx >= len(shap_explanation):
                continue

            plt.figure()
            shap.plots.waterfall(
                shap_explanation[idx],
                max_display=max_display,
                show=False
            )
            plt.title(f"样本 {idx} 预测解释")
            plt.savefig(
                f"{output_dir}/sample_{idx}_explanation.png",
                dpi=150, bbox_inches='tight'
            )
            plt.close()

    @staticmethod
    def compare_analysis(
            explanation1: shap.Explanation,
            explanation2: shap.Explanation,
            feature_names: list,
            label1: str = "Group1",
            label2: str = "Group2",
            output_dir: str = "shap_comparison"
    ) -> pd.DataFrame:
        """安全对比两组SHAP结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 正确获取SHAP值（使用.values属性）
        if isinstance(explanation1, shap.Explanation):
            values1 = explanation1.values
        else:
            values1 = explanation1

        if isinstance(explanation2, shap.Explanation):
            values2 = explanation2.values
        else:
            values2 = explanation2

        # 计算平均重要性（确保是numpy数组）
        imp1 = np.abs(values1).mean(axis=0) if isinstance(values1, np.ndarray) else np.nan
        imp2 = np.abs(values2).mean(axis=0) if isinstance(values2, np.ndarray) else np.nan

        # 创建对比表
        comparison = pd.DataFrame({
            'feature': feature_names,
            f'{label1}_importance': imp1,
            f'{label2}_importance': imp2,
            'importance_diff': imp1 - imp2
        }).sort_values('importance_diff', ascending=False)

        # 保存结果
        comparison.to_excel(f"{output_dir}/comparison.xlsx", index=False)

        # 绘制对比图
        plt.figure(figsize=(10, 8))
        comparison.head(20).plot(
            x='feature',
            y=[f'{label1}_importance', f'{label2}_importance'],
            kind='barh',
            title=f'{label1} vs {label2} 特征重要性对比'
        )
        plt.xlabel('平均|SHAP值|')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/importance_comparison.png", dpi=300)
        plt.close()

        return comparison

    def Analyse(self,df_pos,df_bg,feature_importance_df,metrics,output_dir):
        print("\n🔍 开始SHAP可解释性分析...")
        try:

            # 使用背景样本作为参考分布
            self.create_explainer(background_data=df_bg[self.feature_names].values)

            # 1. 正样本全局分析
            pos_shap_dir = f"{output_dir}/shap_positive"
            pos_shap_values = self.analyze_global(
                df_pos[self.feature_names],
                output_dir=pos_shap_dir
            )

            # 2. 高风险背景样本分析
            high_risk_bg = df_bg[df_bg["预测概率"] > 0.7]
            if len(high_risk_bg) > 0:
                bg_shap_dir = f"{output_dir}/shap_highrisk_bg"
                bg_shap_values = self.analyze_global(
                    high_risk_bg[self.feature_names],
                    output_dir=bg_shap_dir
                )

                # 3. 对比分析
                comparison_dir = f"{output_dir}/shap_comparison"
                comparison = ShapAnalyse.compare_analysis(
                    pos_shap_values,
                    bg_shap_values,
                    self.feature_names,
                    label1="正样本",
                    label2="高风险背景",
                    output_dir=comparison_dir
                )

                # 将对比结果添加到metrics中
                metrics["shap_comparison"] = comparison

            # 4. 特征依赖分析（选择重要性前3的特征）
            if feature_importance_df is not None:
                top_features = feature_importance_df.head(3)['feature'].tolist()
                self.analyze_feature_dependence(
                    X=df_pos[self.feature_names],
                    shap_values=pos_shap_values,  # 传入完整Explanation对象
                    features_to_plot=top_features,
                    output_dir=f"{output_dir}/shap_positive/dependence"
                )

            # 5. 样本级解释（分析前5个样本）
            self.analyze_samples(
                df_pos[self.feature_names],
                pos_shap_values,
                sample_indices=range(5),
                output_dir=f"{pos_shap_dir}/samples"
            )

        except Exception as e:
            print(f"SHAP分析失败: {str(e)}")
