import random
import time

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

import warnings
import Tools.Positive

# 设置中文字体
from EnergeModel.Tools import Config, DataReader
from EnergeModel.Tools.RasterProcessor import RasterProcessor

from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    """固定所有随机种子以确保结果可重现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置PyTorch Lightning的随机种子
    pl.seed_everything(seed, workers=True)

def reset_seed(seed=42):
    """重置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    # 设置环境变量和CuDNN
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或 ':16:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 对于PyTorch 1.7+
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

# ----------------------------
# 数据集
# ----------------------------
class PositiveDataset(Dataset):
    def __init__(self, X_pos: torch.Tensor, X_bg: torch.Tensor):
        self.X_pos = X_pos.cpu()
        self.X_bg = X_bg.cpu()

    def __len__(self):
        return 1  # 全批量训练

    def __getitem__(self, idx):
        return self.X_pos, self.X_bg


class PEMModel(pl.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.save_hyperparameters()


        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1, bias=False)
        )



        # 动态参数
        self.margin = 2.0
        self.temperature = 2.0

        self.l2w=0.08
        self.adjustment_rate=0.05

        self.training_history = {
            'epoch': [],
            'margin': [],
            'temperature': [],
            'e_pos_mean': [],
            'e_bg_mean': [],
            'e_bg_std': []
        }

        # 初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    def forward(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.predict_proba(x)

    def predict_proba(self, x):
        """预测概率（自动处理设备问题）"""
        with torch.no_grad():
            # 确保输入数据在CPU上
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x = x.cpu()
                x = x.numpy()
            elif isinstance(x, np.ndarray):
                x = x.astype(np.float32)
            else:

                raise TypeError("输入必须是numpy数组或PyTorch张量")
            # 转换为PyTorch张量并放到模型设备上
            x_tensor = torch.FloatTensor(x).to(self.device)
            energy = self.encoder(x_tensor)
            probs = torch.sigmoid(-energy / self.temperature).cpu().numpy()
            return probs.flatten()


    def training_step(self, batch, batch_idx):
        x_pos, x_bg = batch
        x_pos = x_pos.to(self.device)
        x_bg = x_bg.to(self.device)
        e_pos = self.forward(x_pos)
        e_bg = self.forward(x_bg)

        scaled_e_pos = e_pos / self.temperature
        scaled_e_bg = e_bg / self.temperature

        prob_pos = torch.sigmoid(-scaled_e_pos)
        prob_bg = torch.sigmoid(-scaled_e_bg)

        pos_mean = scaled_e_pos.mean()
        bg_mean = scaled_e_bg.mean()

        contrast_loss = F.softplus((pos_mean - bg_mean + self.margin) * 0.5)
        energy_reg = self.l2w * (scaled_e_pos.pow(2).mean() + scaled_e_bg.pow(2).mean())
        loss = contrast_loss  + energy_reg

        # 动态调整参数

        if self.current_epoch % 10 == 0:
            self.training_history['epoch'].append(self.current_epoch)
            self.training_history['margin'].append(self.margin)
            self.training_history['temperature'].append(self.temperature)
            self.training_history['e_pos_mean'].append(e_pos.mean().item())
            self.training_history['e_bg_mean'].append(e_bg.mean().item())
            self.training_history['e_bg_std'].append(e_bg.std().item())

        self.log_dict({
            'train_loss': loss,
            'pos_prob': prob_pos.mean(),
            'bg_prob': prob_bg.mean(),
            'bg_prob_std': prob_bg.std(),
            'margin': self.margin,
            'temperature': self.temperature,
        }, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """在训练epoch结束后进行参数调整，避免与梯度计算冲突"""
        if self.current_epoch % 20 == 0:
            # 获取当前batch的统计量（需要记录或重新计算）
            # 这里需要修改：我们需要在training_step中记录必要的统计量
            pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            },
            "gradient_clip_val": 0.5
        }

    def save_training_history(self, output_dir: str):
        """保存训练历史数据到Excel文件"""
        os.makedirs(output_dir, exist_ok=True)
        history_file = os.path.join(output_dir, "training_history.xlsx")

        # 创建DataFrame
        df_history = pd.DataFrame(self.training_history)

        # 保存到Excel
        df_history.to_excel(history_file, index=False)
        print(f"✅ 训练历史数据已保存至: {os.path.abspath(history_file)}")

        return df_history

    def get_feature_importance(self):
        return None

# ----------------------------
# 模型
# ----------------------------
class AutoPEMModel(pl.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1, bias=False)
        )
        # 动态参数
        self.margin = nn.Parameter(torch.tensor(2.0))
        self.temperature = nn.Parameter(torch.tensor(2.0))

        self.l2w=0.08
        self.adjustment_rate=0.05

        self.training_history = {
            'epoch': [],
            'margin': [],
            'temperature': [],
            'e_pos_mean': [],
            'e_bg_mean': [],
            'e_bg_std': []
        }

        # 初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)


    def forward(self, x):
        return self.encoder(x)

    def __call__(self, x):
        return self.predict_proba(x)

    def predict_proba(self, x):
        """预测概率（自动处理设备问题）"""
        with torch.no_grad():
            # 确保输入数据在CPU上
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x = x.cpu()
                x = x.numpy()
            elif isinstance(x, np.ndarray):
                x = x.astype(np.float32)
            else:
                raise TypeError("输入必须是numpy数组或PyTorch张量")

            # 转换为PyTorch张量并放到模型设备上
            x_tensor = torch.FloatTensor(x).to(self.device)
            energy = self.encoder(x_tensor)
            probs = torch.sigmoid(-energy / self.temperature).cpu().numpy()
            return probs.flatten()

    def training_step(self, batch, batch_idx):
        x_pos, x_bg = batch
        x_pos = x_pos.to(self.device)
        x_bg = x_bg.to(self.device)
        e_pos = self.forward(x_pos)
        e_bg = self.forward(x_bg)

        scaled_e_pos = e_pos / self.temperature
        scaled_e_bg = e_bg / self.temperature

        prob_pos = torch.sigmoid(-scaled_e_pos)
        prob_bg = torch.sigmoid(-scaled_e_bg)

        pos_mean = scaled_e_pos.mean()
        bg_mean = scaled_e_bg.mean()

        contrast_loss = F.softplus((pos_mean - bg_mean + self.margin) * 0.5)
        energy_reg = self.l2w * (scaled_e_pos.pow(2).mean() + scaled_e_bg.pow(2).mean())
        loss = contrast_loss  + energy_reg

        if self.current_epoch % 20 == 0:
            self.training_history['epoch'].append(self.current_epoch)
            self.training_history['margin'].append(self.margin.item())
            self.training_history['temperature'].append(self.temperature.item())
            self.training_history['e_pos_mean'].append(e_pos.mean().item())
            self.training_history['e_bg_mean'].append(e_bg.mean().item())
            self.training_history['e_bg_std'].append(e_bg.std().item())

        self.log_dict({
            'train_loss': loss,
            'pos_prob': prob_pos.mean(),
            'bg_prob': prob_bg.mean(),
            'bg_prob_std': prob_bg.std(),
            'margin': self.margin,
            'temperature': self.temperature,
        }, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """在训练epoch结束后进行参数调整，避免与梯度计算冲突"""
        if self.current_epoch % 20 == 0:
            # 获取当前batch的统计量（需要记录或重新计算）
            # 这里需要修改：我们需要在training_step中记录必要的统计量
            pass

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=8, verbose=True)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            },
            "gradient_clip_val": 0.5
        }

    def save_training_history(self, output_dir: str):
        """保存训练历史数据到Excel文件"""
        os.makedirs(output_dir, exist_ok=True)
        history_file = os.path.join(output_dir, "training_history.xlsx")

        # 创建DataFrame
        df_history = pd.DataFrame(self.training_history)

        # 保存到Excel
        df_history.to_excel(history_file, index=False)
        print(f"✅ 训练历史数据已保存至: {os.path.abspath(history_file)}")

        return df_history

    def get_feature_importance(self):
        return None

class PEM_Shallow(PEMModel):
    """
    更浅的架构变体 (256-128)
    继承自 PEMModel，但使用更少的隐藏层
    """

    def __init__(self, input_dim: int):
        # 调用父类初始化，但随后会覆盖encoder
        super().__init__(input_dim)

        # 重写编码器为更浅的架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # 第一层：256个神经元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),  # 第二层：128个神经元
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1, bias=False)  # 输出层
        )

        # 重新初始化新编码器的权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder.apply(init_weights)

class PEM_Deep(PEMModel):
    """
    更深的架构变体 (1024-512-256-128)
    继承自 PEMModel，但增加网络深度
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)

        # 重写编码器为更深的架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),  # 第一层：1024个神经元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),  # 第二层：512个神经元
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),  # 第三层：256个神经元
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # 第四层：128个神经元
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1, bias=False)  # 输出层
        )

        # 重新初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder.apply(init_weights)

class PEM_Wide(PEMModel):
    """
    更宽但更浅的架构变体 (1024-1024)
    继承自 PEMModel，使用更宽的层但减少深度
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)

        # 重写编码器为更宽但更浅的架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),  # 第一层：1024个神经元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),  # 第二层：1024个神经元（保持宽度）
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1, bias=False)  # 输出层
        )

        # 重新初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder.apply(init_weights)

class PEM_Narrow(PEMModel):
    """
    更窄的瓶颈结构变体 (256-128-64)
    继承自 PEMModel，使用更窄的层构造瓶颈结构
    """

    def __init__(self, input_dim: int):
        super().__init__(input_dim)

        # 重写编码器为更窄的瓶颈架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # 第一层：256个神经元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),  # 第二层：128个神经元
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),  # 第三层：64个神经元（瓶颈层）
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1, bias=False)  # 输出层
        )

        # 重新初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder.apply(init_weights)

def train_and_evaluate(
        model,
        pos_path: str,
        bg_path: str,
        test_path: str = None,  # 新增测试集路径
        output_dir: str = "results",
        max_epochs=1000
) -> dict:
    """训练评估流程（使用背景样本划分风险区域）"""
    os.makedirs(output_dir, exist_ok=True)
    # 数据加载
    df_pos, df_bg, df_test, feature_names = Tools.Positive.load_data(pos_path, bg_path, test_path)

    # 转换为numpy数组用于训练
    X_pos = df_pos.values.astype(np.float32)
    X_bg = df_bg.values.astype(np.float32)
    X_pos = torch.FloatTensor(X_pos)
    X_bg = torch.FloatTensor(X_bg)
    X_test = df_test.values.astype(np.float32)

    print(f"正样本数量: {len(X_pos)}")
    print(f"背景样本数量: {len(df_bg)}")
    if df_test is not None:
        print(f"测试样本数量: {len(df_test)}")
    print(f"特征维度: {X_pos.shape[1]}")
    print(f"特征名: {feature_names}")

    # 创建并训练模型
    print("🚀 开始训练模型...")

    dataset = PositiveDataset(X_pos, X_bg)
    train_loader = DataLoader(dataset, batch_size=None, shuffle=False)

    # 回调配置
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="best_model",
        monitor="train_loss",
        mode="min",
        save_top_k=1
    )

    early_stop = EarlyStopping(
        monitor="train_loss",
        patience=30,
        mode="min",
        verbose=True
    )
    # 训练器配置
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint],
        default_root_dir=output_dir,
        deterministic=True,
        precision="16-mixed",
        accumulate_grad_batches=4,
        logger=True,
        enable_progress_bar=True
    )
    # 训练
    trainer.fit(model, train_loader)

    df_history = model.save_training_history(output_dir)

    # 评估
    best_model = model.load_from_checkpoint(
        checkpoint.best_model_path,
        input_dim=X_pos.shape[1]
    )
    # 评估模型（增加测试集评估）
    best_model.eval()
    with torch.no_grad():
        results = Tools.Positive.evaluate_model(best_model, X_pos, X_bg,X_test, df_pos, df_bg, df_test, feature_names, output_dir)

    results['training_history'] = df_history

    return best_model,results


def run_one():
    start_time = time.time()
    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh1800.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    output_dir = "result/pem_positive"

    # 运行训练评估流程
    print("🚀 开始训练正样本模型...")

    df_pos = DataReader.load_data(pos_path)
    model = PEMModel(input_dim=df_pos.shape[1])
    #model = AutoPEMModel(input_dim=df_pos.shape[1])
    model, results = train_and_evaluate(
        model,
        pos_path=pos_path,
        bg_path=bg_path,
        test_path=test_path,  # 传入测试集路径
        output_dir=output_dir,
        max_epochs=500
    )
    # 打印最终结果
    print("\n⭐ 最终评估结果 ⭐")
    print(f"● AD_AUC分数: {results['ad_auc_score']:.4f}")
    print(f"● 预测精度: {results['train_accuracy']:.4f}")
    print(f"● 预测密度: {results['train_density']:.4f}")
    print(
        f"● 正样本概率统计 - 均值: {results['pos_prob_mean']:.3f} ± {results['pos_prob_std']:.3f} | 中位数: {results['pos_median']:.3f}")
    print(
        f"● 背景样本概率统计 - 均值: {results['bg_prob_mean']:.3f} ± {results['bg_prob_std']:.3f} | 中位数: {results['bg_median']:.3f}")

    # 打印测试集结果
    if 'test_accuracy' in results:
        print(f"\n🧪 测试集评估结果（纯灾害样本）:")
        print(f"● AD_AUC分数: {results['ad_auc_score_test']:.4f}")
        print(f"● 分类准确率: {results['test_accuracy']:.2%}")  # 百分比格式更直观
        print(f"● 测试样本数: {results['test_size']}")

    print(f"\n📊 风险区域分析结果:")
    print("=" * 80)
    print(f"{'风险区域':<12} {'阈值范围':<20} {'灾害点数量':<10} {'灾害点比例':<12} {'背景样本比例':<12}")
    print("-" * 80)

    for risk_name, risk_info in results['risk_zones'].items():
        print(f"{risk_name:<12} {risk_info['阈值范围']:<20} {risk_info['灾害点数量']:<10} "
              f"{risk_info['灾害点比例']:<12.1%} {risk_info['背景样本比例']:<12.1%}")

    print(f"● 结果保存路径: {os.path.abspath(output_dir)}")

    if Config.EXPORT_TIFF:
        df_pos = DataReader.load_data(pos_path)
        feature_names = df_pos.columns.tolist()  # 假设df_pos已从pos_path加载
        # 构建特征映射字典（自动匹配entropies目录下的同名tif）
        feature_mapping = {
            feature: os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif")
            for feature in feature_names
            if os.path.exists(os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif"))
        }
        # 初始化处理器
        processor = RasterProcessor(model, feature_mapping)

        # 输出路径
        prob_tif_path = os.path.join(output_dir, "susceptibility_probability.tif")

        # 执行预测
        #processor.predict_to_raster(prob_tif_path)
        processor.predict_to_raster_with_filter(5,prob_tif_path)
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

def batch_training_experiment():
    """批量训练实验：对不同网格文件进行训练并记录指标（确定性计算，单次训练）"""
    start_time = time.time()

    # 配置不同的网格文件路径
    mesh_files = [
        "mesh600.xlsx", "mesh1200.xlsx", "mesh1800.xlsx", "mesh2400.xlsx",
        "mesh3000.xlsx", "mesh3600.xlsx", "mesh4200.xlsx", "mesh4800.xlsx",
        "mesh5400.xlsx", "mesh6000.xlsx"
    ]
    # 固定文件路径
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"

    # 存储所有实验结果
    all_results = []

    # 对每个网格文件进行1次训练（确定性计算）
    for mesh_file in mesh_files:
        reset_seed()

        bg_path = Config.BASE_DIR + mesh_file
        mesh_name = mesh_file.replace('.xlsx', '')

        print(f"\n{'=' * 60}")
        print(f"🔬 开始处理网格文件: {mesh_file}")
        print(f"{'=' * 60}")

        try:
            # 创建输出目录
            output_dir = f"result/pem_positive/pem_positive_{mesh_name}"
            os.makedirs(output_dir, exist_ok=True)

            # 加载正样本数据获取特征维度
            df_pos = DataReader.load_data(pos_path)
            model = PEMModel(input_dim=df_pos.shape[1])

            # 使用简化训练函数
            best_model, metrics = train_and_evaluate(
                model=model,
                pos_path=pos_path,
                bg_path=bg_path,
                test_path=test_path,  # 不使用测试集以加快训练
                output_dir=output_dir,
                max_epochs=1000  # 适当减少epochs以提高效率
            )

            # 保存关键指标
            result = {
                'mesh_file': mesh_file,
                'ad_auc_score': metrics['ad_auc_score'],
                'ad_auc_score_test': metrics['ad_auc_score_test'],
                'train_accuracy': metrics['train_accuracy'],
                'train_density': metrics['train_density']
            }

            all_results.append(result)

            print(f"✅ 训练完成 - AD_AUC: {metrics['ad_auc_score']:.4f}, "
                  f"准确率: {metrics['train_accuracy']:.4f}, 密度: {metrics['train_density']:.4f}")

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            # 记录失败信息
            result = {
                'mesh_file': mesh_file,
                'ad_auc_score': 0.0,
                'ad_auc_score_test': 0.0,
                'train_accuracy': 0.0,
                'train_density': 0.0,
                'mean_pos_prob': 0.0,
                'error': str(e)
            }
            all_results.append(result)

    # 保存结果到Excel文件
    results_df = pd.DataFrame(all_results)

    # 保存到Excel
    output_excel_path = "result/pem_positive/batch_training_results.xlsx"
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    results_df.to_excel(output_excel_path, index=False)

    end_time = time.time()
    total_time = end_time - start_time

    # 输出总结报告
    print(f"\n{'=' * 80}")
    print(f"🎉 批量训练实验完成!")
    print(f"{'=' * 80}")
    print(f"📊 实验概况:")
    print(f"● 处理的网格文件数量: {len(mesh_files)}")
    print(f"● 总运行时间: {total_time / 60:.2f} 分钟")
    print(f"● 结果文件: {output_excel_path}")

    # 统计成功训练的数量
    successful_runs = len([r for r in all_results if r.get('ad_auc_score', 0) > 0])
    print(f"● 成功训练次数: {successful_runs}/{len(mesh_files)}")

    print(f"\n🏆 性能最佳的前3个网格文件:")
    top_3 = results_df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows()):
        print(f"{i + 1}. {row['mesh_file']}: AD_AUC = {row['ad_auc_score']:.4f}")

    return results_df

def run_m_T():
    start_time = time.time()

    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    output_base_dir = "result/pem_positive"

    # 定义不同的 m 和 T 组合
    param_combinations = [
        (0.25, 16.0),  # 作为对比
        (1.0, 8.0),  # 作为对比
        (1.0, 4.0),  # 作为对比
        (2.0, 4.0),  # 作为对比
        (2.0, 2.0),  # 作为对比
        (4.0, 2.0),  # 作为对比
        (8.0, 1.0),  # 作为对比
        (16.0, 0.5),  # 作为对比
    ]
    # 存储所有结果
    all_results = []

    # 加载数据（只需要一次）
    df_pos, df_bg, df_test, feature_names = Tools.Positive.load_data(pos_path, bg_path, test_path)
    X_pos = df_pos.values.astype(np.float32)
    X_bg = df_bg.values.astype(np.float32)

    print(f"正样本数量: {len(X_pos)}")
    print(f"背景样本数量: {len(df_bg)}")
    if df_test is not None:
        print(f"测试样本数量: {len(df_test)}")
    print(f"特征维度: {X_pos.shape[1]}")
    print(f"参数组合数量: {len(param_combinations)}次实验")
    print("=" * 80)

    # 遍历所有参数组合（每个组合只训练1次）
    for i, (initial_margin, initial_temperature) in enumerate(param_combinations):
        reset_seed()
        print(f"\n🔬 正在训练参数组合 {i + 1}/{len(param_combinations)}: m={initial_margin}, T={initial_temperature}")

        # 为当前参数组合创建专门的输出目录
        param_output_dir = os.path.join(output_base_dir, f"m_{initial_margin}_T_{initial_temperature}")
        os.makedirs(param_output_dir, exist_ok=True)

        try:
            # 创建模型并设置初始参数
            model = PEMModel(input_dim=X_pos.shape[1])

            # 手动设置初始参数（覆盖初始化值）
            with torch.no_grad():
                model.margin = initial_margin
                model.temperature = initial_temperature

            # 训练模型
            trained_model, results = train_and_evaluate(
                model,
                pos_path=pos_path,
                bg_path=bg_path,
                test_path=test_path,
                output_dir=param_output_dir
            )

            # 收集结果
            run_result = {
                '参数组合': f'm={initial_margin}, T={initial_temperature}',
                'AD_AUC分数': results.get('ad_auc_score', 0),
                '测试AD_AUC分数': results.get('ad_auc_score_test', 0),
                '预测精度': results.get('train_accuracy', 0),
                '预测密度': results.get('train_density', 0),
                '正样本概率均值': results.get('pos_prob_mean', 0),
                '正样本概率标准差': results.get('pos_prob_std', 0),
                '正样本概率中位数': results.get('pos_median', 0),
                '背景样本概率均值': results.get('bg_prob_mean', 0),
                '背景样本概率标准差': results.get('bg_prob_std', 0),
                '背景样本概率中位数': results.get('bg_median', 0),
                '最终margin': model.margin if hasattr(model, 'margin') else initial_margin,
                '最终temperature': model.temperature if hasattr(model, 'temperature') else initial_temperature,
                '输出路径': param_output_dir
            }

            # 添加测试集结果（如果存在）
            if 'test_accuracy' in results:
                run_result['测试集准确率'] = results['test_accuracy']
                run_result['测试样本数'] = results['test_size']

            all_results.append(run_result)

            print(f"✅ 训练完成 - TEST-AD-AUC: {run_result['测试AD_AUC分数']:.4f}, "
                  f"精度: {run_result['预测精度']:.4f}, 密度: {run_result['预测密度']:.4f}")

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            # 记录失败信息
            failed_result = {
                '参数组合': f'm={initial_margin}, T={initial_temperature}',
                'AD_AUC分数': 0,
                '测试AD_AUC分数': 0,
                '预测精度': 0,
                '预测密度': 0,
                '正样本概率均值': 0,
                '正样本概率标准差': 0,
                '正样本概率中位数': 0,
                '背景样本概率均值': 0,
                '背景样本概率标准差': 0,
                '背景样本概率中位数': 0,
                '最终margin': initial_margin,
                '最终temperature': initial_temperature,
                '输出路径': param_output_dir,
                '状态': f'失败: {e}'
            }
            all_results.append(failed_result)

    # 保存所有结果到Excel
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(output_base_dir, "all_parameter_results.xlsx")
        results_df.to_excel(results_file, index=False)

        print(f"\n✅ 所有实验结果已保存至: {os.path.abspath(results_file)}")

        # 显示总体统计信息
        print(f"\n📈 参数调优实验结果汇总:")
        print("=" * 100)

        # 只显示成功的实验
        successful_results = [r for r in all_results if r.get('测试AD_AUC分数', 0) > 0]

        if successful_results:
            # 按AD-AUC分数降序排列
            successful_results.sort(key=lambda x: x['测试AD_AUC分数'], reverse=True)

            print(
                f"{'排名':<4} {'参数组合':<20} {'TEST-AD-AUC':<8} {'预测精度':<8} {'预测密度':<8} {'最终margin':<10} {'最终temperature':<12}")
            print("-" * 100)

            for i, result in enumerate(successful_results, 1):
                print(f"{i:<4} {result['参数组合']:<20} {result['测试AD_AUC分数']:.4f}   {result['预测精度']:.4f}    "
                      f"{result['预测密度']:.4f}    {result['最终margin']:.4f}      {result['最终temperature']:.4f}")

            # 找出最佳参数组合
            best_result = successful_results[0]
            print(f"\n🏆 最佳参数组合: {best_result['参数组合']}")
            print(f"   测试AD_AUC分数: {best_result['测试AD_AUC分数']:.4f}")
            print(f"   预测精度: {best_result['预测精度']:.4f}")
            print(f"   预测密度: {best_result['预测密度']:.4f}")
            print(f"   最终margin值: {best_result['最终margin']:.4f}")
            print(f"   最终temperature值: {best_result['最终temperature']:.4f}")

    # 空间预测（使用最佳参数）
    if Config.EXPORT_TIFF and all_results:
        # 找出最佳结果
        successful_results = [r for r in all_results if r.get('AD_AUC分数', 0) > 0]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['AD_AUC分数'])
            best_model_path = os.path.join(best_result['输出路径'], "best_model.ckpt")

            if os.path.exists(best_model_path):
                print(f"\n🗺️  使用最佳参数组合 {best_result['参数组合']} 进行空间预测...")

                # 加载最佳模型
                best_model = PEMModel.load_from_checkpoint(
                    best_model_path,
                    input_dim=X_pos.shape[1]
                )

                # 构建特征映射
                df_pos = DataReader.load_data(pos_path)
                feature_names = df_pos.columns.tolist()
                feature_mapping = {
                    feature: os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif")
                    for feature in feature_names
                    if os.path.exists(os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif"))
                }

                # 初始化处理器并执行预测
                processor = RasterProcessor(best_model, feature_mapping)
                prob_tif_path = os.path.join(output_base_dir, "best_susceptibility_probability.tif")
                processor.predict_to_raster(prob_tif_path)
                print(f"✅ 最佳模型空间概率分布已保存至: {os.path.abspath(prob_tif_path)}")

    end_time = time.time()
    total_time = end_time - start_time

    # 格式化显示运行时间
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    if hours > 0:
        time_str = f"{hours}小时{minutes}分钟{seconds:.1f}秒"
    elif minutes > 0:
        time_str = f"{minutes}分钟{seconds:.1f}秒"
    else:
        time_str = f"{seconds:.1f}秒"

    print(f"\n🎉 参数调优实验完成！总运行时间: {time_str}")
    print("=" * 80)

    return all_results

def run_Auto_m_T():
    start_time = time.time()

    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    output_base_dir = "result/pem_positive"

    # 定义不同的 m 和 T 组合
    param_combinations = [
        (2.0, 4.0),
    ]

    # 存储所有结果
    all_results = []

    # 加载数据（只需要一次）
    df_pos, df_bg, df_test, feature_names = Tools.Positive.load_data(pos_path, bg_path, test_path)
    X_pos = df_pos.values.astype(np.float32)
    X_bg = df_bg.values.astype(np.float32)

    print(f"正样本数量: {len(X_pos)}")
    print(f"背景样本数量: {len(df_bg)}")
    if df_test is not None:
        print(f"测试样本数量: {len(df_test)}")
    print(f"特征维度: {X_pos.shape[1]}")
    print(f"参数组合数量: {len(param_combinations)}次实验")
    print("=" * 80)

    # 遍历所有参数组合（每个组合只训练1次）
    for i, (initial_margin, initial_temperature) in enumerate(param_combinations):
        reset_seed()
        print(f"\n🔬 正在训练参数组合 {i + 1}/{len(param_combinations)}: m={initial_margin}, T={initial_temperature}")

        # 为当前参数组合创建专门的输出目录
        param_output_dir = os.path.join(output_base_dir, f"m_{initial_margin}_T_{initial_temperature}")
        os.makedirs(param_output_dir, exist_ok=True)

        try:
            # 创建模型并设置初始参数
            model = AutoPEMModel(input_dim=X_pos.shape[1])

            # 手动设置初始参数（覆盖初始化值）
            model.margin = initial_margin
            model.temperature = initial_temperature

            # 训练模型
            trained_model, results = train_and_evaluate(
                model,
                pos_path=pos_path,
                bg_path=bg_path,
                test_path=test_path,
                output_dir=param_output_dir
            )

            # 收集结果
            run_result = {
                '参数组合': f'm={initial_margin}, T={initial_temperature}',
                'AD_AUC分数': results.get('ad_auc_score', 0),
                '测试AD_AUC分数': results.get('ad_auc_score_test', 0),
                '预测精度': results.get('train_accuracy', 0),
                '预测密度': results.get('train_density', 0),
                '正样本概率均值': results.get('pos_prob_mean', 0),
                '正样本概率标准差': results.get('pos_prob_std', 0),
                '正样本概率中位数': results.get('pos_median', 0),
                '背景样本概率均值': results.get('bg_prob_mean', 0),
                '背景样本概率标准差': results.get('bg_prob_std', 0),
                '背景样本概率中位数': results.get('bg_median', 0),
                '最终margin': model.margin if hasattr(model, 'margin') else initial_margin,
                '最终temperature': model.temperature if hasattr(model, 'temperature') else initial_temperature,
                '输出路径': param_output_dir
            }

            # 添加测试集结果（如果存在）
            if 'test_accuracy' in results:
                run_result['测试集准确率'] = results['test_accuracy']
                run_result['测试样本数'] = results['test_size']

            all_results.append(run_result)

            print(f"✅ 训练完成 - TEST-AD-AUC: {run_result['测试AD_AUC分数']:.4f}, "
                  f"精度: {run_result['预测精度']:.4f}, 密度: {run_result['预测密度']:.4f}")

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            # 记录失败信息
            failed_result = {
                '参数组合': f'm={initial_margin}, T={initial_temperature}',
                'AD_AUC分数': 0,
                '测试AD_AUC分数': 0,
                '预测精度': 0,
                '预测密度': 0,
                '正样本概率均值': 0,
                '正样本概率标准差': 0,
                '正样本概率中位数': 0,
                '背景样本概率均值': 0,
                '背景样本概率标准差': 0,
                '背景样本概率中位数': 0,
                '最终margin': initial_margin,
                '最终temperature': initial_temperature,
                '输出路径': param_output_dir,
                '状态': f'失败: {e}'
            }
            all_results.append(failed_result)

    # 保存所有结果到Excel
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(output_base_dir, "all_parameter_results.xlsx")
        results_df.to_excel(results_file, index=False)

        print(f"\n✅ 所有实验结果已保存至: {os.path.abspath(results_file)}")

        # 显示结果
        print(f"\n📈 参数调优实验结果:")
        print("=" * 100)

        # 只显示成功的实验
        successful_results = [r for r in all_results if r.get('测试AD_AUC分数', 0) > 0]

        if successful_results:
            print(f"{'参数组合':<20} {'TEST-AD-AUC':<8} {'预测精度':<8} {'预测密度':<8} {'最终margin':<10} {'最终temperature':<12}")
            print("-" * 100)

            for result in successful_results:
                print(f"{result['参数组合']:<20} {result['测试AD_AUC分数']:.4f}   {result['预测精度']:.4f}    "
                      f"{result['预测密度']:.4f}    {result['最终margin']:.4f}      {result['最终temperature']:.4f}")

    # 空间预测（使用最佳参数）
    if Config.EXPORT_TIFF and all_results:
        # 找出最佳结果
        successful_results = [r for r in all_results if r.get('AD_AUC分数', 0) > 0]
        if successful_results:
            best_result = successful_results[0]  # 只有一个参数组合
            best_model_path = os.path.join(best_result['输出路径'], "best_model.ckpt")

            if os.path.exists(best_model_path):
                print(f"\n🗺️  使用参数组合 {best_result['参数组合']} 进行空间预测...")

                # 加载最佳模型
                best_model = AutoPEMModel.load_from_checkpoint(
                    best_model_path,
                    input_dim=X_pos.shape[1]
                )

                # 构建特征映射
                df_pos = DataReader.load_data(pos_path)
                feature_names = df_pos.columns.tolist()
                feature_mapping = {
                    feature: os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif")
                    for feature in feature_names
                    if os.path.exists(os.path.join(Config.BASE_DIR + "entropies", f"{feature}.tif"))
                }

                # 初始化处理器并执行预测
                processor = RasterProcessor(best_model, feature_mapping)
                prob_tif_path = os.path.join(output_base_dir, "susceptibility_probability.tif")
                processor.predict_to_raster(prob_tif_path)
                print(f"✅ 空间概率分布已保存至: {os.path.abspath(prob_tif_path)}")

    end_time = time.time()
    total_time = end_time - start_time

    # 格式化显示运行时间
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    if hours > 0:
        time_str = f"{hours}小时{minutes}分钟{seconds:.1f}秒"
    elif minutes > 0:
        time_str = f"{minutes}分钟{seconds:.1f}秒"
    else:
        time_str = f"{seconds:.1f}秒"

    print(f"\n🎉 参数调优实验完成！总运行时间: {time_str}")
    print("=" * 80)

    return all_results

# ----------------------------
# 修改后的超参数网格搜索函数（同时记录AD-AUC和正样本平均概率）
# ----------------------------
def hyperparameter_grid_search(pos_path, bg_path, test_path=None, base_output_dir="grid_search_results"):
    """执行超参数网格搜索（确定性计算，每个组合只训练1次）"""

    # 定义超参数网格
    lambda_values = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    alpha_values = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    # 准备结果存储
    results = []
    ad_auc_matrix = np.zeros((len(alpha_values), len(lambda_values)))
    mean_pos_prob_matrix = np.zeros((len(alpha_values), len(lambda_values)))

    print(f"🔬 开始超参数网格搜索（每个组合进行1次实验，确定性计算）...")
    total_combinations = len(lambda_values) * len(alpha_values)
    pbar = tqdm(total=total_combinations, desc="超参数网格搜索")

    for i, alpha in enumerate(alpha_values):
        for j, lambda_val in enumerate(lambda_values):
            reset_seed()
            # 为每个组合创建独立的输出目录
            output_dir = os.path.join(base_output_dir, f"lambda_{lambda_val}_alpha_{alpha}")
            os.makedirs(output_dir, exist_ok=True)

            try:
                # 加载正样本数据（用于后续概率计算）
                df_pos = DataReader.load_data(pos_path)

                # 创建模型实例
                model = PEMModel(input_dim=df_pos.shape[1])

                # 设置超参数
                model.l2w = lambda_val
                model.adjustment_rate = alpha

                # 使用现有的train_and_evaluate函数进行训练
                trained_model, result = train_and_evaluate(
                    model,
                    pos_path=pos_path,
                    bg_path=bg_path,
                    test_path=test_path,  # 传入测试集路径
                    output_dir=output_dir
                )
                # 获取AD-AUC分数和正样本平均概率
                test_ad_auc = result.get('ad_auc_score_test', 0.5)
                mean_pos_prob = result.get('pos_prob_mean', 0)
                mean_bg_prob = result.get('bg_prob_mean', 0)
                # 记录单次运行结果
                results.append({
                    'lambda': lambda_val,
                    'alpha': alpha,
                    'run_id': 1,  # 确定性计算，只运行1次
                    'ad_auc': test_ad_auc,
                    'mean_pos_prob': mean_pos_prob,
                    'mean_bg_prob': mean_bg_prob,
                    'train_accuracy': result.get('train_accuracy', 0),
                    'train_density': result.get('train_density', 0),
                    'output_dir': output_dir
                })

                # 存储到矩阵中
                ad_auc_matrix[i, j] = test_ad_auc
                mean_pos_prob_matrix[i, j] = mean_pos_prob

                pbar.set_description(f"λ={lambda_val}, α={alpha}, TEST-AD-AUC={test_ad_auc:.4f}, MeanP={mean_pos_prob:.4f}")
                pbar.update(1)

            except Exception as e:
                print(f"错误: λ={lambda_val}, α={alpha}, 错误信息: {e}")
                # 如果运行失败，用NaN填充
                results.append({
                    'lambda': lambda_val,
                    'alpha': alpha,
                    'run_id': 1,
                    'ad_auc': np.nan,
                    'mean_pos_prob': np.nan,
                    'mean_bg_prob': np.nan,
                    'train_accuracy': np.nan,
                    'train_density': np.nan,
                    'output_dir': output_dir
                })
                ad_auc_matrix[i, j] = np.nan
                mean_pos_prob_matrix[i, j] = np.nan
                pbar.update(1)
                continue

    pbar.close()

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 创建汇总统计表（由于是确定性计算，每个组合只有一次运行）
    summary_df = results_df.groupby(['lambda', 'alpha']).agg({
        'ad_auc': ['mean', 'min', 'max'],
        'mean_pos_prob': ['mean', 'min', 'max'],
        'mean_bg_prob': ['mean', 'min', 'max'],
        'train_accuracy': 'mean',
        'train_density': 'mean'
    }).round(4)

    # 扁平化列名
    summary_df.columns = [
        'ad_auc_mean', 'ad_auc_min', 'ad_auc_max',
        'mean_pos_prob_mean', 'mean_pos_prob_min', 'mean_pos_prob_max',
        'mean_bg_prob_mean', 'mean_bg_prob_min', 'mean_bg_prob_max',
        'train_accuracy_mean', 'train_density_mean'
    ]
    summary_df = summary_df.reset_index()

    # 找出最佳参数组合
    valid_results = summary_df[summary_df['ad_auc_mean'].notna()]
    if not valid_results.empty:
        best_idx = valid_results['ad_auc_mean'].idxmax()
        best_combo = valid_results.loc[best_idx]

        print(f"\n🏆 最佳参数组合:")
        print(f"● λ = {best_combo['lambda']}, α = {best_combo['alpha']}")
        print(f"● TEST-AD-AUC: {best_combo['ad_auc_mean']:.4f}")
        print(f"● 正样本平均概率: {best_combo['mean_pos_prob_mean']:.4f}")

    return (summary_df, results_df, ad_auc_matrix, mean_pos_prob_matrix,
            lambda_values, alpha_values)


def plot_comprehensive_heatmaps(ad_auc_matrix, mean_pos_prob_matrix,
                                lambda_values, alpha_values,
                                output_path="comprehensive_hyperparameter_analysis.png"):
    """绘制包含AD-AUC和正样本平均概率的综合热力图（SCI格式，简化版）"""

    # 设置SCI论文的绘图风格
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11

    # 创建2x1的子图布局，专注于两个核心指标
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：平均AD-AUC热力图
    if not np.isnan(ad_auc_matrix).all():
        ad_auc_vmin = np.nanmin(ad_auc_matrix)
        ad_auc_vmax = np.nanmax(ad_auc_matrix)
    else:
        ad_auc_vmin, ad_auc_vmax = 0.5, 1.0  # 默认范围

    im1 = axes[0].imshow(ad_auc_matrix, cmap='viridis', aspect='auto',
                         vmin=ad_auc_vmin, vmax=ad_auc_vmax)

    # 设置刻度标签
    for ax in axes:
        ax.set_xticks(np.arange(len(lambda_values)))
        ax.set_yticks(np.arange(len(alpha_values)))
        ax.set_xticklabels([f"{l:.3f}" for l in lambda_values])
        ax.set_yticklabels([f"{a:.3f}" for a in alpha_values])

    # 在AD-AUC热力图中显示数值
    for i in range(len(alpha_values)):
        for j in range(len(lambda_values)):
            if not np.isnan(ad_auc_matrix[i, j]):
                text_color = "white" if ad_auc_matrix[i, j] < (ad_auc_vmin + ad_auc_vmax) * 0.5 else "black"
                text = axes[0].text(j, i, f'{ad_auc_matrix[i, j]:.2f}',
                                    ha="center", va="center", color=text_color, fontsize=9,
                                    fontweight='bold')

    axes[0].set_xlabel('Regularization Coefficient (λ)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Adaptation Rate (α)', fontsize=12, fontweight='bold')
    axes[0].set_title('(a) Testing AD-AUC Performance\n(Higher is better)',
                      fontsize=13, fontweight='bold', pad=15)

    # 高亮最佳AD-AUC
    if not np.isnan(ad_auc_matrix).all():
        best_idx = np.unravel_index(np.nanargmax(ad_auc_matrix), ad_auc_matrix.shape)
        axes[0].add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                                        fill=False, edgecolor='red', lw=2, linestyle='--'))
        # 添加最佳值标注
        axes[0].text(best_idx[1], best_idx[0] + 0.3, 'Best',
                     ha='center', va='bottom', color='red', fontsize=8, fontweight='bold')

    # 子图2：平均正样本概率热力图
    if not np.isnan(mean_pos_prob_matrix).all():
        prob_vmin = np.nanmin(mean_pos_prob_matrix)
        prob_vmax = np.nanmax(mean_pos_prob_matrix)
    else:
        prob_vmin, prob_vmax = 0.0, 1.0  # 默认范围

    im2 = axes[1].imshow(mean_pos_prob_matrix, cmap='RdYlGn_r', aspect='auto',
                         vmin=prob_vmin, vmax=prob_vmax)

    # 在正样本概率热力图中显示数值
    for i in range(len(alpha_values)):
        for j in range(len(lambda_values)):
            if not np.isnan(mean_pos_prob_matrix[i, j]):
                text_color = "black"
                text = axes[1].text(j, i, f'{mean_pos_prob_matrix[i, j]:.2f}',
                                    ha="center", va="center", color=text_color, fontsize=9,
                                    fontweight='bold')

    axes[1].set_xlabel('Regularization Coefficient (λ)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Adaptation Rate (α)', fontsize=12, fontweight='bold')
    axes[1].set_title('(b) Positive Sample Probability\n(Higher is better)',
                      fontsize=13, fontweight='bold', pad=15)

    # 高亮最高正样本概率
    if not np.isnan(mean_pos_prob_matrix).all():
        best_prob_idx = np.unravel_index(np.nanargmax(mean_pos_prob_matrix), mean_pos_prob_matrix.shape)
        axes[1].add_patch(plt.Rectangle((best_prob_idx[1] - 0.5, best_prob_idx[0] - 0.5), 1, 1,
                                        fill=False, edgecolor='blue', lw=2, linestyle='--'))
        # 添加最佳值标注
        axes[1].text(best_prob_idx[1], best_prob_idx[0] + 0.3, 'Best',
                     ha='center', va='bottom', color='blue', fontsize=8, fontweight='bold')

    # 添加颜色条
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='AD-AUC Score', pad=0.05)
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Probability', pad=0.05)

    # 添加整体标题
    plt.suptitle('Hyperparameter Grid Search Results', fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # 输出最佳参数组合信息
    if not np.isnan(ad_auc_matrix).all() and not np.isnan(mean_pos_prob_matrix).all():
        best_auc_idx = np.unravel_index(np.nanargmax(ad_auc_matrix), ad_auc_matrix.shape)
        best_prob_idx = np.unravel_index(np.nanargmax(mean_pos_prob_matrix), mean_pos_prob_matrix.shape)

        best_auc_lambda = lambda_values[best_auc_idx[1]]
        best_auc_alpha = alpha_values[best_auc_idx[0]]
        best_auc_value = ad_auc_matrix[best_auc_idx]

        best_prob_lambda = lambda_values[best_prob_idx[1]]
        best_prob_alpha = alpha_values[best_prob_idx[0]]
        best_prob_value = mean_pos_prob_matrix[best_prob_idx]

        print(f"\n🏆 最佳参数组合分析:")
        print(f"● 最高TEST-AD-AUC: λ={best_auc_lambda:.3f}, α={best_auc_alpha:.3f} (AD-AUC: {best_auc_value:.4f})")
        print(f"● 最高正样本概率: λ={best_prob_lambda:.3f}, α={best_prob_alpha:.3f} (概率: {best_prob_value:.4f})")

    return fig

def run_comprehensive_grid_search():
    """运行综合网格搜索，同时考虑AD-AUC和正样本平均概率"""
    start_time = time.time()

    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    base_output_dir = "result/pem_positive/grid_search_results"

    print(f"🔬 开始综合超参数网格搜索（平衡AD-AUC与正样本概率）...")

    # 执行修改后的网格搜索
    (summary_df, detailed_df, ad_auc_matrix, mean_pos_prob_matrix,
      lambda_values, alpha_values) = hyperparameter_grid_search(
        pos_path=pos_path,
        bg_path=bg_path,
        test_path=test_path,
        base_output_dir=base_output_dir,
    )

    # 绘制综合热力图 - 现在传入所有4个矩阵
    heatmap_path = os.path.join(base_output_dir, "comprehensive_hyperparameter_analysis.png")
    fig = plot_comprehensive_heatmaps(
        ad_auc_matrix=ad_auc_matrix,
        mean_pos_prob_matrix=mean_pos_prob_matrix,
        lambda_values=lambda_values,
        alpha_values=alpha_values,
        output_path=heatmap_path
    )
    # 多准则参数选择
    if not summary_df.empty:
        # 准则1：AD-AUC最高
        best_ad_auc = summary_df.loc[summary_df['ad_auc_mean'].idxmax()]

        # 准则2：正样本平均概率最高（且大于0.7）
        high_prob_df = summary_df[summary_df['mean_pos_prob_mean'] > 0.7]
        if not high_prob_df.empty:
            best_prob = high_prob_df.loc[high_prob_df['mean_pos_prob_mean'].idxmax()]
        else:
            best_prob = summary_df.loc[summary_df['mean_pos_prob_mean'].idxmax()]

        # 准则3：平衡选择（在正样本概率>0.8的条件下，选择AD-AUC最高）
        if not high_prob_df.empty:
            balanced_choice = high_prob_df.loc[high_prob_df['ad_auc_mean'].idxmax()]
        else:
            # 如果没有满足>0.7的，选择最接近0.7的
            summary_df['prob_diff'] = abs(summary_df['mean_pos_prob_mean'] - 0.7)
            balanced_choice = summary_df.loc[summary_df['prob_diff'].idxmin()]

        print("\n" + "=" * 80)
        print("⭐ 综合超参数网格搜索结果分析 ⭐")
        print("=" * 80)
        print("1. 基于单一准则的最优参数：")
        print(f"   - 最高AD-AUC: λ={best_ad_auc['lambda']:.3f}, α={best_ad_auc['alpha']:.3f}")
        print(f"     AD-AUC: {best_ad_auc['ad_auc_mean']:.4f}")
        print(f"     正样本平均概率: {best_ad_auc['mean_pos_prob_mean']:.4f}")

        print(f"   - 最高正样本概率: λ={best_prob['lambda']:.3f}, α={best_prob['alpha']:.3f}")
        print(f"     AD-AUC: {best_prob['ad_auc_mean']:.4f}")
        print(f"     正样本平均概率: {best_prob['mean_pos_prob_mean']:.4f}")

        print("\n2. 基于多准则平衡选择（推荐）：")
        print(f"   - 平衡选择: λ={balanced_choice['lambda']:.3f}, α={balanced_choice['alpha']:.3f}")
        print(f"     AD-AUC: {balanced_choice['ad_auc_mean']:.4f}")
        print(f"     正样本平均概率: {balanced_choice['mean_pos_prob_mean']:.4f}")

        # 保存结果
        results_csv_path = os.path.join(base_output_dir, "comprehensive_grid_search_results.csv")
        summary_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n📊 详细结果已保存至: {results_csv_path}")

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\n⏱️ 总运行时间: {total_time:.1f} 分钟")
    print("✅ 综合网格搜索完成！")


def lambda_grid_search(pos_path, bg_path, test_path=None, base_output_dir="lambda_grid_search_results"):
    """执行lambda超参数网格搜索（确定性计算，每个lambda只训练1次）"""
    lambda_values = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04,
                     0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    # 准备结果存储
    results = []
    ad_auc_list = []
    pos_prob_list = []
    bg_prob_list = []
    energy_gap_list = []  # 正负样本能量差
    pos_energy_list = []  # 正样本平均能量
    bg_energy_list = []  # 背景样本平均能量

    print(f"🔬 开始Lambda超参数搜索（{len(lambda_values)}个值）...")

    total_lambdas = len(lambda_values)
    pbar = tqdm(total=total_lambdas, desc="Lambda搜索进度")

    for lambda_idx, lambda_val in enumerate(lambda_values):
        reset_seed()  # 确保可重复性

        # 为每个lambda创建独立的输出目录
        output_dir = os.path.join(base_output_dir, f"lambda_{lambda_val:.4f}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 加载正样本数据
            df_pos = DataReader.load_data(pos_path)

            # 创建模型实例
            model = PEMModel(input_dim=df_pos.shape[1])

            # 设置超参数
            model.l2w = lambda_val

            # 使用现有的train_and_evaluate函数进行训练
            trained_model, result = train_and_evaluate(
                model,
                pos_path=pos_path,
                bg_path=bg_path,
                test_path=test_path,
                output_dir=output_dir
            )

            # 获取各项指标
            test_ad_auc = result.get('ad_auc_score_test', 0.5)
            mean_pos_prob = result.get('pos_prob_mean', 0)
            mean_bg_prob = result.get('bg_prob_mean', 0)

            # 获取能量值（如果模型提供）
            mean_pos_energy = result.get('pos_energy_mean', 0)
            mean_bg_energy = result.get('bg_energy_mean', 0)
            energy_gap = mean_bg_energy - mean_pos_energy  # 能量差 = 背景能量 - 正样本能量

            # 记录结果
            result_dict = {
                'lambda': lambda_val,
                'ad_auc': test_ad_auc,
                'mean_pos_prob': mean_pos_prob,
                'mean_bg_prob': mean_bg_prob,
                'mean_pos_energy': mean_pos_energy,
                'mean_bg_energy': mean_bg_energy,
                'energy_gap': energy_gap,
                'train_accuracy': result.get('train_accuracy', 0),
                'train_density': result.get('train_density', 0),
                'output_dir': output_dir
            }

            results.append(result_dict)

            # 存储到列表用于绘图
            ad_auc_list.append(test_ad_auc)
            pos_prob_list.append(mean_pos_prob)
            bg_prob_list.append(mean_bg_prob)
            pos_energy_list.append(mean_pos_energy)
            bg_energy_list.append(mean_bg_energy)
            energy_gap_list.append(energy_gap)

            pbar.set_description(f"λ={lambda_val:.4f}, AD-AUC={test_ad_auc:.4f}, "
                                 f"P(pos)={mean_pos_prob:.4f}, ΔE={energy_gap:.4f}")
            pbar.update(1)

        except Exception as e:
            print(f"\n❌ 错误: λ={lambda_val:.4f}, 错误信息: {e}")
            # 如果运行失败，用NaN填充
            result_dict = {
                'lambda': lambda_val,
                'ad_auc': np.nan,
                'mean_pos_prob': np.nan,
                'mean_bg_prob': np.nan,
                'mean_pos_energy': np.nan,
                'mean_bg_energy': np.nan,
                'energy_gap': np.nan,
                'train_accuracy': np.nan,
                'train_density': np.nan,
                'output_dir': output_dir
            }

            results.append(result_dict)
            ad_auc_list.append(np.nan)
            pos_prob_list.append(np.nan)
            bg_prob_list.append(np.nan)
            pos_energy_list.append(np.nan)
            bg_energy_list.append(np.nan)
            energy_gap_list.append(np.nan)

            pbar.update(1)
            continue

    pbar.close()

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 找出最佳lambda（基于AD-AUC）
    valid_results = results_df[results_df['ad_auc'].notna()]
    if not valid_results.empty:
        best_idx = valid_results['ad_auc'].idxmax()
        best_lambda = valid_results.loc[best_idx]

        print(f"\n🏆 最佳参数:")
        print(f"● λ = {best_lambda['lambda']:.6f}")
        print(f"● TEST-AD-AUC: {best_lambda['ad_auc']:.4f}")
        print(f"● 正样本平均概率: {best_lambda['mean_pos_prob']:.4f}")
        print(f"● 能量差(ΔE): {best_lambda['energy_gap']:.4f}")
        print(f"● 正样本平均能量: {best_lambda['mean_pos_energy']:.4f}")
        print(f"● 背景样本平均能量: {best_lambda['mean_bg_energy']:.4f}")

    return (results_df, lambda_values, ad_auc_list, pos_prob_list,
            bg_prob_list, pos_energy_list, bg_energy_list, energy_gap_list)


def run_lambda_grid_search():
    """运行Lambda参数网格搜索"""
    start_time = time.time()

    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE + ".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"
    test_path = Config.BASE_DIR + "test.xlsx"
    base_output_dir = "result/pem_positive/lambda_grid_search"
    os.makedirs(base_output_dir, exist_ok=True)

    print("=" * 80)
    print("🔬 Lambda超参数网格搜索分析")
    print("=" * 80)

    # 执行Lambda网格搜索
    (results_df, lambda_values, ad_auc_list, pos_prob_list,
     bg_prob_list, pos_energy_list, bg_energy_list, energy_gap_list) = lambda_grid_search(
        pos_path=pos_path,
        bg_path=bg_path,
        test_path=test_path,
        base_output_dir=base_output_dir
    )

    # 创建详细结果表格
    summary_df = results_df[['lambda', 'ad_auc', 'mean_pos_prob', 'mean_bg_prob',
                             'mean_pos_energy', 'mean_bg_energy', 'energy_gap',
                             'train_accuracy', 'train_density']].copy()

    # 按AD-AUC排序
    summary_df = summary_df.sort_values('ad_auc', ascending=False)

    # 输出详细结果
    print("\n" + "=" * 80)
    print("📊 Lambda超参数搜索结果汇总")
    print("=" * 80)
    print("\n按AD-AUC排序（前10名）:")
    print("-" * 120)
    print(
        f"{'Lambda':<10} {'AD-AUC':<10} {'P(pos)':<10} {'P(bg)':<10} {'E(pos)':<10} {'E(bg)':<10} {'ΔE':<10} {'A_train':<10} {'D_train':<10}")
    print("-" * 120)

    for idx, row in summary_df.head(10).iterrows():
        print(f"{row['lambda']:<10.6f} {row['ad_auc']:<10.4f} {row['mean_pos_prob']:<10.4f} "
              f"{row['mean_bg_prob']:<10.4f} {row['mean_pos_energy']:<10.4f} "
              f"{row['mean_bg_energy']:<10.4f} {row['energy_gap']:<10.4f} "
              f"{row['train_accuracy']:<10.4f} {row['train_density']:<10.4f}")

    # 输出完整结果到CSV
    csv_path = os.path.join(base_output_dir, "lambda_grid_search_results.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 输出最佳参数建议
    best_row = summary_df.iloc[0]
    print("\n" + "=" * 80)
    print("💡 最佳参数推荐（基于AD-AUC）")
    print("=" * 80)
    print(f"推荐 λ = {best_row['lambda']:.6f}")
    print(f"理由:")
    print(f"1. AD-AUC最高: {best_row['ad_auc']:.4f}")
    print(f"2. 正样本平均概率: {best_row['mean_pos_prob']:.4f}")
    print(f"3. 能量差(ΔE): {best_row['energy_gap']:.4f} (越高表示模型判别力越强)")

    # 分析λ的影响趋势
    print("\n📈 Lambda参数影响趋势分析:")
    print("-" * 60)

    # 将数据按λ排序
    trend_df = results_df.sort_values('lambda').dropna()
    if len(trend_df) > 1:
        # 计算相关系数
        corr_auc = trend_df['lambda'].corr(trend_df['ad_auc'])
        corr_prob = trend_df['lambda'].corr(trend_df['mean_pos_prob'])
        corr_gap = trend_df['lambda'].corr(trend_df['energy_gap'])

        print(f"Lambda与AD-AUC的相关系数: {corr_auc:.4f}")
        print(f"Lambda与正样本概率的相关系数: {corr_prob:.4f}")
        print(f"Lambda与能量差的相关系数: {corr_gap:.4f}")

        if corr_auc > 0.3:
            print("→ Lambda增大倾向于提高AD-AUC")
        elif corr_auc < -0.3:
            print("→ Lambda增大倾向于降低AD-AUC")
        else:
            print("→ Lambda与AD-AUC无明显线性关系")

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"\n⏱️ 总运行时间: {total_time:.1f} 分钟")
    print(f"📁 结果已保存至: {base_output_dir}")
    print("✅ Lambda网格搜索完成！")

    return results_df, summary_df

def run_architecture_ablation_study():
    """
    执行网络架构消融研究（确定性计算，每个架构只训练1次）
    """
    start_time = time.time()

    # 文件路径配置
    pos_path = Config.BASE_DIR + Config.POSITIVE+".xlsx"
    bg_path = Config.BASE_DIR + "mesh3000.xlsx"  # 固定使用mesh3000作为背景样本
    test_path = Config.BASE_DIR + "test.xlsx"

    # 定义要比较的模型架构列表
    model_classes = {
        'PEM_Shallow': PEM_Shallow,
        'PEM-Base': PEMModel,  # 基准模型
        'PEM_Deep': PEM_Deep,
        'PEM_Wide': PEM_Wide,
        'PEM_Narrow': PEM_Narrow
    }

    # 实验参数（确定性计算，每个架构只训练1次）
    max_epochs = 1000

    # 加载数据获取输入维度（所有模型共享）
    df_pos = DataReader.load_data(pos_path)
    input_dim = df_pos.shape[1]

    print("🔬 开始网络架构消融研究（确定性计算，每个架构训练1次）...")
    print("=" * 80)
    print(f"● 参与比较的架构数量: {len(model_classes)}")
    print(f"● 输入特征维度: {input_dim}")
    print(f"● 背景样本文件: {bg_path}")
    print("=" * 80)

    # 存储所有实验结果
    all_results = []

    # 对每个模型架构进行一次训练（确定性计算保证结果可重现）
    for model_name, model_class in model_classes.items():
        reset_seed()
        print(f"\n🏗️ 正在测试模型: {model_name}")

        # 为当前模型创建输出目录
        output_dir = f"result/pem_positive/architecture_ablation/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 创建模型实例
            model = model_class(input_dim=input_dim)

            # 使用简化训练函数
            best_model, metrics = train_and_evaluate(
                model=model,
                pos_path=pos_path,
                bg_path=bg_path,
                test_path=test_path,
                output_dir=output_dir,
                max_epochs=max_epochs
            )

            # 记录关键指标
            model_result = {
                'model_name': model_name,
                'ad_auc_score': metrics.get('ad_auc_score', 0),
                'ad_auc_score_test': metrics.get('ad_auc_score_test', 0),
                'train_accuracy': metrics.get('train_accuracy', 0),
                'train_density': metrics.get('train_density', 0),
                'mean_pos_prob': metrics.get('pos_prob_mean', 0),
                'bg_prob_mean': metrics.get('bg_prob_mean', 0),
                'final_margin': best_model.margin if hasattr(best_model, 'margin') else 0,
                'final_temperature': best_model.temperature if hasattr(best_model, 'temperature') else 0,
                'output_dir': output_dir,
                'training_epochs': metrics.get('training_epochs', 0),
                'training_time': metrics.get('training_time', 0)
            }

            # 添加测试集结果（如果存在）
            if 'test_accuracy' in metrics:
                model_result['test_accuracy'] = metrics['test_accuracy']
                model_result['test_size'] = metrics['test_size']

            all_results.append(model_result)

            print(f"✅ {model_name} 训练完成")
            print(f"   ● AD-AUC: {model_result['ad_auc_score']:.4f}")
            print(f"   ● 测试集AD-AUC: {model_result['ad_auc_score_test']:.4f}")
            print(f"   ● 训练精度: {model_result['train_accuracy']:.4f}")
            print(f"   ● 正样本平均概率: {model_result['mean_pos_prob']:.4f}")

        except Exception as e:
            print(f"❌ {model_name} 训练失败: {e}")
            # 记录失败信息
            failed_result = {
                'model_name': model_name,
                'ad_auc_score': 0,
                'ad_auc_score_test': 0,
                'train_accuracy': 0,
                'train_density': 0,
                'mean_pos_prob': 0,
                'bg_prob_mean': 0,
                'final_margin': 0,
                'final_temperature': 0,
                'output_dir': output_dir,
                'training_epochs': 0,
                'training_time': 0,
                'error': str(e)
            }
            all_results.append(failed_result)

    # 创建结果DataFrame并按性能排序
    results_df = pd.DataFrame(all_results)

    # 只对成功的训练结果进行排序
    successful_results = results_df[results_df['ad_auc_score'] > 0]
    if not successful_results.empty:
        results_df_sorted = successful_results.sort_values('ad_auc_score', ascending=False)

        # 重新整合失败的结果
        failed_results = results_df[results_df['ad_auc_score'] == 0]
        results_df = pd.concat([results_df_sorted, failed_results])
    else:
        results_df = results_df.sort_values('ad_auc_score', ascending=False)

    # 保存详细结果到Excel
    output_file = "result/pem_positive/architecture_ablation_results.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_excel(output_file, index=False)

    # 输出实验总结
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🎉 网络架构消融研究完成!")
    print("=" * 80)

    # 显示成功的模型排名
    successful_models = [r for r in all_results if r.get('ad_auc_score', 0) > 0]
    if successful_models:
        print("🏆 模型性能排名:")
        print("-" * 80)
        successful_models_sorted = sorted(successful_models, key=lambda x: x['ad_auc_score'], reverse=True)

        for i, result in enumerate(successful_models_sorted, 1):
            print(f"{i}. {result['model_name']}:")
            print(f"   AD-AUC: {result['ad_auc_score']:.4f}")
            print(f"   测试集AD-AUC: {result['ad_auc_score_test']:.4f}")
            print(f"   训练精度: {result['train_accuracy']:.4f}")
            print(f"   正样本平均概率: {result['mean_pos_prob']:.4f}")
            print()

    # 显示失败模型
    failed_models = [r for r in all_results if r.get('ad_auc_score', 0) == 0]
    if failed_models:
        print("❌ 训练失败的模型:")
        print("-" * 80)
        for result in failed_models:
            error_msg = result.get('error', '未知错误')
            print(f"● {result['model_name']}: {error_msg}")

    print(f"\n📈 实验概况:")
    print(f"● 总模型数量: {len(model_classes)}")
    print(f"● 成功训练: {len(successful_models)}")
    print(f"● 训练失败: {len(failed_models)}")
    print(f"● 总耗时: {total_time / 60:.2f} 分钟")
    print(f"● 结果文件: {output_file}")
    print("=" * 80)

    return results_df

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    seed_everything()
    run_one()
    #batch_training_experiment()
    #run_m_T()
    #run_Auto_m_T()
    #run_lambda_grid_search()
    #run_comprehensive_grid_search()
    #run_architecture_ablation_study()


