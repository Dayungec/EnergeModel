import os

import rasterio
import numpy as np
from typing import Dict

import matplotlib.colors as mcolors
from scipy import ndimage  # 用于中值滤波


class RasterProcessor:
    def __init__(self, model, feature_mapping: Dict[str, str]):
        """
        :param model: 训练好的逻辑回归模型
        :param feature_mapping: 字段名到TIFF路径的映射 (如 {'dcmd1': 'path/to/dcmd1.tif'})
        """
        self.model = model
        self.feature_mapping = feature_mapping
        self.required_features = list(feature_mapping.keys())
        self.batch_size = 100

    def predict_to_raster(self, output_path: str):
        """逐行批处理预测，避免内存不足"""
        # 第一步：读取第一个TIFF获取空间参考
        sample_tif = next(iter(self.feature_mapping.values()))
        with rasterio.open(sample_tif) as src:
            meta = src.meta.copy()
            height, width = src.shape
            transform = src.transform
            nodata = src.nodata

        # 第二步：初始化输出栅格
        output = np.full((height, width), nodata, dtype=np.float32)

        print(f"开始逐行预测，图像尺寸: {width} x {height}")
        print(f"批处理大小: {self.batch_size} 行")

        # 第三步：逐行读取和预测
        for start_row in range(0, height, self.batch_size):
            end_row = min(start_row + self.batch_size, height)
            batch_rows = end_row - start_row

            print(f"处理行: {start_row}-{end_row - 1} ({batch_rows}行)")

            # 初始化当前批次的输入数据
            batch_data = np.full((len(self.required_features), batch_rows, width), np.nan)
            batch_valid_mask = np.ones((batch_rows, width), dtype=bool)

            # 读取当前批次的所有特征数据
            for i, feature in enumerate(self.required_features):
                tif_path = self.feature_mapping[feature]
                with rasterio.open(tif_path) as src:
                    # 读取当前批次的图像数据
                    window = rasterio.windows.Window(0, start_row, width, batch_rows)
                    band_data = src.read(1, window=window)

                    batch_data[i] = band_data
                    # 更新有效掩膜
                    batch_valid_mask &= (band_data != src.nodata)

            # 处理当前批次的有效像元
            valid_positions = np.where(batch_valid_mask)
            num_valid = len(valid_positions[0])

            if num_valid > 0:
                # 提取有效像元数据
                valid_pixels = np.zeros((num_valid, len(self.required_features)), dtype=np.float32)

                for feat_idx in range(len(self.required_features)):
                    valid_pixels[:, feat_idx] = batch_data[feat_idx][valid_positions]

                # 标准化数据（如果模型训练时做了标准化）
                if hasattr(self.model, 'scaler'):
                    valid_pixels = self.model.scaler.transform(valid_pixels)

                # 批量预测概率
                try:
                    batch_probs = self.model.predict_proba(valid_pixels)

                    # 将预测结果填回输出数组
                    global_row_indices = valid_positions[0] + start_row
                    global_col_indices = valid_positions[1]
                    output[global_row_indices, global_col_indices] = batch_probs

                    print(f"  - 有效像元: {num_valid:,}个，预测完成")

                except MemoryError:
                    # 如果批量预测仍然内存不足，进一步分批处理
                    print(f"  - 内存不足，进一步分批处理 {num_valid} 个像元")
                    self._predict_in_minibatches(valid_pixels, global_row_indices,
                                                 global_col_indices, output, minibatch_size=10000)
            else:
                print(f"  - 无有效像元")

        # 第四步：写入输出
        meta.update({
            'dtype': 'float32',
            'nodata': nodata,
            'count': 1
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output, 1)

        print(f"✅ 预测完成，结果保存至: {output_path}")
        return output_path

    def predict_to_raster_with_filter(self, median_filter_size,output_path: str):
        """逐行批处理预测，避免内存不足"""
        # 第一步：读取第一个TIFF获取空间参考
        sample_tif = next(iter(self.feature_mapping.values()))
        with rasterio.open(sample_tif) as src:
            meta = src.meta.copy()
            height, width = src.shape
            transform = src.transform
            nodata = src.nodata

        # 第二步：初始化输出栅格
        output = np.full((height, width), nodata, dtype=np.float32)

        print(f"开始逐行预测，图像尺寸: {width} x {height}")
        print(f"批处理大小: {self.batch_size} 行")

        # 第三步：逐行读取和预测
        for start_row in range(0, height, self.batch_size):
            end_row = min(start_row + self.batch_size, height)
            batch_rows = end_row - start_row

            print(f"处理行: {start_row}-{end_row - 1} ({batch_rows}行)")

            # 初始化当前批次的输入数据
            batch_data = np.full((len(self.required_features), batch_rows, width), np.nan)
            batch_valid_mask = np.ones((batch_rows, width), dtype=bool)

            # 读取当前批次的所有特征数据
            for i, feature in enumerate(self.required_features):
                tif_path = self.feature_mapping[feature]
                with rasterio.open(tif_path) as src:
                    # 读取当前批次的图像数据
                    window = rasterio.windows.Window(0, start_row, width, batch_rows)
                    band_data = src.read(1, window=window)

                    batch_data[i] = band_data
                    # 更新有效掩膜
                    batch_valid_mask &= (band_data != src.nodata)

            # 处理当前批次的有效像元
            valid_positions = np.where(batch_valid_mask)
            num_valid = len(valid_positions[0])

            if num_valid > 0:
                # 提取有效像元数据
                valid_pixels = np.zeros((num_valid, len(self.required_features)), dtype=np.float32)

                for feat_idx in range(len(self.required_features)):
                    valid_pixels[:, feat_idx] = batch_data[feat_idx][valid_positions]

                # 标准化数据（如果模型训练时做了标准化）
                if hasattr(self.model, 'scaler'):
                    valid_pixels = self.model.scaler.transform(valid_pixels)

                # 批量预测概率
                try:
                    batch_probs = self.model.predict_proba(valid_pixels)

                    # 将预测结果填回输出数组
                    global_row_indices = valid_positions[0] + start_row
                    global_col_indices = valid_positions[1]
                    output[global_row_indices, global_col_indices] = batch_probs

                    print(f"  - 有效像元: {num_valid:,}个，预测完成")

                except MemoryError:
                    # 如果批量预测仍然内存不足，进一步分批处理
                    print(f"  - 内存不足，进一步分批处理 {num_valid} 个像元")
                    self._predict_in_minibatches(valid_pixels, global_row_indices,
                                                 global_col_indices, output, minibatch_size=10000)
            else:
                print(f"  - 无有效像元")

        valid_mask = (output != nodata)

        # 对有效数据区域进行中值滤波
        # 注意：仅对有效数据区域进行处理，避免NoData区域参与计算
        filtered_output = output.copy()

        # 应用中值滤波
        # size参数指定滤波窗口大小，例如3表示3x3的窗口
        print("✅ 预测完成，开始进行中值滤波以平滑空间噪声...")
        filtered_data = ndimage.median_filter(output, size=median_filter_size)

        # 确保只有原本有效的区域被更新，保持NoData区域不变
        filtered_output[valid_mask] = filtered_data[valid_mask]

        print(f"✅ 中值滤波完成 (窗口大小: {median_filter_size}x{median_filter_size})")

        # 使用滤波后的数据更新output
        output = filtered_output

        # 第四步：写入输出
        meta.update({
            'dtype': 'float32',
            'nodata': nodata,
            'count': 1
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(output, 1)

        print(f"✅ 预测完成，结果保存至: {output_path}")
        return output_path

    def generate_susceptibility_zones(self,prob_tif_path, risk_thresholds, output_tif_path, colors):
        """
        根据概率TIFF和风险阈值生成易发性区划TIFF

        参数:
            prob_tif_path: 输入概率TIFF路径
            risk_thresholds: 风险阈值字典，格式如：
                {
                    '高风险区': {'下限阈值': 0.8, '上限阈值': 1.0},
                    '中高风险区': {'下限阈值': 0.6, '上限阈值': 0.8},
                    ...
                }
            output_tif_path: 输出区划TIFF路径
            colors: 各风险区颜色列表，如 ['red', 'orange', ...]
        """
        with rasterio.open(prob_tif_path) as src:
            prob = src.read(1)
            meta = src.meta.copy()
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

        # 初始化区划矩阵 (0=最高风险，4=最低风险)
        zones = np.full_like(prob, fill_value=len(risk_thresholds) - 1, dtype=np.uint8)

        # 按风险阈值划分区域
        sorted_zones = sorted(risk_thresholds.items(),
                              key=lambda x: x[1]['下限阈值'], reverse=True)

        for i, (zone_name, thresholds) in enumerate(sorted_zones):
            lower = thresholds['下限阈值']
            upper = thresholds['上限阈值']
            mask = (prob >= lower) & (prob < upper) if i != 0 else (prob >= lower)
            zones[mask] = i

        # 处理NoData区域
        if nodata is not None:
            zones[prob == nodata] = 255  # 用255表示NoData

        # 更新元数据
        meta.update({
            'dtype': 'uint8',
            'nodata': 255,
            'count': 1
        })

        # 写入TIFF文件
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(zones, 1)

            # 添加颜色表
            colormap = {
                i: tuple(int(c * 255) for c in mcolors.to_rgb(colors[i]))
                for i in range(len(risk_thresholds))
            }
            colormap[255] = (0, 0, 0, 0)  # NoData区域透明
            dst.write_colormap(1, colormap)

        print(f"✅ 易发性区划已保存至: {os.path.abspath(output_tif_path)}")
        return zones