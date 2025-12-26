import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm

from EnergeModel.Tools import Config


def sample_susceptibility_probability(tif_path, output_excel_path):
    """
    对易发性概率TIFF文件进行间隔采样，统计大于0.5的像元比例

    Parameters:
    tif_path: str, TIFF文件路径
    output_excel_path: str, 输出Excel文件路径
    """

    # 读取TIFF文件
    with rasterio.open(tif_path) as src:
        # 读取第一个波段（概率数据）
        probability_data = src.read(1)
        nodata = src.nodata

        # 获取图像尺寸
        height, width = probability_data.shape
        print(f"图像尺寸: {width} x {height}")
        print(f"NoData值: {nodata}")

    # 创建掩膜，排除NoData区域
    if nodata is not None:
        valid_mask = (probability_data != nodata)
        total_valid_pixels = np.sum(valid_mask)
        print(f"有效像元数量: {total_valid_pixels:,}")
    else:
        valid_mask = np.ones_like(probability_data, dtype=bool)
        total_valid_pixels = height * width
        print(f"总像元数量: {total_valid_pixels:,}")

    # 生成采样间隔序列：1, 3, 5, ..., 99
    sampling_intervals = list(range(1, 1000, 10))
    print(f"采样间隔: {sampling_intervals}")

    # 存储结果
    results = []

    # 对每个采样间隔进行采样统计
    for interval in tqdm(sampling_intervals, desc="采样进度"):
        # 创建采样掩膜
        sampling_mask = np.zeros_like(probability_data, dtype=bool)

        # 设置采样点（行列间隔都为interval）
        sampling_mask[::interval, ::interval] = True

        # 结合有效像元掩膜
        final_sampling_mask = sampling_mask & valid_mask

        # 统计采样点数量
        sampled_pixels_count = np.sum(final_sampling_mask)

        thresh=Config.SAMPLE_TRESH
        if sampled_pixels_count > 0:
            # 获取采样点的概率值
            sampled_probabilities = probability_data[final_sampling_mask]

            # 统计大于0.5的像元数量
            high_susceptibility_count = np.sum(sampled_probabilities > thresh)

            # 计算比例
            high_susceptibility_ratio = high_susceptibility_count / sampled_pixels_count

            results.append({
                '采样间隔': interval,
                '采样像元数': sampled_pixels_count,
                '高易发性像元数(>'+format(thresh, ".1f")+')': high_susceptibility_count,
                '高易发性比例': high_susceptibility_ratio,
                '采样密度(%)': (sampled_pixels_count / total_valid_pixels) * 100
            })
        else:
            print(f"警告: 间隔 {interval} 没有采样到有效像元")
            results.append({
                '采样间隔': interval,
                '采样像元数': 0,
                '高易发性像元数(>'+format(thresh, ".1f")+')': 0,
                '高易发性比例': 0.0,
                '采样密度(%)': 0.0
            })

    # 创建DataFrame
    df_results = pd.DataFrame(results)

    # 计算总体统计（不采样的情况，即间隔=1）
    overall_high_susceptibility_count = np.sum(probability_data[valid_mask] > thresh)
    overall_ratio = overall_high_susceptibility_count / total_valid_pixels

    # 添加总体统计信息
    summary_info = {
        '文件路径': tif_path,
        '图像尺寸': f"{width} x {height}",
        '有效像元总数': total_valid_pixels,
        '高易发性像元总数(>'+format(thresh, ".1f")+')': overall_high_susceptibility_count,
        '总体高易发性比例': overall_ratio,
        '采样间隔范围': f"{min(sampling_intervals)}-{max(sampling_intervals)}",
        '采样间隔数量': len(sampling_intervals)
    }

    # 保存到Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 主数据表
        df_results.to_excel(writer, sheet_name='采样统计结果', index=False)

        # 创建汇总信息表
        df_summary = pd.DataFrame(list(summary_info.items()), columns=['项目', '值'])
        df_summary.to_excel(writer, sheet_name='汇总信息', index=False)

        # 创建统计摘要表
        stats_summary = {
            '平均高易发性比例': df_results['高易发性比例'].mean(),
            '最大高易发性比例': df_results['高易发性比例'].max(),
            '最小高易发性比例': df_results['高易发性比例'].min(),
            '标准差': df_results['高易发性比例'].std(),
            '与总体比例的最大偏差': abs(df_results['高易发性比例'] - overall_ratio).max()
        }
        df_stats = pd.DataFrame(list(stats_summary.items()), columns=['统计量', '值'])
        df_stats.to_excel(writer, sheet_name='统计摘要', index=False)

    print(f"\n✅ 采样统计完成！")
    print(f"📊 总体高易发性比例: {overall_ratio:.4f} ({overall_high_susceptibility_count}/{total_valid_pixels})")
    print(f"📈 采样比例范围: {df_results['高易发性比例'].min():.4f} - {df_results['高易发性比例'].max():.4f}")
    print(f"💾 结果已保存至: {output_excel_path}")

    return df_results, summary_info

def create_sampling_visualization(df_results, output_image_path):
    """
    创建采样结果可视化图表
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1: 高易发性比例随采样间隔的变化
    ax1.plot(df_results['采样间隔'], df_results['高易发性比例'], 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('采样间隔')
    ax1.set_ylabel('高易发性比例')
    ax1.set_title('高易发性比例随采样间隔的变化')
    ax1.grid(True, alpha=0.3)

    # 子图2: 采样密度和高易发性比例的双Y轴图
    ax2_twin = ax2.twinx()

    # 采样密度（左Y轴）
    ax2.plot(df_results['采样间隔'], df_results['采样密度(%)'], 'g-s', linewidth=2, markersize=4, label='采样密度')
    ax2.set_xlabel('采样间隔')
    ax2.set_ylabel('采样密度 (%)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # 高易发性比例（右Y轴）
    ax2_twin.plot(df_results['采样间隔'], df_results['高易发性比例'], 'r-o', linewidth=2, markersize=4, label='高易发性比例')
    ax2_twin.set_ylabel('高易发性比例', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')

    ax2.set_title('采样密度与高易发性比例关系')
    ax2.grid(True, alpha=0.3)

    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"📊 可视化图表已保存至: {output_image_path}")
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    base_dir = "../result/pem_positive/"
    tif_file = base_dir+"susceptibility_probability.tif"  # 替换为您的TIFF文件路径
    output_excel = base_dir+"sampling_statistics.xlsx"
    output_chart = base_dir+"sampling_analysis.png"

    try:
        # 执行采样统计
        df_results, summary_info = sample_susceptibility_probability(tif_file, output_excel)

        # 创建可视化图表
        create_sampling_visualization(df_results, output_chart)

        # 打印关键结果
        print("\n" + "=" * 50)
        print("关键发现:")
        print("=" * 50)
        for interval in [1, 5, 10, 20, 50]:
            if interval in df_results['采样间隔'].values:
                row = df_results[df_results['采样间隔'] == interval].iloc[0]
                print(f"间隔 {interval}: 比例={row['高易发性比例']:.4f}, 采样密度={row['采样密度(%)']:.2f}%")

    except FileNotFoundError:
        print(f"❌ 文件未找到: {tif_file}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")