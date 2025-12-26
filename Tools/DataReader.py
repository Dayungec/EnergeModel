import pandas as pd
import numpy as np
import os
from typing import List, Optional

def get_final_columns(file_path: str, columns_to_filter: Optional[List[str]] = None) -> List[str]:
    """
    静态方法：获取过滤指定列后的最终列名列表，而不加载全部数据。

    Args:
        file_path (str): 数据文件路径。
        columns_to_filter (Optional[List[str]], optional): 需要过滤掉的列名列表。

    Returns:
        List[str]: 过滤后的列名列表。
    """
    if columns_to_filter is None:
        columns_to_filter = ['lon', 'lat']

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    # 只读取列名，不加载全部数据
    df_temp = pd.read_excel(file_path, nrows=0)
    final_columns = [col for col in df_temp.columns if col not in columns_to_filter]

    return final_columns


def load_data(file_path: str, columns_to_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """
    静态方法：加载数据文件，并过滤掉指定的列。

    Args:
        file_path (str): 要读取的数据文件路径。
        columns_to_filter (Optional[List[str]], optional): 需要过滤掉的列名列表。
            默认为 None，即使用 ['lon', 'lat']。

    Returns:
        pd.DataFrame: 读取并处理后的DataFrame。

    Raises:
        FileNotFoundError: 当指定的文件路径不存在时。
        Exception: 读取文件过程中可能出现的其他错误。
    """
    # 设置默认的过滤列
    if columns_to_filter is None:
        columns_to_filter = ['lon', 'lat','label','year','month','day']

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    try:
        # 1. 使用pandas读取文件
        df = pd.read_excel(file_path)  # 如果是CSV，使用 pd.read_csv

        # 2. 检查并过滤指定的列
        columns_exist_to_drop = [col for col in columns_to_filter if col in df.columns]

        if columns_exist_to_drop:
            df = df.drop(columns=columns_exist_to_drop)
            print(f"提示: 已从文件 '{file_path}' 中过滤掉以下列: {columns_exist_to_drop}")

        # 3. 统一数据类型，例如转换为 float32
        df = df.astype(np.float32)

        return df

    except Exception as e:
        raise Exception(f"读取文件 {file_path} 时发生错误: {e}")


