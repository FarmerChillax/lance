#!/usr/bin/env python3
"""
Lance vs Parquet 文件大小对比分析
分析为什么 Lance 文件可能比 Parquet 文件大，并提供优化建议
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lance
from lance.file import LanceFileWriter, LanceFileReader


def create_test_data(num_rows=10000):
    """创建测试数据集"""
    return pd.DataFrame({
        'id': range(num_rows),
        'name': [f'user_{i}' for i in range(num_rows)],
        'email': [f'user{i}@example.com' for i in range(num_rows)],
        'status': ['active'] * (num_rows // 2) + ['inactive'] * (num_rows // 2),
        'score': [i * 0.1 for i in range(num_rows)],
        'description': [f'Long description text for user {i} with repetitive patterns' * 3 for i in range(num_rows)],
        'data': [f'{"x" * 100}-{i}' for i in range(num_rows)],  # 高度可压缩的数据
    })


def write_parquet_with_compression(df, filepath, compression='zstd'):
    """写入 Parquet 文件，使用指定压缩"""
    table = pa.Table.from_pandas(df)

    # Parquet 支持列级压缩配置
    compression_config = {col: compression for col in df.columns}

    pq.write_table(
        table,
        filepath,
        compression=compression_config,
        use_dictionary=True,  # 启用字典编码
        write_statistics=True,
        row_group_size=50000
    )


def write_lance_basic(df, filepath):
    """基础 Lance 写入（无压缩）"""
    table = pa.Table.from_pandas(df)
    dataset = lance.write_dataset(table, filepath, mode="create")
    return dataset


def write_lance_with_zstd(df, filepath, compression_level=3):
    """Lance 写入，使用 zstd 压缩"""
    # 为所有字符串字段添加 zstd 压缩
    string_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(exclude=['object']).columns

    fields = []

    # 字符串字段使用 zstd 压缩
    for col in string_columns:
        fields.append(
            pa.field(
                col,
                pa.string(),
                metadata={
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": str(compression_level)
                }
            )
        )

    # 数值字段也可以尝试压缩
    for col in numeric_columns:
        dtype = pa.int64() if col == 'id' else pa.float64()
        fields.append(
            pa.field(
                col,
                dtype,
                metadata={
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": str(compression_level)
                }
            )
        )

    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema)

    dataset = lance.write_dataset(table, filepath, mode="create")
    return dataset


def write_lance_optimized(df, filepath):
    """Lance 优化写入，使用多种编码策略"""
    string_columns = df.select_dtypes(include=['object']).columns
    numeric_columns = df.select_dtypes(exclude=['object']).columns

    fields = []

    # 字符串字段使用 zstd + 字典编码
    for col in string_columns:
        # 对于重复性高的字段使用字典编码
        metadata = {
            "lance-encoding:compression": "zstd",
            "lance-encoding:compression-level": "6"
        }

        # 对于状态类字段，启用字典编码
        if col in ['status'] or df[col].nunique() / len(df) < 0.1:
            metadata.update({
                "lance-encoding:dict-divisor": "4",  # 更激进的字典编码
                "lance-encoding:dict-size-ratio": "0.9"
            })

        fields.append(pa.field(col, pa.string(), metadata=metadata))

    # 数值字段
    for col in numeric_columns:
        dtype = pa.int64() if col == 'id' else pa.float64()
        metadata = {}

        # 对于浮点数，可以尝试 BSS + 压缩
        if dtype == pa.float64():
            metadata = {
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "3",
                "lance-encoding:bss": "auto"  # 字节流分离
            }

        fields.append(pa.field(col, dtype, metadata=metadata))

    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema)

    dataset = lance.write_dataset(
        table,
        filepath,
        mode="create",
        max_rows_per_file=50000,  # 调整文件大小
        max_rows_per_group=2048   # 调整组大小
    )
    return dataset


def get_directory_size(path):
    """计算目录或文件大小"""
    if os.path.isfile(path):
        return os.path.getsize(path)

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size


def analyze_compression_effectiveness():
    """分析压缩效果对比"""
    print("=== Lance vs Parquet 压缩效果分析 ===\n")

    # 创建测试数据
    print("1. 创建测试数据...")
    df = create_test_data(10000)
    print(f"   数据集大小: {len(df):,} 行")
    print(f"   列数: {len(df.columns)}")
    print(f"   内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # 分析数据特征
    print("\n2. 数据特征分析:")
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        print(f"   {col}: 唯一值比例 {unique_ratio:.4f}")

    results = {}
    temp_dir = tempfile.mkdtemp()

    try:
        print(f"\n3. 文件格式对比 (临时目录: {temp_dir}):")

        # 测试 Parquet (不同压缩)
        parquet_configs = [
            ('none', 'none'),
            ('snappy', 'snappy'),
            ('gzip', 'gzip'),
            ('zstd', 'zstd')
        ]

        for name, compression in parquet_configs:
            filepath = os.path.join(temp_dir, f'test_{name}.parquet')
            start_time = time.time()
            write_parquet_with_compression(df, filepath, compression)
            write_time = time.time() - start_time

            size = get_directory_size(filepath)
            results[f'parquet_{name}'] = {
                'size': size,
                'write_time': write_time,
                'path': filepath
            }
            print(f"   Parquet ({compression}): {size:,} bytes, 写入时间: {write_time:.2f}s")

        # 测试 Lance (不同配置)
        lance_configs = [
            ('basic', lambda df, path: write_lance_basic(df, path)),
            ('zstd_level3', lambda df, path: write_lance_with_zstd(df, path, 3)),
            ('zstd_level6', lambda df, path: write_lance_with_zstd(df, path, 6)),
            ('optimized', lambda df, path: write_lance_optimized(df, path))
        ]

        for name, write_func in lance_configs:
            dirpath = os.path.join(temp_dir, f'lance_{name}')
            start_time = time.time()
            dataset = write_func(df, dirpath)
            write_time = time.time() - start_time

            size = get_directory_size(dirpath)
            results[f'lance_{name}'] = {
                'size': size,
                'write_time': write_time,
                'path': dirpath,
                'dataset': dataset
            }
            print(f"   Lance ({name}): {size:,} bytes, 写入时间: {write_time:.2f}s")

        # 分析结果
        print(f"\n4. 详细对比分析:")

        parquet_zstd_size = results['parquet_zstd']['size']
        lance_zstd_size = results['lance_zstd_level3']['size']

        print(f"   Parquet (zstd): {parquet_zstd_size:,} bytes")
        print(f"   Lance (zstd level 3): {lance_zstd_size:,} bytes")
        print(f"   Lance 比 Parquet 大: {lance_zstd_size / parquet_zstd_size:.2f} 倍")

        # 压缩比对比
        raw_size = results['parquet_none']['size']
        print(f"\n   压缩比对比 (vs 无压缩 Parquet {raw_size:,} bytes):")
        for key, data in results.items():
            compression_ratio = raw_size / data['size']
            print(f"   {key}: {compression_ratio:.2f}x 压缩")

        # 读取性能测试
        print(f"\n5. 读取性能测试:")
        test_random_access_performance(results, df)

    finally:
        # 清理
        shutil.rmtree(temp_dir)

    return results


def test_random_access_performance(results, original_df):
    """测试随机访问性能"""
    import random

    # 生成随机索引
    indices = sorted([random.randint(0, len(original_df) - 1) for _ in range(100)])

    print("   随机访问 100 行的性能:")

    # 测试 Parquet
    parquet_path = results['parquet_zstd']['path']
    start_time = time.time()
    parquet_table = pq.read_table(parquet_path)
    parquet_result = parquet_table.take(indices)
    parquet_time = time.time() - start_time
    print(f"   Parquet: {parquet_time:.4f}s")

    # 测试 Lance
    lance_path = results['lance_zstd_level3']['path']
    start_time = time.time()
    lance_dataset = lance.dataset(lance_path)
    lance_result = lance_dataset.to_table().take(indices)
    lance_time = time.time() - start_time
    print(f"   Lance: {lance_time:.4f}s")

    print(f"   Lance 比 Parquet 快: {parquet_time / lance_time:.2f}x")


def explain_size_difference():
    """解释文件大小差异的原因"""
    print("""
=== 为什么 Lance 文件可能比 Parquet 大？===

1. **元数据开销**:
   - Lance 存储额外的元数据用于随机访问优化
   - 包括搜索缓存、重复索引等
   - Parquet 主要优化顺序扫描

2. **编码策略差异**:
   - Parquet 有成熟的列式压缩和字典编码
   - Lance 的压缩策略可能需要调优
   - 不同的块大小和页面布局

3. **格式设计权衡**:
   - Lance 优化随机访问性能
   - Parquet 优化存储效率和顺序扫描
   - Lance 可能牺牲一些压缩比换取访问性能

4. **压缩算法应用**:
   - Parquet 在列级别应用压缩
   - Lance 在页面/块级别应用压缩
   - 压缩粒度的差异影响效果

优化建议:
- 调整 Lance 的 max_rows_per_group 和 max_rows_per_file
- 合理使用字典编码 (lance-encoding:dict-*)
- 根据数据特征选择合适的压缩级别
- 考虑使用 RLE 编码处理重复数据
- 启用 BSS (字节流分离) 处理浮点数
""")


def provide_optimization_recommendations():
    """提供 Lance 压缩优化建议"""
    print("""
=== Lance 压缩优化建议 ===

1. **字符串字段优化**:
   ```python
   pa.field("text_col", pa.string(), metadata={
       "lance-encoding:compression": "zstd",
       "lance-encoding:compression-level": "6",
       "lance-encoding:dict-divisor": "4",        # 更激进的字典编码
       "lance-encoding:dict-size-ratio": "0.9"    # 允许更大的字典
   })
   ```

2. **浮点数优化**:
   ```python
   pa.field("float_col", pa.float64(), metadata={
       "lance-encoding:compression": "zstd",
       "lance-encoding:bss": "on",               # 字节流分离
       "lance-encoding:compression-level": "3"
   })
   ```

3. **重复数据优化**:
   ```python
   pa.field("status_col", pa.string(), metadata={
       "lance-encoding:compression": "zstd",
       "lance-encoding:rle-threshold": "0.3",    # 降低 RLE 阈值
   })
   ```

4. **写入参数优化**:
   ```python
   lance.write_dataset(
       table,
       path,
       max_rows_per_group=1024,      # 较小的组提高压缩比
       max_rows_per_file=50000,      # 适中的文件大小
   )
   ```

5. **数据预处理**:
   - 对字符串数据进行排序可以提高压缩比
   - 将低基数字段转换为分类类型
   - 考虑使用更紧凑的数据类型
""")


if __name__ == "__main__":
    # 运行分析
    analyze_compression_effectiveness()
    explain_size_difference()
    provide_optimization_recommendations()