#!/usr/bin/env python3
"""
Lance 压缩优化实战示例
针对 Lance vs Parquet 文件大小差异提供具体的优化方案
"""

import os
import tempfile
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lance


def create_optimized_lance_dataset(df, output_path):
    """
    创建经过优化的 Lance 数据集，尽可能减小文件大小
    """
    # 分析每一列的特征
    column_configs = {}

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)

        if df[col].dtype == 'object':  # 字符串列
            if unique_ratio < 0.1:  # 低基数字段，适合字典编码
                column_configs[col] = {
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": "9",
                    "lance-encoding:dict-divisor": "2",
                    "lance-encoding:dict-size-ratio": "0.95"
                }
            elif unique_ratio < 0.5:  # 中等基数
                column_configs[col] = {
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": "6",
                    "lance-encoding:dict-divisor": "4"
                }
            else:  # 高基数字段
                column_configs[col] = {
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": "6"
                }

        elif df[col].dtype in ['int64', 'int32']:  # 整数列
            # 检查是否有重复值，适合 RLE
            if unique_ratio < 0.8:
                column_configs[col] = {
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": "3",
                    "lance-encoding:rle-threshold": "0.4"
                }
            else:
                column_configs[col] = {
                    "lance-encoding:compression": "zstd",
                    "lance-encoding:compression-level": "3"
                }

        elif df[col].dtype in ['float64', 'float32']:  # 浮点数列
            column_configs[col] = {
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "3",
                "lance-encoding:bss": "auto"  # 字节流分离
            }

    # 创建优化的 schema
    fields = []
    for col in df.columns:
        if df[col].dtype == 'object':
            dtype = pa.string()
        elif df[col].dtype == 'int64':
            dtype = pa.int64()
        elif df[col].dtype == 'float64':
            dtype = pa.float64()
        else:
            dtype = pa.string()  # 默认

        metadata = column_configs.get(col, {})
        fields.append(pa.field(col, dtype, metadata=metadata))

    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema)

    # 使用优化的写入参数
    dataset = lance.write_dataset(
        table,
        output_path,
        mode="create",
        max_rows_per_group=1024,    # 较小的组大小提高压缩效果
        max_rows_per_file=25000,    # 适中的文件大小
    )

    return dataset, column_configs


def compare_compression_strategies():
    """对比不同的压缩策略"""

    print("=== Lance 压缩策略优化对比 ===\n")

    # 创建具有不同特征的测试数据
    num_rows = 10000
    df = pd.DataFrame({
        # 低基数字段 (适合字典编码)
        'category': (['A', 'B', 'C'] * (num_rows // 3 + 1))[:num_rows],  # 3 个唯一值
        'status': (['active'] * 7000 + ['inactive'] * 3000)[:num_rows],  # 2 个唯一值

        # 中等基数字段
        'region': [f'region_{i%50}' for i in range(num_rows)],  # 50 个唯一值

        # 高基数字段
        'user_id': [f'user_{i}' for i in range(num_rows)],  # 10000 个唯一值
        'email': [f'user{i}@domain{i%100}.com' for i in range(num_rows)],

        # 数值字段
        'count': [i % 100 for i in range(num_rows)],  # 重复的整数
        'score': [i * 0.1 for i in range(num_rows)],  # 浮点数

        # 长文本字段
        'description': [f'This is a long description for item {i} with many repeated words and patterns' * 3
                       for i in range(num_rows)]
    })

    temp_dir = tempfile.mkdtemp()

    try:
        print(f"测试数据: {len(df)} 行, {len(df.columns)} 列")
        print("数据特征分析:")
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            print(f"  {col}: 唯一值比例 {unique_ratio:.4f}")

        print(f"\n压缩结果对比:")

        # 1. Parquet baseline
        parquet_path = os.path.join(temp_dir, 'baseline.parquet')
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression='zstd')
        parquet_size = os.path.getsize(parquet_path)
        print(f"Parquet (zstd): {parquet_size:,} bytes")

        # 2. Lance 基础压缩
        lance_basic_path = os.path.join(temp_dir, 'lance_basic')
        schema_basic = pa.schema([
            pa.field(col,
                    pa.string() if df[col].dtype == 'object'
                    else (pa.int64() if df[col].name in ['count'] or 'int' in str(df[col].dtype)
                         else pa.float64()),
                    metadata={"lance-encoding:compression": "zstd"})
            for col in df.columns
        ])

        # 确保数据类型正确
        df_converted = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                df_converted[col] = df[col].astype(str)

        table_basic = pa.Table.from_pandas(df_converted, schema=schema_basic)
        lance.write_dataset(table_basic, lance_basic_path)
        lance_basic_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(lance_basic_path)
                              for filename in filenames)
        print(f"Lance (基础 zstd): {lance_basic_size:,} bytes ({lance_basic_size/parquet_size:.2f}x)")

        # 3. Lance 优化压缩
        lance_opt_path = os.path.join(temp_dir, 'lance_optimized')
        dataset, configs = create_optimized_lance_dataset(df, lance_opt_path)
        lance_opt_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(lance_opt_path)
                            for filename in filenames)
        print(f"Lance (优化): {lance_opt_size:,} bytes ({lance_opt_size/parquet_size:.2f}x)")

        # 显示优化配置
        print(f"\n优化配置详情:")
        for col, config in configs.items():
            print(f"  {col}: {config}")

        # 4. 极致优化 - 调整写入参数
        lance_extreme_path = os.path.join(temp_dir, 'lance_extreme')

        # 对数据进行预处理：排序可以提高压缩效果
        df_sorted = df.sort_values(['category', 'status', 'region'])

        # 创建极致优化的 schema
        extreme_fields = []
        for col in df.columns:
            metadata = configs.get(col, {}).copy()

            # 进一步优化低基数字段
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:
                metadata.update({
                    "lance-encoding:dict-divisor": "1",
                    "lance-encoding:dict-size-ratio": "0.98"
                })

            if df[col].dtype == 'object':
                dtype = pa.string()
            elif df[col].dtype == 'int64':
                dtype = pa.int64()
            elif df[col].dtype == 'float64':
                dtype = pa.float64()
            else:
                dtype = pa.string()

            extreme_fields.append(pa.field(col, dtype, metadata=metadata))

        extreme_schema = pa.schema(extreme_fields)
        extreme_table = pa.Table.from_pandas(df_sorted, schema=extreme_schema)

        lance.write_dataset(
            extreme_table,
            lance_extreme_path,
            mode="create",
            max_rows_per_group=512,     # 更小的组
            max_rows_per_file=10000,    # 更小的文件
        )

        lance_extreme_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(lance_extreme_path)
                                for filename in filenames)
        print(f"Lance (极致优化): {lance_extreme_size:,} bytes ({lance_extreme_size/parquet_size:.2f}x)")

        print(f"\n总结:")
        print(f"- Parquet 作为基准: {parquet_size:,} bytes")
        print(f"- Lance 基础优化相对 Parquet: {(lance_basic_size/parquet_size - 1)*100:+.1f}%")
        print(f"- Lance 深度优化相对 Parquet: {(lance_opt_size/parquet_size - 1)*100:+.1f}%")
        print(f"- Lance 极致优化相对 Parquet: {(lance_extreme_size/parquet_size - 1)*100:+.1f}%")

    finally:
        shutil.rmtree(temp_dir)


def provide_practical_recommendations():
    """提供实用的优化建议"""

    print("""
=== Lance 压缩优化实战建议 ===

基于测试结果，Lance 文件比 Parquet 大 2-3 倍是相对合理的，原因如下：

1. **设计目标不同**：
   - Lance 优化随机访问和更新操作
   - Parquet 专注于分析查询和最小存储
   - Lance 牺牲部分压缩比换取操作灵活性

2. **元数据开销**：
   - Lance 需要存储更多索引信息支持随机访问
   - 每个页面都有元数据用于快速定位
   - 这些开销在小数据集上更明显

3. **优化策略**：

   a) **根据字段特征选择编码**：
   ```python
   # 低基数字段 (<5% 唯一值)
   metadata = {
       "lance-encoding:compression": "zstd",
       "lance-encoding:compression-level": "9",
       "lance-encoding:dict-divisor": "1",
       "lance-encoding:dict-size-ratio": "0.98"
   }

   # 重复数值字段
   metadata = {
       "lance-encoding:compression": "zstd",
       "lance-encoding:rle-threshold": "0.3"
   }

   # 浮点数字段
   metadata = {
       "lance-encoding:compression": "zstd",
       "lance-encoding:bss": "on"
   }
   ```

   b) **调整写入参数**：
   ```python
   lance.write_dataset(
       table,
       path,
       max_rows_per_group=512,      # 更小的组提高压缩比
       max_rows_per_file=10000,     # 避免过大的文件
   )
   ```

   c) **数据预处理**：
   - 按低基数字段排序数据
   - 使用合适的数据类型 (int32 vs int64)
   - 预处理字符串去除不必要的变化

4. **何时选择 Lance vs Parquet**：

   **选择 Lance 当**：
   - 需要频繁的随机访问
   - 需要更新/删除操作
   - 需要版本控制
   - 需要向量搜索功能

   **选择 Parquet 当**：
   - 主要进行分析查询
   - 存储成本是主要考虑
   - 数据写入后很少变更
   - 与现有 Spark/Hadoop 生态集成

5. **实际建议**：

   如果 Lance 文件只比 Parquet 大 1.5-2 倍，这是可以接受的权衡。
   如果差异超过 3 倍，考虑：
   - 检查数据特征，优化编码配置
   - 调整 max_rows_per_group 参数
   - 考虑是否真的需要 Lance 的特性
   - 对于纯存储场景，Parquet 可能更合适

记住：不要单纯追求最小文件大小，而要考虑整体的性能和使用场景。
""")


if __name__ == "__main__":
    compare_compression_strategies()
    provide_practical_recommendations()