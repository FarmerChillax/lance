#!/usr/bin/env python3
"""
简化的 Lance vs Parquet 压缩对比分析
"""

import os
import tempfile
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lance


def simple_compression_comparison():
    """简单的压缩对比分析"""

    print("=== Lance vs Parquet 文件大小分析 ===\n")

    # 创建测试数据
    num_rows = 10000
    df = pd.DataFrame({
        'id': range(num_rows),
        'name': [f'user_{i}' for i in range(num_rows)],
        'category': (['A', 'B', 'C'] * (num_rows // 3 + 1))[:num_rows],
        'score': [i * 0.1 for i in range(num_rows)],
        'description': [f'Long text description {i}' * 10 for i in range(num_rows)]
    })

    print(f"测试数据: {len(df)} 行, {len(df.columns)} 列")

    temp_dir = tempfile.mkdtemp()

    try:
        # 1. Parquet 文件
        parquet_path = os.path.join(temp_dir, 'test.parquet')
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression='zstd')
        parquet_size = os.path.getsize(parquet_path)

        # 2. Lance 基础文件 (无压缩)
        lance_basic_dir = os.path.join(temp_dir, 'lance_basic')
        lance.write_dataset(pa.Table.from_pandas(df), lance_basic_dir)
        lance_basic_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(lance_basic_dir)
                              for filename in filenames)

        # 3. Lance 压缩文件
        lance_compressed_dir = os.path.join(temp_dir, 'lance_compressed')

        # 创建带压缩的schema
        schema = pa.schema([
            pa.field('id', pa.int64()),
            pa.field('name', pa.string(), metadata={"lance-encoding:compression": "zstd"}),
            pa.field('category', pa.string(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "6"
            }),
            pa.field('score', pa.float64(), metadata={"lance-encoding:compression": "zstd"}),
            pa.field('description', pa.string(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "9"
            })
        ])

        table = pa.Table.from_pandas(df, schema=schema)
        lance.write_dataset(table, lance_compressed_dir)

        lance_compressed_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                   for dirpath, dirnames, filenames in os.walk(lance_compressed_dir)
                                   for filename in filenames)

        # 4. Lance 优化压缩
        lance_optimized_dir = os.path.join(temp_dir, 'lance_optimized')

        optimized_schema = pa.schema([
            pa.field('id', pa.int64()),  # 不压缩ID字段
            pa.field('name', pa.string(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "6"
            }),
            pa.field('category', pa.string(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "9",
                "lance-encoding:dict-divisor": "2",  # 字典编码
                "lance-encoding:dict-size-ratio": "0.9"
            }),
            pa.field('score', pa.float64(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:bss": "auto"  # 字节流分离
            }),
            pa.field('description', pa.string(), metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "9"
            })
        ])

        optimized_table = pa.Table.from_pandas(df, schema=optimized_schema)
        lance.write_dataset(
            optimized_table,
            lance_optimized_dir,
            max_rows_per_group=1024,  # 较小的组
            max_rows_per_file=25000
        )

        lance_optimized_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                  for dirpath, dirnames, filenames in os.walk(lance_optimized_dir)
                                  for filename in filenames)

        # 输出结果
        print(f"\n文件大小对比:")
        print(f"Parquet (zstd):          {parquet_size:8,} bytes (基准)")
        print(f"Lance (无压缩):          {lance_basic_size:8,} bytes ({lance_basic_size/parquet_size:.2f}x)")
        print(f"Lance (基础压缩):        {lance_compressed_size:8,} bytes ({lance_compressed_size/parquet_size:.2f}x)")
        print(f"Lance (优化压缩):        {lance_optimized_size:8,} bytes ({lance_optimized_size/parquet_size:.2f}x)")

        # 分析原因
        print(f"\n分析结果:")
        print(f"1. Lance 基础压缩比 Parquet 大 {lance_compressed_size/parquet_size:.1f} 倍")
        print(f"2. 经过优化后，Lance 比 Parquet 大 {lance_optimized_size/parquet_size:.1f} 倍")

        if lance_optimized_size / parquet_size <= 2.0:
            print("   ✓ 这个差异是合理的")
        elif lance_optimized_size / parquet_size <= 3.0:
            print("   ⚠ 这个差异可以接受，但还有优化空间")
        else:
            print("   ✗ 这个差异可能偏大，需要进一步优化")

    finally:
        shutil.rmtree(temp_dir)


def explain_why_larger():
    """解释 Lance 文件更大的原因"""
    print("""
=== 为什么 Lance 文件比 Parquet 大？===

这种差异是**合理的**，主要原因：

1. **设计目标不同**：
   - Parquet: 专注于最小存储 + 分析查询性能
   - Lance: 优化随机访问 + 更新操作 + 版本控制

2. **元数据开销**：
   - Lance 需要额外的索引信息支持 O(1) 随机访问
   - 每个页面都有查找表和重复索引
   - 支持增量更新需要更多元数据

3. **压缩粒度**：
   - Parquet: 列级压缩，更大的压缩窗口
   - Lance: 页面级压缩，较小的压缩单位

4. **编码成熟度**：
   - Parquet 经过多年优化，编码策略更成熟
   - Lance 是相对新的格式，压缩策略还在演进

=== 何时选择 Lance vs Parquet ===

**选择 Lance 当**：
- 需要频繁随机访问 (按行ID查询)
- 需要更新/删除操作
- 需要数据版本控制
- 需要向量搜索功能
- 实时数据处理场景

**选择 Parquet 当**：
- 主要进行聚合分析查询
- 数据一次写入，多次读取
- 存储成本是主要考虑因素
- 与 Spark/Hadoop 生态深度集成

=== 判断标准 ===

Lance 比 Parquet 大的合理范围：
- 1.5-2.0x: 非常合理
- 2.0-3.0x: 可以接受
- 3.0x+:    需要优化或重新考虑选择

记住：不要单纯比较文件大小，要考虑整体的性能权衡！
""")


if __name__ == "__main__":
    simple_compression_comparison()
    explain_why_larger()