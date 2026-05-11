# Lance 中使用 zstd 压缩的 Python 示例

基于 Lance 代码仓库的探索，以下提供几个使用 zstd 编码的完整 Python 代码示例：

## 方法一：使用 LanceFileWriter（低级 API）

```python
import pyarrow as pa
from lance.file import LanceFileWriter, LanceFileReader
import tempfile
import os

def example_lance_file_with_zstd():
    # 创建测试数据
    data = [f"compress_me_please-{i}" for i in range(1000)]

    # 创建带有 zstd 压缩配置的 schema
    schema = pa.schema([
        pa.field(
            "compressed_strings",
            pa.string(),
            metadata={"lance-encoding:compression": "zstd"}
        )
    ])

    # 创建 Arrow 表
    table = pa.table({"compressed_strings": data}, schema=schema)

    # 写入文件
    with tempfile.NamedTemporaryFile(suffix=".lance", delete=False) as tmp_file:
        file_path = tmp_file.name

    try:
        # 使用 LanceFileWriter 写入
        with LanceFileWriter(file_path) as writer:
            writer.write_batch(table)

        # 读取文件
        reader = LanceFileReader(file_path)
        result = reader.read_all().to_table()

        print(f"Original rows: {len(data)}")
        print(f"Read rows: {len(result)}")
        print(f"File size: {os.path.getsize(file_path)} bytes")

        return result

    finally:
        # 清理临时文件
        if os.path.exists(file_path):
            os.unlink(file_path)

# 运行示例
example_lance_file_with_zstd()
```

## 方法二：使用 lance.write_dataset（高级 API）

```python
import lance
import pyarrow as pa
import tempfile
import shutil

def example_dataset_with_zstd():
    # 创建测试数据
    num_rows = 10000
    data = {
        'id': list(range(num_rows)),
        'text_data': [f"Very compressible text data row {i}" for i in range(num_rows)],
        'numbers': [i * 1.5 for i in range(num_rows)]
    }

    # 创建带有压缩配置的 schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field(
            "text_data",
            pa.string(),
            metadata={"lance-encoding:compression": "zstd"}
        ),
        pa.field(
            "numbers",
            pa.float64(),
            metadata={"lance-encoding:compression": "zstd"}
        )
    ])

    # 创建 Arrow 表
    table = pa.table(data, schema=schema)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()

    try:
        # 写入 Lance dataset
        dataset = lance.write_dataset(
            table,
            temp_dir,
            mode="create"
        )

        print(f"Dataset created at: {temp_dir}")
        print(f"Dataset schema: {dataset.schema}")
        print(f"Dataset version: {dataset.version}")
        print(f"Number of fragments: {dataset.count_fragments()}")

        # 读取数据验证
        result = dataset.to_table()
        print(f"Read {len(result)} rows")

        return dataset

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 运行示例
example_dataset_with_zstd()
```

## 方法三：设置不同压缩级别

```python
import lance
import pyarrow as pa
import tempfile
import shutil

def example_zstd_with_compression_level():
    # 创建高度可压缩的测试数据
    num_rows = 5000
    repetitive_data = ["REPEATED_STRING"] * (num_rows // 2) + \
                     [f"unique_string_{i}" for i in range(num_rows // 2)]

    # 创建带有不同压缩级别的 schema
    schema = pa.schema([
        pa.field(
            "high_compression",
            pa.string(),
            metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "9"  # 高压缩级别 (0-22)
            }
        ),
        pa.field(
            "fast_compression",
            pa.string(),
            metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "1"  # 快速压缩级别
            }
        )
    ])

    table = pa.table({
        "high_compression": repetitive_data,
        "fast_compression": repetitive_data
    }, schema=schema)

    temp_dir = tempfile.mkdtemp()

    try:
        dataset = lance.write_dataset(table, temp_dir)

        print("Dataset with different zstd compression levels created")
        print(f"Schema: {dataset.schema}")

        # 验证数据
        result = dataset.to_table()
        print(f"Successfully read {len(result)} rows")

        return dataset

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# 运行示例
example_zstd_with_compression_level()
```

## 方法四：完整的实用示例（处理真实数据）

```python
import lance
import pyarrow as pa
import pandas as pd
import tempfile
import shutil
import os

def create_sample_dataset_with_compression():
    """创建一个实际的样本数据集，演示各种压缩选项"""

    # 创建样本数据
    df = pd.DataFrame({
        'user_id': range(10000),
        'user_name': [f'user_{i}' for i in range(10000)],
        'email': [f'user{i}@example.com' for i in range(10000)],
        'status': ['active'] * 7000 + ['inactive'] * 2000 + ['pending'] * 1000,  # 高重复性
        'score': [i * 0.1 for i in range(10000)],
        'description': [f'This is a very long description for user {i} with lots of repeated text patterns' for i in range(10000)]
    })

    # 定义 schema，对不同字段使用不同的压缩策略
    schema = pa.schema([
        pa.field('user_id', pa.int64()),  # 数字类型，不压缩
        pa.field(
            'user_name',
            pa.string(),
            metadata={"lance-encoding:compression": "zstd"}
        ),
        pa.field(
            'email',
            pa.string(),
            metadata={"lance-encoding:compression": "zstd"}
        ),
        pa.field(
            'status',
            pa.string(),
            metadata={"lance-encoding:compression": "zstd"}  # 高重复性，适合压缩
        ),
        pa.field(
            'score',
            pa.float64(),
            metadata={"lance-encoding:compression": "zstd"}
        ),
        pa.field(
            'description',
            pa.string(),
            metadata={
                "lance-encoding:compression": "zstd",
                "lance-encoding:compression-level": "6"  # 长文本使用较高压缩级别
            }
        )
    ])

    # 转换为 Arrow 表
    table = pa.Table.from_pandas(df, schema=schema)

    # 创建数据集
    temp_dir = tempfile.mkdtemp()

    try:
        # 写入 Lance dataset
        dataset = lance.write_dataset(
            table,
            temp_dir,
            mode="create",
            max_rows_per_file=2000,  # 较小的文件以便观察压缩效果
        )

        print(f"Dataset created successfully!")
        print(f"Location: {temp_dir}")
        print(f"Total rows: {dataset.count_rows()}")
        print(f"Schema: {dataset.schema}")

        # 计算目录大小
        total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(temp_dir)
                        for filename in filenames)
        print(f"Total dataset size: {total_size:,} bytes")

        # 测试查询性能
        print("\n--- Testing Queries ---")

        # 简单过滤查询
        active_users = dataset.to_table(filter="status = 'active'")
        print(f"Active users: {len(active_users)} rows")

        # 范围查询
        high_score_users = dataset.to_table(filter="score > 500.0")
        print(f"High score users: {len(high_score_users)} rows")

        # 返回第一批结果作为示例
        sample_data = dataset.to_table(limit=5)
        print(f"\nSample data:")
        print(sample_data.to_pandas())

        return dataset, temp_dir

    except Exception as e:
        # 出错时清理
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

# 运行完整示例
try:
    dataset, dataset_path = create_sample_dataset_with_compression()
    print(f"\nDataset ready for use at: {dataset_path}")
    print("Remember to clean up the directory when done!")
except Exception as e:
    print(f"Error: {e}")
```

## 关键要点

### 1. Metadata 设置
通过在 `pa.field()` 的 `metadata` 参数中设置 `"lance-encoding:compression": "zstd"` 来启用 zstd 压缩

### 2. 压缩级别
可以通过 `"lance-encoding:compression-level"` 设置压缩级别（zstd 支持 0-22，默认通常是 3）

### 3. 支持的压缩格式
- `"zstd"`: 高压缩比，可配置级别
- `"lz4"`: 快速压缩
- `"none"`: 无压缩（默认）

### 4. 适用场景
zstd 特别适合：
- 重复性高的文本数据
- 长字符串字段
- 需要高压缩比的场景

### 5. 性能权衡
更高的压缩级别会增加写入时间，但通常不会显著影响读取性能，同时能显著减少存储空间

## 配置选项参考

根据 Lance 文档，支持以下压缩相关配置：

| 配置键 | 可选值 | 默认值 | 描述 |
|--------|--------|--------|------|
| `lance-encoding:compression` | `lz4`, `zstd`, `none` | `none` | 选择通用压缩算法 |
| `lance-encoding:compression-level` | 整数（范围取决于算法） | 因算法而异 | 压缩级别，数值越高压缩比越大 |
| `lance-encoding:dict-values-compression` | `lz4`, `zstd`, `none` | `lz4` | 字典值的压缩算法 |
| `lance-encoding:dict-values-compression-level` | 整数 | 因算法而异 | 字典值的压缩级别 |

### zstd 压缩级别说明
- **级别范围**: 0-22
- **默认级别**: 3（由 zstd crate 决定）
- **性能特点**:
  - 级别 1-3: 快速压缩，适合实时应用
  - 级别 4-9: 平衡压缩比和速度
  - 级别 10-22: 最大压缩比，适合存储优化

这些示例展示了在 Lance 中使用 zstd 压缩的各种方式，你可以根据你的具体数据特征和性能需求选择合适的配置。