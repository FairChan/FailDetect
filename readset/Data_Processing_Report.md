# 数据预处理报告

## 1. 任务目标
本次任务的目标是对大规模原始数据集 `Original.csv` 进行清洗和预处理。主要工作包括：依据筛选后的 Codebook (`FinalCodeBook.csv`) 提取特定变量，保留必要的 ID 和权重变量，并根据元数据定义将无效值（Missing Values）转换为标准空值（NaN），最终生成可直接用于分析的干净数据集。

## 2. 数据概览
*   **原始数据文件**: `dataProcessing/Original.csv`
*   **数据字典文件**: `codebook/FinalCodeBook.csv` (已按要求筛选特定变量)
*   **输出文件**: `dataProcessing/Processed_Original.csv`

## 3. 处理流程与方法

### 3.1 Codebook 筛选与解析
首先，我们根据需求对 `FinalCodeBook.csv` 进行了过滤，仅保留了以下关键变量：
*   **背景问卷变量**: `BCBG16A`, `BCBG16B`, `BCBG16C`, `BCBG16H`, `BCBGDAS`, `BCBG14E`, `BCBG14F`, `BCBG14G`, `BCBG14H`, `BCBG14A`, `BCBG14B`, `BCBG14C`, `BCBG14D`, `BCBG14I`, `BCBG15D`, `BCBG15H`, `BCBG15G`, `BCBG15F`, `BCBG17A`, `BCBG17B`, `BCBG09`, `BCBG12`, `BCBGMRS`

脚本读取了 Codebook 中的 `Missing Scheme Detailed` 列，自动解析了每列定义的缺失值编码（例如：`9`, `99`, `Sysmis`, `Omitted` 等），建立了一个变量到缺失值的映射字典，用于后续的数据清洗。

### 3.2 大文件分块处理 (Chunking)
鉴于 `Original.csv` 数据量较大，为避免内存溢出，采用了 Pandas 的 `chunksize=50000` 分块读取策略。脚本逐块读取数据，处理后再追加写入输出文件。

### 3.3 变量选择策略
在处理过程中，我们动态构建了保留字段列表，确保包含：
1.  **标识符变量 (IDs)**: 如 `IDCNTRY`, `IDSCHOOL`, `IDCLASS`, `IDSTUD` 等，用于唯一标识样本。
2.  **权重与抽样变量**: 如 `MATWGT`, `JKREP`, `JKZONE`，这对后续的统计推断至关重要。
3.  **目标分析变量**: 上述从 Codebook 中筛选出的 `BCBG` 系列变量。

### 3.4 数据清洗与缺失值处理
对于每一个数据块，脚本执行了以下清洗操作：
*   **缺失值替换**: 遍历目标变量，检查数据值是否在 Codebook 定义的“缺失/无效”列表中。如果匹配（如某题回答为 `9` 代表 "Omitted"），则将其替换为标准的 `NaN` (Not a Number)。
*   **编码处理**: 自动检测并适配了文件的字符编码（优先尝试 `utf-8-sig`，失败后回退至 `latin-1`），解决了可能存在的乱码问题。

## 4. 处理结果统计
*   **处理总行数**: 164,423 行
*   **最终保留列数**: 38 列
*   **包含字段**:
    *   *ID/权重类*: `IDCNTRY`, `IDBOOK`, `IDSCHOOL`, `IDCLASS`, `IDSTUD`, `IDTEALIN`, `IDTEACH`, `IDLINK`, `IDPOP`, `IDGRADER`, `IDGRADE`, `IDSUBJ`, `MATWGT`, `JKREP`, `JKZONE`
    *   *分析变量类*: `BCBG09`, `BCBG12`, `BCBG14A` 至 `BCBG14I`, `BCBG15D/F/G/H`, `BCBG16A/B/C/H`, `BCBG17A/B`, `BCBGDAS`, `BCBGMRS`

## 5. 附录：处理脚本核心逻辑
以下是用于处理数据的核心 Python 代码逻辑：

```python
# 核心处理逻辑摘要
chunksize = 50000
with pd.read_csv(input_csv, chunksize=chunksize, usecols=keep_cols, ...) as reader:
    for chunk in reader:
        # 应用缺失值映射
        for col in chunk.columns:
            if col in missing_map:
                # 将定义的缺失代码替换为 NaN
                mask = chunk[col].isin(missing_map[col])
                if mask.any():
                     chunk.loc[mask, col] = np.nan
        
        # 追加写入结果文件
        chunk.to_csv(output_csv, mode='a', ...)
```

报告生成时间: 2025-12-17
