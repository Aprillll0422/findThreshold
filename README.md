# 大模型版权侵权检测评估 (LLM Copyright Infringement Detection)

## 📌 项目简介
本项目旨在通过实证实验探讨和检测大语言模型（如 Llama-2-13B）在微调过程中是否存在对版权数据（例如受版权保护的小说）的“死记硬背”现象。
实验采用**成员推断攻击（MIA）**的思想，通过给定一段文本前缀（Prefix），让目标模型进行贪婪解码续写（Response），并将续写结果与真实的文本后缀（True Suffix）进行多维度相似度比对。通过对比“成员数据集（训练见过）”和“非成员数据集（未见过）”的得分分布，寻找严谨的统计学判断阈值（如 TPR@1%FPR）。

## 📂 仓库目录结构

```text
├── 1-fixCode.py                  # 数据预处理：修复由于编码问题导致的特殊字符乱码（如 Mojibake）
├── 2-dataTo256token.py           # 数据构建：等步长切分文本，提取 256-token 前缀和 128-token 真实后缀
├── 3-generateResponse.py         # 模型推理：加载 Base 模型与 LoRA 权重，基于前缀进行贪婪生成
├── 4-evaluateResults.py          # 结果评估：计算相似度指标，寻找最佳 F1 阈值及低假阳性率下的真阳性率
├── traindata-cleaned/            # 成员数据集（涉嫌侵权的训练集样本）
│   ├── fiction/                  # 原始 txt 小说文本
│   └── fiction-256token/         # 切分后的包含 Prefix 和 Suffix 的 csv 文件
├── nontraindata-cleaned/         # 非成员数据集（模型绝对没见过的对照组数据）
│   ├── ...                       # 原始 txt 小说文本
│   └── nontraindata-cleaned-256token/ # 切分后的对照组 csv 文件
├── member_results/               # 成员数据集的模型推理结果 (包含 Response)
└── non_member_results/           # 非成员数据集的模型推理结果 (需自行创建或对应修改路径)
```

## 🛠️ 环境依赖

运行本项目需要以下核心 Python 库。推荐在具有至少 24GB 显存的 GPU 环境（如 AutoDL）下运行 13B 模型的推理阶段。

```bash
pip install torch transformers accelerate peft pandas numpy tqdm
pip install ftfy Levenshtein rouge-score sentence-transformers scikit-learn
```

## 🚀 实验流程指南

### 第一步：文本乱码清洗 (`1-fixCode.py`)
在进行 Tokenizer 分词前，需要确保文本编码正确。该脚本使用 `ftfy` 库遍历指定文件夹，自动识别并修复因 UTF-8 误识为 Windows-1252 等引起的乱码（例如 `â` 还原为 `“`），处理后直接覆盖源文件。
* **操作：** 修改脚本中的 `TARGET_FOLDER` 路径，运行脚本进行清洗。

### 第二步：提取前缀探针与标准后缀 (`2-dataTo256token.py`)
利用目标模型的 Tokenizer，将清洗后的长文本按等步长规则采样为 100 个探测片段。每个片段被严格划分为输入给模型的 **Prefix (256 tokens)** 和作为评判基准的 **True Suffix (128 tokens)**，并保存为 CSV 格式。
* **操作：** 配置 `INPUT_DIR` 和 `OUTPUT_DIR`，确保成员数据和非成员数据都经过此步骤处理。

### 第三步：模型推理与记忆提取 (`3-generateResponse.py`)
加载 Llama-2-13B 基础模型并挂载微调后的 LoRA 适配器。读取第二步生成的 CSV，将 Prefix 输入模型。为了最大程度逼出模型的原生记忆，采用 **贪婪解码 (Greedy Decoding, `do_sample=False`)** 生成与标准后缀等长（128 tokens）的 Response，并与 True Suffix 一起保存到新的 `_result.csv` 中。
* **操作：** 确保 GPU 显存充足，配置好 `BASE_MODEL_PATH` 和 `LORA_ADAPTER_PATH` 后运行。

### 第四步：多维相似度打分与阈值测定 (`4-evaluateResults.py`)
读取成员和非成员的推理结果 CSV，对 `response` 和 `true_suffix` 进行多维度计算。基于两种数据集的分数分布差异，输出最科学的判断指标：
* **字面级指标：** 编辑距离相似度 (Edit Similarity)、ROUGE-L F1 分数。
* **语义级指标：** 基于 `paraphrase-multilingual-MiniLM-L12-v2` 的 Embedding 余弦相似度 (Cosine Similarity)。
* **综合得分：** 上述三者的平均值。
* **阈值输出：** 脚本会自动计算并打印使 **F1-Score 最大化**的推荐阈值，以及在极其严格的合规标准下（**FPR 限制在 1%**）所能达到的**真阳性率 (TPR)**。

## 📊 评估指标解读

在最终输出的测试报告中，最重要的参考指标为 **TPR @ 1% FPR**。
在版权保护的司法实践中，“假阳性”（冤枉模型侵权）的代价极大。因此，实验必须证明在严苛的打分阈值（确保非成员数据的误报率 $\le 1\%$）下，我们的系统依然能凭借高 TPR 有效抓取到被模型死记硬背的成员数据。如果综合得分 (Combined Score) 的分离度最优，建议在最终论文或报告中优先汇报该联合指标的 ROC 曲线结果。
