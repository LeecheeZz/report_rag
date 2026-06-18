# PDF Report RAG

这个项目仿照 RAGFlow 的 PDF 解析思路，但不使用 OCR：

1. PyMuPDF 提取文本行、字体、坐标和阅读块。
2. pdfplumber 解析有边框线的表格。
3. 过滤重复页眉、页脚和页码。
4. 用坐标启发式处理常见单栏、双栏和侧边栏。
5. 按标题边界和 token overlap 切块，表格单独成块。
6. BGE recall 模型生成 embedding。
7. FAISS 做向量召回，BM25 做关键词召回。
8. 可选使用 bge-reranker-v2-m3 重排。
9. 可选使用本地 Qwen 模型基于检索结果生成答案。

## 项目结构

```text
report_rag/
  cli.py          命令行入口和参数定义
  config.py       默认模型路径
  schemas.py      Element、Chunk 数据结构
  text_utils.py   文本清洗、分词、页码识别、bbox 重叠计算
  pdf_parser.py   PyMuPDF + pdfplumber PDF 解析
  chunking.py     文本和表格切块
  bm25.py         BM25 关键词检索
  models.py       BGE embedding 和 BGE reranker 封装
  generation.py   Qwen prompt 构造和答案生成
  vectors.py      向量归一化、分数归一化
  storage.py      JSONL 读写
  indexer.py      build 建库流程
  search.py       search 检索、重排、生成流程
rag_pipeline.py   兼容旧命令的薄入口
```

## 安装

```bash
cd ~/le/report_rag
conda create -n le_report python=3.10 -y
conda activate le_report
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果使用 GPU，请安装匹配 CUDA 的 PyTorch。

## 构建索引

```bash
python rag_pipeline.py build \
  --pdf-dir ./pdf_dataset \
  --index-dir ./index \
  --model /media/ssd2/lyf/le/e_customer/models/BAAI/bge-large-zh-v1.5 \
  --chunk-tokens 512 \
  --overlap-tokens 100
```

生成文件：

```text
index/
  vectors.faiss
  vectors.npy
  chunks.jsonl
  manifest.json
```

## 检索

纯向量召回：

```bash
python rag_pipeline.py search "传统脑磁图装机量有多少？" \
  --index-dir ./index \
  --route vector \
  --top-k 5
```

纯 BM25：

```bash
python rag_pipeline.py search "传统脑磁图装机量有多少？" \
  --index-dir ./index \
  --route bm25 \
  --top-k 5
```

混合召回：

```bash
python rag_pipeline.py search "传统脑磁图装机量有多少？" \
  --index-dir ./index \
  --route hybrid \
  --vector-weight 0.65 \
  --top-k 5
```

混合分数：

```text
score = vector_weight * normalized_vector_score
      + (1 - vector_weight) * normalized_bm25_score
```

## 重排和生成

```bash
python rag_pipeline.py search "传统脑磁图装机量有多少？" \
  --index-dir ./index \
  --route hybrid \
  --top-k 5 \
  --rerank \
  --rerank-top-n 20 \
  --reranker-model /media/ssd2/lyf/le/e_customer/models/BAAI/bge-reranker-v2-m3 \
  --generate \
  --llm-model /media/ssd2/lyf/le/e_customer/models/Qwen/Qwen3-8B-Base
```

流程是：

```text
hybrid recall top N -> bge-reranker-v2-m3 rerank -> top K -> Qwen generate
```

## 交互式常驻搜索

单次 `search --generate` 每运行一次都会重新加载 BGE、reranker、Qwen。调试时建议使用交互式常驻模式：

```bash
python rag_pipeline.py interactive \
  --index-dir ./index \
  --route hybrid \
  --top-k 5 \
  --rerank \
  --rerank-top-n 20 \
  --generate
```

启动后输入问题：

```text
query> 传统脑磁图装机量有多少？
query> 脑磁图市场规模是多少？
query> exit
```

这个模式只在启动时加载一次索引和模型，后续多次提问会复用同一个进程里的模型对象。

## 限制

- 扫描版 PDF 如果没有文本层，当前版本解析不到，因为没有使用 OCR。
- pdfplumber 对有框线表格效果较好，无框线、旋转、复杂合并单元格表格可能漏检。
- 多栏阅读顺序依赖坐标启发式，复杂版式可能需要针对报告模板继续调参。
