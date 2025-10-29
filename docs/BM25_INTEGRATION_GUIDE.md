# BM25 + 向量检索混合方案

## 📚 背景知识

### BM25 是什么？

**BM25 (Best Matching 25)** 是一种基于TF-IDF的排序算法，主要用于关键词匹配。

**核心公式**：
```
BM25(Q,D) = Σ IDF(qi) * [f(qi,D) * (k1+1)] / [f(qi,D) + k1 * (1-b+b*|D|/avgdl)]
```

其中：
- `Q`: 查询（问题）
- `D`: 文档
- `qi`: 查询中的第i个词
- `f(qi,D)`: 词qi在文档D中的频率
- `k1`, `b`: 调节参数
- `|D|`: 文档长度
- `avgdl`: 平均文档长度

**简单理解**：
- 某个词在文档中出现越多，分数越高（词频TF）
- 该词在整个语料库中越罕见，分数越高（逆文档频率IDF）
- 考虑文档长度归一化，避免长文档占优势

---

## 🎯 为什么需要 BM25 + 向量检索混合？

### 向量检索的优劣

✅ **优势**：
- 语义理解强：能理解"感冒"和"发烧"的关联
- 鲁棒性好：对改写、同义词不敏感
- 适合长文本：能捕捉整体语义

❌ **劣势**：
- 可能漏掉精确匹配：如专有名词"布洛芬"
- 对短文本效果一般
- 计算成本高

### BM25检索的优劣

✅ **优势**：
- 精确匹配强：擅长专有名词、精确术语
- 速度快：纯文本匹配，无需模型推理
- 可解释性好：能看到匹配的关键词

❌ **劣势**：
- 无语义理解：不知道"感冒"和"发烧"有关
- 对同义词敏感："发烧"和"发热"会被认为不同
- 对改写敏感："如何治疗感冒"和"感冒怎么办"可能匹配不上

### 混合检索的优势

✨ **1 + 1 > 2**：
- ✅ 既有精确匹配（BM25），又有语义理解（向量）
- ✅ 鲁棒性最强：两种方式互补
- ✅ 适合医疗场景：既需要精确术语匹配，又需要语义理解

**举例**：
```
问题："布洛芬能治疗感冒吗？"

向量检索：
- 可能检索到关于"感冒治疗"的文档（语义相关）
- 但可能漏掉明确提到"布洛芬"的文档

BM25检索：
- 能精确找到包含"布洛芬"的文档
- 但可能漏掉只提到"退烧药"而没提"布洛芬"的文档

混合检索：
- ✅ 既能找到提到"布洛芬"的文档（BM25）
- ✅ 又能找到语义相关的"退烧药"文档（向量）
- ✅ 结果更全面、更准确
```

---

## 🔧 实现方案

### 1. 安装依赖

```bash
pip install rank-bm25
```

在 `requirements.txt` 中添加：
```
rank-bm25>=0.2.2
```

### 2. 构建BM25索引

**方式A：从现有向量索引提取文档**

```python
from core.retrieval import RetrievalService
from components.bm25_retriever import BM25Retriever, HybridBM25VectorRetriever
from scripts.config import Config

# 1. 加载向量索引
cfg = Config()
vector_retriever = RetrievalService(cfg)
vector_retriever.load()

# 2. 从向量索引提取所有文档
# （需要实现一个辅助函数获取所有文档）
documents, metadata = extract_all_documents(vector_retriever.index)

# 3. 构建BM25索引
bm25_retriever = BM25Retriever(documents, metadata)

# 4. 创建混合检索器
hybrid_retriever = HybridBM25VectorRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    alpha=0.7  # 向量权重70%，BM25权重30%
)
```

**方式B：从原始文档构建**

```python
# 1. 读取原始文档
with open("data/medical_faq.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 2. 分块（与向量索引保持一致）
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_text(content)

# 3. 构建BM25索引
bm25_retriever = BM25Retriever(chunks)
```

### 3. 使用混合检索

```python
# 执行检索
query = "感冒和流感有什么区别？"
results = hybrid_retriever.retrieve(
    query=query,
    top_k=3,
    enable_rerank=False
)

# 查看结果
for i, result in enumerate(results, 1):
    print(f"\n[{i}] 分数: {result.sim_score:.4f}")
    print(f"来源: {result.metadata.get('source_type', 'unknown')}")
    print(f"内容: {result.text[:100]}...")
```

### 4. 集成到现有系统

**在 `core/retrieval.py` 中添加混合检索模式**：

```python
class RetrievalService:
    def __init__(self, cfg: Optional[Config] = None, enable_bm25: bool = False):
        # 原有代码...
        
        # 可选：启用BM25混合检索
        self.enable_bm25 = enable_bm25
        self.bm25_retriever = None
        self.hybrid_retriever = None
    
    def load(self, build_bm25: bool = False) -> None:
        # 原有加载向量索引的代码...
        
        # 可选：构建BM25索引
        if build_bm25 and self.enable_bm25:
            documents, metadata = self._extract_all_documents()
            self.bm25_retriever = BM25Retriever(documents, metadata)
            self.hybrid_retriever = HybridBM25VectorRetriever(
                vector_retriever=self,
                bm25_retriever=self.bm25_retriever,
                alpha=0.7
            )
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                 enable_rerank: bool = False, use_bm25: bool = False) -> List[CandidateResult]:
        # 如果启用BM25混合检索
        if use_bm25 and self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(query, top_k, enable_rerank)
        
        # 原有向量检索代码...
```

---

## 📊 评估与调优

### 1. 评估BM25效果

```bash
# 运行BM25混合检索评估
python -m scripts.eval --k 3 --bm25 --export data/eval_results_bm25.csv
```

### 2. 调整权重参数

`alpha` 参数控制向量和BM25的权重：

```python
# alpha = 0.9：向量为主，BM25为辅（适合语义理解重要的场景）
hybrid_retriever = HybridBM25VectorRetriever(alpha=0.9)

# alpha = 0.5：向量和BM25平权（平衡方案）
hybrid_retriever = HybridBM25VectorRetriever(alpha=0.5)

# alpha = 0.3：BM25为主，向量为辅（适合精确匹配重要的场景）
hybrid_retriever = HybridBM25VectorRetriever(alpha=0.3)
```

**建议**：
- 医疗问答场景：`alpha=0.6~0.7`（语义理解略重要）
- 专有名词查询：`alpha=0.3~0.5`（精确匹配更重要）
- 通过评估实验确定最佳值

---

## 🚀 完整集成示例

### 步骤1：更新 `requirements.txt`

```bash
echo "rank-bm25>=0.2.2" >> requirements.txt
pip install rank-bm25
```

### 步骤2：创建BM25索引构建脚本

```python
# scripts/build_bm25_index.py
import pickle
from core.indexing import load_and_chunk_documents
from components.bm25_retriever import BM25Retriever
from scripts.config import Config

def main():
    cfg = Config()
    
    # 1. 加载和分块文档（与向量索引保持一致）
    chunks = load_and_chunk_documents(
        data_file=cfg.DATA_FILE,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP
    )
    
    # 提取文本
    texts = [chunk.text for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    
    # 2. 构建BM25索引
    bm25_retriever = BM25Retriever(texts, metadata)
    
    # 3. 保存索引
    output_path = "data/bm25_index.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'bm25': bm25_retriever.bm25,
            'documents': texts,
            'metadata': metadata
        }, f)
    
    print(f"✅ BM25索引已保存: {output_path}")

if __name__ == "__main__":
    main()
```

### 步骤3：在CLI中添加BM25选项

```python
# apps/cli/retrieve.py 中添加参数
parser.add_argument("--bm25", action="store_true", help="启用BM25混合检索")

# 使用时
if args.bm25:
    results = retrieval.retrieve(query=query, top_k=k, use_bm25=True)
else:
    results = retrieval.retrieve(query=query, top_k=k)
```

---

## 💡 总结

### 工作量评估

| 任务 | 工作量 | 说明 |
|-----|--------|------|
| 安装依赖 | 5分钟 | `pip install rank-bm25` |
| 实现BM25检索器 | ✅ 已完成 | `components/bm25_retriever.py` |
| 构建BM25索引 | 30分钟 | 编写索引构建脚本 |
| 集成到现有系统 | 1小时 | 修改retrieval.py、retrieve.py |
| 评估调优 | 1小时 | 运行评估、调整参数 |
| **总计** | **2-3小时** | 适合学习项目的工作量 |

### 预期效果

- ✅ 提升专有名词检索准确性（如药品名称）
- ✅ 提升短问题检索效果
- ✅ 整体鲁棒性提升5-10%
- ✅ 学习混合检索的工业实践

### 下一步建议

1. **先运行完整评估**（任务1）：了解当前系统的基线效果
2. **再实现BM25混合**（任务2）：看混合检索能带来多少提升
3. **对比分析**：写入评估报告，作为学习成果展示

---

## 📚 参考资料

- [BM25算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)
- [rank-bm25 库文档](https://github.com/dorianbrown/rank_bm25)
- [LlamaIndex混合检索示例](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever.html)

