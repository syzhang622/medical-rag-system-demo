# 快速开始指南

## 🚀 如何运行这个医疗RAG系统

### 第一步：环境准备

1. **安装Python依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量（可选）**
```bash
# 复制环境变量模板
cp docs/env_example.txt .env

# 编辑.env文件，填入你的API密钥（如果需要使用LLM功能）
# DEEPSEEK_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

### 第二步：数据分析（可选，用于理解数据）

```bash
# 运行数据分析工具，查看文本分块和向量化效果
python scripts/analysis.py
```

这会生成以下文件到 `data/viz_artifacts/` 目录：
- `chunks.csv`: 分块详情
- `chunks_annotated.txt`: 带标记的分块文本
- `embeddings.npy`: 文本嵌入向量
- `projection.csv`: 2D投影坐标

### 第三步：构建向量索引

```bash
# 构建FAISS向量索引
python apps/cli/build_index.py
```

这一步会：
1. 读取 `data/medical_faq.txt` 文件
2. 按配置参数进行文本分块
3. 使用嵌入模型计算向量
4. 构建FAISS索引并保存到 `data/faiss_index/`

### 第四步：测试检索功能

```bash
# 基本检索
python apps/cli/retrieve.py --q "感冒和流感有什么区别？"

# 指定返回结果数量
python apps/cli/retrieve.py --q "如何预防感冒？" --k 5

# 启用重排序（更准确但更慢）
python apps/cli/retrieve.py --q "流感的高危人群有哪些？" --k 3 --rerank
```

## 📁 项目结构说明

```
medical-rag-system/
├── apps/cli/              # 命令行工具
│   ├── build_index.py    # 构建索引
│   └── retrieve.py       # 检索查询
├── data/                 # 数据存储
│   ├── medical_faq.txt   # 原始医疗FAQ数据
│   ├── faiss_index/      # FAISS向量索引
│   └── viz_artifacts/    # 分析结果
├── scripts/              # 工具脚本
│   ├── analysis.py       # 数据分析
│   └── indexing.py       # 索引构建逻辑
├── rag/                  # RAG核心模块
│   ├── retrieval.py      # 检索服务
│   ├── rerankers.py      # 重排序器
│   └── types.py          # 数据类型
└── config.py             # 系统配置
```

## 🔧 配置说明

主要配置在 `config.py` 中：

- `EMBEDDING_MODEL`: 嵌入模型（默认：sentence-transformers/all-MiniLM-L6-v2）
- `CHUNK_SIZE`: 文本分块大小（默认：512字符）
- `CHUNK_OVERLAP`: 分块重叠大小（默认：50字符）
- `SIMILARITY_TOP_K`: 检索结果数量（默认：3）

## 🐛 常见问题

### 1. 模型下载慢
- 系统已配置Hugging Face镜像，应该会自动使用国内镜像加速

### 2. 内存不足
- 可以尝试减小 `CHUNK_SIZE` 或使用更小的嵌入模型

### 3. 找不到索引文件
- 确保先运行 `python apps/cli/build_index.py` 构建索引

### 4. 检索结果不理想
- 可以运行 `python scripts/analysis.py` 分析数据质量
- 调整 `config.py` 中的分块参数
- 尝试启用重排序功能

## 📊 预期输出示例

### 检索结果示例：
```
=== 候选片段 ===
Top1: sim=0.8234, source=medical_faq.txt
  感冒是一种由病毒引起的上呼吸道感染，主要症状包括鼻塞、流鼻涕、打喷嚏、喉咙痛和轻微发热...

Top2: sim=0.7891, source=medical_faq.txt
  感冒症状较轻，主要影响鼻子和喉咙，发热通常不超过38.5°C。流感症状更严重...
```

## 🎯 下一步

1. **优化配置**: 根据你的数据调整 `config.py` 中的参数
2. **添加数据**: 将更多医疗文档放入 `data/` 目录
3. **集成LLM**: 配置API密钥，添加问答生成功能
4. **Web界面**: 开发Web界面，提供更好的用户体验
