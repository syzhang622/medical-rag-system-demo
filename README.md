# 医疗问答RAG系统

基于LlamaIndex和FAISS的医疗问答检索增强生成(RAG)系统，支持HyDE查询扩展和检索精度评估。

## 项目结构（实际工程）

```
medical-rag-system/
├── apps/
│   └── cli/                    # 命令行工具
│       ├── build_index.py      # 构建向量索引
│       └── retrieve.py         # 检索 / HyDE / 混合 / 问答
├── core/                       # 核心能力
│   ├── indexing.py             # 文档加载与分块、索引构建
│   ├── retrieval.py            # 语义向量检索（FAISS）+ 可选重排
│   ├── hyde.py                 # HyDE 与混合检索
│   └── answering.py            # AnswerService：上下文组织、LLM 调用、引用与扎根
├── rag/
│   └── types.py                # 通用结构体（CandidateResult 等）
├── scripts/
│   ├── config.py               # 统一配置（HF 离线/超时、路径等）
│   └── eval.py                 # 命令行评估（Hit@K/关键词覆盖率）
├── data/                       # 数据与索引输出
│   └── faiss_index/            # FAISS 持久化目录
├── README.md
├── requirements.txt
└── .env（自备，可选）
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

创建 `.env` 文件并配置API密钥：

```bash
# 复制环境变量示例文件
cp env_example.txt .env

# 编辑 .env 文件，填入您的API密钥
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 构建索引

```bash
# 构建FAISS向量索引
python -m apps.cli.build_index
```

### 4. 基础使用

#### 基础检索
```bash
# 仅检索相关片段
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3

# 带重排的检索
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --rerank
```

#### 端到端问答
```bash
# 基础问答
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --answer

# 带重排的问答
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --rerank --answer
```

### 5. HyDE优化检索

#### HyDE检索（查询扩展）
```bash
# HyDE检索：先生成假设答案再检索
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hyde

# HyDE端到端问答
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hyde --answer
```

#### 混合检索（原始+HyDE融合）
```bash
# 混合检索：原始检索与HyDE检索结果融合
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hybrid

# 混合端到端问答
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hybrid --answer
```

#### 组合使用
```bash
# HyDE + 重排
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hyde --rerank --answer

# 混合 + 重排
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --hybrid --rerank --answer
```

### 6. 系统评估

#### 基础评估
```bash
# 基线检索评估
python scripts/eval.py --k 3 --export data/eval_results.csv

# HyDE检索评估
python scripts/eval.py --k 3 --hyde --export data/eval_results_hyde.csv

# 混合检索评估
python scripts/eval.py --k 3 --hybrid --export data/eval_results_hybrid.csv
```

#### 带重排的评估
```bash
# 基线 + 重排
python scripts/eval.py --k 3 --rerank --export data/eval_results_rerank.csv

# HyDE + 重排
python scripts/eval.py --k 3 --hyde --rerank --export data/eval_results_hyde_rerank.csv
```


## 功能特性

### 核心功能
- **基础RAG**: 向量检索 + LLM生成，支持引用和证据质量评估
- **HyDE优化**: 假设答案生成，提升对比类问题的检索效果
- **混合检索**: 原始检索与HyDE检索结果融合，兼顾精度和召回
- **交叉编码器重排**: 提升检索结果的相关性排序
- **证据质量检查**: 自动评估证据充分性，支持降速策略
- **引用验证**: 验证答案中引用编号的有效性和覆盖率

### 技术特性
- **Token 级截断**: 使用 tiktoken 精确控制上下文长度（与 DeepSeek/GPT BPE 对齐）
- **降级策略**: 证据不足或 LLM 失败时，自动降级（启用 hybrid+rerank / 提高 Top-K / 拒答并附证据）
- **结构化输出**: 返回答案、引用、证据清单、证据质量、引用校验等
- **混合扎根判定**: 关键词优先 + 相似度兜底，提升可信度判定鲁棒性

## 技术栈

- **核心框架**: LlamaIndex
- **向量数据库**: FAISS
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2（默认），可替换为 BGE/BERT 家族
- **LLM**: DeepSeek Chat API（OpenAI 兼容）
- **重排模型**: Cross-Encoder（可选启用）
- **评估指标**: Hit@K、关键词覆盖率、引用覆盖率/有效性

> 网络受限环境：已在 `scripts/config.py` 中默认开启 `HF_HUB_OFFLINE=1`，并在 `core/retrieval.py` 中延长 `HF_HUB_DOWNLOAD_TIMEOUT`。

## 设计思路（What/Why/How）

- **HyDE（Why）**: 用户问题可能短/口语化，语义密度不足；让 LLM 先生成“假设答案”作为高密度查询，可显著增强召回。
- **混合检索（How）**: 原始检索 + HyDE 检索结果融合（去重+统一排序）。保留直观匹配，又引入语义扩展，鲁棒性更好。
- **证据质量评估（Why）**: 防止弱证据硬生成带来幻觉；先评估关键词覆盖/最佳命中/平均相似度。
- **降级策略（How）**: 证据不足时，依次尝试 hybrid+rerank、启用 rerank、提升 top_k；仍不足则拒答并附证据预览。
- **Token 级截断（How）**: 使用 tiktoken 与模型 BPE 对齐，避免中文被错误切断，确保 token 预算可控。
- **引用校验（How）**: 要求答案内内联 `[来源1]` 等编号，生成后验证范围与覆盖率，提升可追溯性。
- **混合扎根判定（Why/How）**: 先用关键词回查（快、可解释），不足再用 3-gram 相似度兜底，兼顾效率与鲁棒性。



## 开发阶段

### 阶段一：环境搭建与数据准备 ✅
- [x] 项目结构创建
- [x] 依赖安装配置
- [x] 医疗FAQ数据准备
- [x] 文档加载和分块
- [x] BGE-M3嵌入向量化
- [x] FAISS索引构建

### 阶段二：基础RAG与检索精度评估 ✅
- [x] 基础问答链实现
- [x] 检索精度评估函数
- [x] Hit@3指标计算
- [x] 测试集评估

### 阶段三：HyDE优化检索 ✅
- [x] HyDE查询扩展实现
- [x] 假设答案生成
- [x] 优化检索效果对比

### 阶段四：工业化思维与扩展（待实现）
- [ ] 模型切换机制
- [ ] 高级检索策略
- [ ] 生产环境监控
- [ ] 数据安全考虑
- [ ] 评估流程文档化（如何运行、解读结果、工业界标准）
- [ ] HyDE评估方法文档化
- [ ] 对比结果展示（运行固定对比，展示HyDE效果）
- [ ] RRF融合策略（更稳健的结果融合）
- [ ] BM25+向量检索器维度混合
- [ ] HyDE假设答案缓存优化

## 注意事项

1. **首次运行**：需要下载BGE-M3模型，可能需要几分钟时间
2. **API密钥**：确保配置正确的DeepSeek API密钥
3. **硬件要求**：建议至少4GB内存用于模型推理
4. **网络要求**：需要稳定的网络连接下载模型和调用API
