# 医疗问答RAG系统

> 🎯 **我的RAG学习记录项目** - 从零开始学习检索增强生成技术，适合初学者参考

这是我基于LlamaIndex和FAISS开发的医疗问答检索增强生成(RAG)系统，支持HyDE查询扩展和检索精度评估。通过这个项目，我学习了RAG系统的完整实现流程。


### ✅ 已完成的核心功能

#### 1. **基础RAG架构**
- **文档处理**: 支持医疗FAQ文档的加载、分块和向量化
- **向量检索**: 基于FAISS的语义向量检索
- **LLM集成**: 集成DeepSeek API进行答案生成
- **引用系统**: 完整的引用验证和格式化

#### 2. **高级检索优化**
- **HyDE技术**: 假设答案生成，提升对比类问题的检索效果
- **混合检索**: 原始检索与HyDE检索结果融合
- **交叉编码器重排**: 使用CrossEncoder提升检索精度
- **证据质量检查**: 自动评估证据充分性，支持降级策略

#### 3. **工业级特性**
- **降级策略**: 证据不足时自动尝试不同检索模式
- **Token级截断**: 使用tiktoken精确控制上下文长度
- **错误处理**: 完善的异常处理和降级机制
- **配置管理**: 统一的配置系统，支持环境变量

#### 4. **评估体系**
- **Hit@K指标**: 评估检索精度
- **关键词覆盖率**: 评估内容相关性
- **引用验证**: 验证答案引用的有效性
- **批量评估**: 支持多问题批量测试

### 📈 当前运行状态

#### 数据准备 ✅
- 医疗FAQ数据已准备完成（`data/medical_faq.txt`）
- 已构建两个不同参数的FAISS索引：
  - `cs384_co100`: 分块大小384，重叠100
  - `cs512_co50`: 分块大小512，重叠50

#### 评估结果 ✅
从完整评估报告（`docs/EVALUATION_REPORT.md` 和 `data/eval_summary.txt`）可以看到：
- **所有模式Hit@3**: 100% (3/3问题全部命中)
- **关键词覆盖率**: 平均83.3%
- **重要发现**: Base、HyDE、Hybrid三种模式在当前数据集上表现一致，说明系统稳定可靠
- 测试问题覆盖：感冒vs流感区别、预防感冒、流感高危人群

### 🚀 项目优势

1. **架构设计优秀**: 模块化设计，职责分离清晰
2. **功能完整**: 从基础检索到高级优化一应俱全
3. **工业级特性**: 完善的错误处理、降级策略、配置管理
4. **评估体系**: 科学的评估指标和测试流程
5. **文档完善**: 代码注释详细，README文档完整

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

### 阶段四：工业化思维与扩展（进行中）
- [x] 高级检索策略（HyDE、混合检索、重排、证据质量检查、降级策略）
- [x] 评估流程文档化（一键评估脚本 + 评估报告模板）
- [x] HyDE评估方法文档化（详见 docs/EVALUATION_REPORT.md）
- [ ] 对比结果展示（运行完整评估，填充评估报告）
- [ ] HyDE假设答案缓存优化（计划实现）
- [ ] 模型切换机制（支持动态切换不同嵌入模型）
- [ ] 生产环境监控（结构化日志、性能指标）
- [ ] 数据安全考虑（敏感信息过滤、访问控制）
- [ ] RRF融合策略（考虑中）
- [ ] BM25+向量检索混合（计划实现）

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
├── components/                 # 组件模块
│   ├── llm_client.py          # LLM客户端封装
│   ├── evidence_quality.py    # 证据质量检查
│   ├── rerankers.py           # 重排器实现
│   ├── text_processing.py     # 文本处理工具
│   └── types.py                # 通用结构体（CandidateResult 等）
├── scripts/
│   ├── config.py               # 统一配置（HF 离线/超时、路径等）
│   └── eval.py                 # 命令行评估（Hit@K/关键词覆盖率）
├── data/                       # 数据与索引输出
│   ├── faiss_index/            # FAISS 持久化目录
│   ├── medical_faq.txt         # 医疗FAQ数据
│   ├── eval_results_*.csv      # 各模式评估结果（base/hyde/hybrid）
│   └── eval_summary.txt        # 评估结果汇总
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

#### 🚀 一键运行完整评估（推荐）
```bash
# 自动运行所有模式的评估并生成对比报告
python -m scripts.run_eval_all
```

这个脚本会自动：
- 运行 Base、HyDE、Hybrid 三种模式的评估
- 可选：运行带重排的评估
- 生成汇总对比表格
- 保存详细结果到 CSV 文件

#### 手动运行单个评估
```bash
# 基线检索评估
python -m scripts.eval --k 3 --export data/eval_results_base.csv

# HyDE检索评估
python -m scripts.eval --k 3 --hyde --export data/eval_results_hyde.csv

# 混合检索评估
python -m scripts.eval --k 3 --hybrid --export data/eval_results_hybrid.csv
```

#### 带重排的评估
```bash
# 基线 + 重排
python -m scripts.eval --k 3 --rerank --export data/eval_results_base_rerank.csv

# HyDE + 重排
python -m scripts.eval --k 3 --hyde --rerank --export data/eval_results_hyde_rerank.csv

# Hybrid + 重排
python -m scripts.eval --k 3 --hybrid --rerank --export data/eval_results_hybrid_rerank.csv
```

#### 📊 查看评估报告
评估完成后，查看详细分析报告：[docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)


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

- **核心框架**: LlamaIndex 0.10.12
- **向量数据库**: FAISS
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2（默认），可替换为 BGE/BERT 家族
- **LLM**: DeepSeek Chat API（OpenAI 兼容）
- **重排模型**: Cross-Encoder（可选启用）
- **评估指标**: Hit@K、关键词覆盖率、引用覆盖率/有效性
- **文本处理**: jieba分词、tiktoken tokenization
- **数据处理**: pandas、numpy、scikit-learn

> 网络受限环境：已在 `scripts/config.py` 中默认开启 `HF_HUB_OFFLINE=1`，并在 `core/retrieval.py` 中延长 `HF_HUB_DOWNLOAD_TIMEOUT`。

## 设计思路（What/Why/How）

- **HyDE（Why）**: 用户问题可能短/口语化，语义密度不足；让 LLM 先生成“假设答案”作为高密度查询，可显著增强召回。
- **混合检索（How）**: 原始检索 + HyDE 检索结果融合（去重+统一排序）。保留直观匹配，又引入语义扩展，鲁棒性更好。
- **证据质量评估（Why）**: 防止弱证据硬生成带来幻觉；先评估关键词覆盖/最佳命中/平均相似度。
- **降级策略（How）**: 证据不足时，依次尝试 hybrid+rerank、启用 rerank、提升 top_k；仍不足则拒答并附证据预览。
- **Token 级截断（How）**: 使用 tiktoken 与模型 BPE 对齐，避免中文被错误切断，确保 token 预算可控。
- **引用校验（How）**: 要求答案内内联 `[来源1]` 等编号，生成后验证范围与覆盖率，提升可追溯性。
- **混合扎根判定（Why/How）**: 先用关键词回查（快、可解释），不足再用 3-gram 相似度兜底，兼顾效率与鲁棒性。





### 🎯 优先改进方向

1. **扩展数据**: 增加更多医疗文档和测试问题
   - 添加更多医疗领域的FAQ数据
   - 扩展测试问题集，覆盖更多医疗场景
   - 考虑添加多语言支持

2. **模型优化**: 尝试更大的嵌入模型（如BGE系列）
   - 测试BGE-M3、BGE-large等更大模型
   - 对比不同嵌入模型的检索效果
   - 优化模型加载和缓存策略

3. **性能调优**: 基于评估结果优化参数配置
   - 调整分块大小和重叠参数
   - 优化Top-K值和重排策略
   - 测试不同温度参数对HyDE的影响

4. **生产部署**: 添加Docker、API服务等部署相关功能
   - 容器化部署配置
   - REST API接口封装
   - 生产环境监控和日志

5. **评估扩展**: 建立更全面的评估体系
   - 添加更多评估指标（如BLEU、ROUGE）
   - 建立人工评估流程
   - 添加A/B测试框架

## 注意事项

1. **首次运行**：需要下载sentence-transformers/all-MiniLM-L6-v2模型，可能需要几分钟时间
2. **API密钥**：确保配置正确的DeepSeek API密钥
3. **硬件要求**：建议至少4GB内存用于模型推理
4. **网络要求**：需要稳定的网络连接下载模型和调用API
5. **索引构建**：首次使用需要先运行`python -m apps.cli.build_index`构建索引
6. **评估结果**：当前系统在3个测试问题上达到100% Hit@3和83.3%关键词覆盖率
