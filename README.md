# 医疗问答RAG系统

基于LlamaIndex和FAISS的医疗问答检索增强生成(RAG)系统，支持HyDE查询扩展和检索精度评估。

## 项目结构

```
medical-rag-system/
├── requirements.txt              # 项目依赖
├── config.py                    # 配置文件
├── medical_faq.txt              # 医疗FAQ数据
├── env_example.txt              # 环境变量示例
├── stage1_data_preparation.py   # 阶段一：数据准备
├── stage2_basic_rag.py          # 阶段二：基础RAG（待实现）
├── stage3_hyde_optimization.py  # 阶段三：HyDE优化（待实现）
├── stage4_industrial_guide.py   # 阶段四：工业化指导（待实现）
└── README.md                    # 项目说明
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

### 3. 运行阶段一：数据准备

```bash
python stage1_data_preparation.py
```

这将：
- 加载医疗FAQ文档
- 使用BGE-M3模型生成嵌入向量
- 创建FAISS向量索引
- 保存索引到磁盘

## 技术栈

- **核心框架**: LlamaIndex
- **向量数据库**: FAISS
- **嵌入模型**: BAAI/bge-m3 (主要), OpenAI text-embedding-3-small (备用)
- **LLM**: DeepSeek Chat API
- **评估指标**: Hit@3

## 开发阶段

### 阶段一：环境搭建与数据准备 ✅
- [x] 项目结构创建
- [x] 依赖安装配置
- [x] 医疗FAQ数据准备
- [x] 文档加载和分块
- [x] BGE-M3嵌入向量化
- [x] FAISS索引构建

### 阶段二：基础RAG与检索精度评估（进行中）
- [ ] 基础问答链实现
- [ ] 检索精度评估函数
- [ ] Hit@3指标计算
- [ ] 测试集评估

### 阶段三：HyDE优化检索（待实现）
- [ ] HyDE查询扩展实现
- [ ] 假设答案生成
- [ ] 优化检索效果对比

### 阶段四：工业化思维与扩展（待实现）
- [ ] 模型切换机制
- [ ] 高级检索策略
- [ ] 生产环境监控
- [ ] 数据安全考虑

## 注意事项

1. **首次运行**：需要下载BGE-M3模型，可能需要几分钟时间
2. **API密钥**：确保配置正确的DeepSeek API密钥
3. **硬件要求**：建议至少4GB内存用于模型推理
4. **网络要求**：需要稳定的网络连接下载模型和调用API

## 下一步

运行阶段一后，我们将继续实现阶段二的基础RAG功能。请确保：
1. 成功运行了数据准备脚本
2. 配置了正确的API密钥
3. 理解了项目的基本结构

准备好后，我们可以继续阶段二的开发！
