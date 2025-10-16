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


阶段二

开启真LLM的方法
在项目根目录新建或编辑.env，加入：
DEEPSEEK_API_KEY=你的key
可选网络设置（需要代理时）：
HTTPS_PROXY / HTTP_PROXY
验证是否生效（任一其一）：
重新运行：python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --answer
或快速测试（会真实调API）：
在core/answering.py里，LLMClient会直接请求DeepSeek；若key有效，就不会降级为片段摘要
怎么做你要的三类对比（最简指令集）
多问几个问题（端到端RAG）：
python -m apps.cli.retrieve --q "如何预防感冒？" --k 3 --answer
python -m apps.cli.retrieve --q "流感的高危人群是谁？" --k 3 --answer
只调K（不重建索引）：
python -m apps.cli.retrieve --q "如何预防感冒？" --k 3 --answer
python -m apps.cli.retrieve --q "如何预防感冒？" --k 6 --answer
关注点：K↑覆盖更广但噪声↑；看答案是否更完整且不发散，引用是否更集中
开关重排（对同一K）：
无重排：python -m apps.cli.retrieve --q "如何预防感冒？" --k 6 --answer
有重排：python -m apps.cli.retrieve --q "如何预防感冒？" --k 6 --rerank --answer
关注点：是否更“对题”、更精炼；速度会慢一点
改分块策略（要重建索引）：
修改scripts/config.py的CHUNK_SIZE/CHUNK_OVERLAP
重建：python -m apps.cli.build_index
再问：python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --answer
关注点：块大上下文足但定位粗；块小定位准但易缺上下文；重叠↑跨块更连续
小结你该看什么
答案是否覆盖问题关键点（对照expected_keywords）
末尾引用是否合理、集中
参数变化后：答案是否更完整但不过度发散；性能是否可接受