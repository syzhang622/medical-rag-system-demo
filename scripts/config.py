"""
配置文件 - 医疗问答RAG系统
包含所有重要的配置参数和模型设置

- 这个文件就像项目的"设置面板"，所有参数都在这里统一管理
- 修改这里的参数会影响整个系统的行为
"""

import os
from dotenv import load_dotenv

# 加载环境变量（从.env文件读取API密钥等敏感信息）
load_dotenv()

# 设置 Hugging Face 镜像（加速模型下载，国内访问更快）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Config:
    """系统配置类 - 所有配置参数的集中管理"""
    
    # ========== 嵌入模型配置 ==========
    # 嵌入模型：将文本转换为数字向量的AI模型
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 主要嵌入模型
    # 为什么选这个模型：
    # - 向量维度: 384维（相对较小，速度快）
    # - 体积小、速度快、效果稳定
    # - 是做相似度检索/RAG的高性价比选择
    EMBEDDING_MODEL_BACKUP = "text-embedding-3-small"  # 备用嵌入模型（OpenAI，需要API密钥）
    
    # ========== 文本分块配置 ==========
    # 文本分块：将长文档切成小块，便于处理和搜索
    CHUNK_SIZE = 384      # 每个文本块的最大字符数
    CHUNK_OVERLAP = 100    # 相邻文本块之间的重叠字符数
    # 为什么需要分块：
    # - 块太小：会丢失上下文信息，影响理解
    # - 块太大：搜索精度下降，找到的内容不够精准
    # - 重叠：确保重要信息不会因为分块边界而丢失
    
    # ========== 检索配置 ==========
    SIMILARITY_TOP_K = 3  # 检索时返回最相关的文档数量（Top-K）
    FAISS_INDEX_PATH = "data/faiss_index"  # FAISS向量索引的保存路径
    
    # ========== LLM配置 ==========
    # LLM：大语言模型，用于生成回答（后续阶段使用）
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 从环境变量读取API密钥
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek聊天模型
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API密钥（备用）
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # ========== 评估配置 ==========
    # 测试问题：用于评估系统效果的标准化问题集
    TEST_QUESTIONS = [
        {
            "question": "感冒和流感有什么区别？",
            "expected_keywords": ["症状", "严重程度", "发热", "并发症"]  # 期望包含的关键词
        },
        {
            "question": "如何预防感冒？",
            "expected_keywords": ["洗手", "疫苗", "营养", "运动"]
        },
        {
            "question": "流感的高危人群有哪些？",
            "expected_keywords": ["老年人", "儿童", "孕妇", "慢性病"]
        }
    ]
    
    # ========== 文件路径配置 ==========
    MEDICAL_FAQ_PATH = "data/medical_faq.txt"  # 医疗FAQ数据文件路径
    LOG_FILE = "logs/rag_system.log"  # 日志文件路径
    
    # ========== 其他配置 ==========
    DEVICE = "cpu"  # 计算设备：cpu（通用）或cuda（需要GPU）
    MAX_TOKENS = 1000  # LLM生成的最大token数（控制回答长度）
    TEMPERATURE = 0.7  # LLM生成温度（0-1，越高越随机，越低越确定）
