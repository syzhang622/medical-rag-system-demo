"""
配置文件 - 医疗问答RAG系统
包含所有重要的配置参数和模型设置
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """系统配置类"""
    
    # 嵌入模型配置
    EMBEDDING_MODEL = "BAAI/bge-m3"  # 主要嵌入模型
    EMBEDDING_MODEL_BACKUP = "text-embedding-3-small"  # 备用嵌入模型（OpenAI）
    
    # 文本分块配置
    CHUNK_SIZE = 512  # 文本块大小
    CHUNK_OVERLAP = 50  # 文本块重叠大小
    
    # 检索配置
    SIMILARITY_TOP_K = 3  # 检索最相关的文档数量
    FAISS_INDEX_PATH = "faiss_index"  # FAISS索引保存路径
    
    # LLM配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL = "deepseek-chat"
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # 评估配置
    TEST_QUESTIONS = [
        {
            "question": "感冒和流感有什么区别？",
            "expected_keywords": ["症状", "严重程度", "发热", "并发症"]
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
    
    # 文件路径配置
    MEDICAL_FAQ_PATH = "medical_faq.txt"
    LOG_FILE = "rag_system.log"
    
    # 其他配置
    DEVICE = "cpu"  # 使用CPU进行推理，如果有GPU可改为"cuda"
    MAX_TOKENS = 1000  # LLM生成的最大token数
    TEMPERATURE = 0.7  # LLM生成温度
