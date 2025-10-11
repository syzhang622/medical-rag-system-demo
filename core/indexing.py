# -*- coding: utf-8 -*-
"""
索引构建服务：核心索引构建逻辑

初学者理解：
- 这是RAG系统的索引构建核心服务
- 负责将文档转换为向量索引
- 可以被CLI、API或其他模块调用

功能：
1. 文档加载和预处理
2. 文本分块
3. 向量化
4. 索引构建和保存
"""

import os
import shutil
import logging
from typing import List

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
import faiss
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IndexingService:
    """索引构建服务类
    
    职责：
    - 文档加载和预处理
    - 文本分块
    - 向量化
    - 索引构建和保存
    
    设计思路：
    - 将复杂的索引构建流程封装成一个服务类
    - 支持配置驱动的处理流程
    - 提供完整的错误处理和日志记录
    """
    
    def __init__(self, config):
        """初始化索引构建服务
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.embed_model = None
        self.vector_store = None
        self.index = None
        
    def _get_index_dir(self) -> str:
        """根据当前配置生成索引目录，避免不同配置互相覆盖"""
        base_dir = self.config.FAISS_INDEX_PATH
        safe_model = str(self.config.EMBEDDING_MODEL).replace("/", "_")
        sub_dir = f"{safe_model}_cs{self.config.CHUNK_SIZE}_co{self.config.CHUNK_OVERLAP}"
        return os.path.join(base_dir, sub_dir)

    def load_documents(self) -> List:
        """加载医疗FAQ文档"""
        try:
            logger.info(f"正在加载文档: {self.config.MEDICAL_FAQ_PATH}")
            
            # 创建临时目录结构（LlamaIndex需要）
            if not os.path.exists("temp_docs"):
                os.makedirs("temp_docs")
            
            # 复制文件到临时目录
            shutil.copy(self.config.MEDICAL_FAQ_PATH, "temp_docs/")
            
            # 使用LlamaIndex的文档加载器
            reader = SimpleDirectoryReader("temp_docs")
            documents = reader.load_data()
            
            logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
    
    def setup_embedding_model(self):
        """设置嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.config.EMBEDDING_MODEL}")
            
            # 初始化嵌入模型
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL,
                device=self.config.DEVICE
            )
            logger.info("嵌入模型加载成功")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def create_vector_store(self):
        """创建FAISS向量存储"""
        try:
            logger.info("正在创建FAISS向量存储")
            
            # 动态探测嵌入维度
            probe_vector = self.embed_model.get_text_embedding("hello")
            dimension = len(probe_vector)
            logger.info(f"自动探测到嵌入维度: {dimension}")
            
            # 创建FAISS索引（使用内积相似度）
            faiss_index = faiss.IndexFlatIP(dimension)
            
            # 创建LlamaIndex的向量存储包装器
            self.vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            logger.info("FAISS向量存储创建成功")
            
        except Exception as e:
            logger.error(f"FAISS向量存储创建失败: {e}")
            raise
    
    def build_index(self, documents: List):
        """构建向量索引"""
        try:
            logger.info("正在构建向量索引")
            
            # 创建服务上下文
            Settings.llm = None
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separator="\n\n"
            )
            
            # 让系统自动管理文档存储，或者从文档构建
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
                # 不要手动创建空的docstore
            )
            
            # 构建索引
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                store_nodes_override=True  # ⬅️ 这行很重要！
            )
            
            logger.info("向量索引构建成功")
            
        except Exception as e:
            logger.error(f"向量索引构建失败: {e}")
            raise
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            target_dir = self._get_index_dir()
            logger.info(f"正在保存索引到: {target_dir}")

            # 规范化路径并确保目录存在
            persist_dir = os.path.abspath(target_dir)
            if os.path.isdir(persist_dir) and os.listdir(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                except Exception as _:
                    pass
            os.makedirs(persist_dir, exist_ok=True)
            
            # 保存完整的索引
            self.index.storage_context.persist(persist_dir=persist_dir)
            
            # 验证文件是否正确保存
            logger.info("验证保存的文件:")
            for file in os.listdir(persist_dir):
                file_path = os.path.join(persist_dir, file)
                size = os.path.getsize(file_path)
                logger.info(f"  {file}: {size} bytes")
            
            # 保存索引元数据
            index_metadata = {
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "similarity_top_k": self.config.SIMILARITY_TOP_K
            }
            
            with open(os.path.join(persist_dir, "metadata.pkl"), "wb") as f:
                pickle.dump(index_metadata, f)
            
            logger.info("索引保存成功")
            
        except Exception as e:
            logger.error(f"索引保存失败: {e}")
            raise
    
    def build_index_pipeline(self, rebuild_index: bool = True):
        """完整的索引构建流程"""
        try:
            logger.info("开始索引构建流程")
            
            # 1. 设置嵌入模型
            self.setup_embedding_model()
            
            if rebuild_index:
                # 2. 加载文档
                documents = self.load_documents()
                
                # 3. 创建向量存储
                self.create_vector_store()
                
                # 4. 构建索引
                self.build_index(documents)
                
                # 5. 保存索引
                self.save_index()
                
                logger.info("索引构建流程完成，索引已构建并保存")
            else:
                logger.info("索引构建流程完成，跳过重建")
            
            # 清理临时文件
            if os.path.exists("temp_docs"):
                shutil.rmtree("temp_docs")
            
            return True
            
        except Exception as e:
            logger.error(f"索引构建流程失败: {e}")
            # 清理临时文件
            if os.path.exists("temp_docs"):
                shutil.rmtree("temp_docs")
            raise
