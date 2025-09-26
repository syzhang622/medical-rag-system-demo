# -*- coding: utf-8 -*-
"""
阶段一：数据准备与向量化
实现文档加载、文本分块、向量化并存入FAISS索引
"""

import os
import logging
from typing import List, Dict
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    ServiceContext,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import pickle

from config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalRAGDataProcessor:
    """医疗RAG数据处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self.vector_store = None
        self.index = None
        
    def setup_embedding_model(self):
        """设置嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.config.EMBEDDING_MODEL}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.config.EMBEDDING_MODEL,
                device=self.config.DEVICE
            )
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise
    
    def load_documents(self) -> List:
        """加载医疗FAQ文档"""
        try:
            logger.info(f"正在加载文档: {self.config.MEDICAL_FAQ_PATH}")
            
            # 创建临时目录结构
            if not os.path.exists("temp_docs"):
                os.makedirs("temp_docs")
            
            # 复制文件到临时目录
            import shutil
            shutil.copy(self.config.MEDICAL_FAQ_PATH, "temp_docs/")
            
            # 使用SimpleDirectoryReader加载文档
            reader = SimpleDirectoryReader("temp_docs")
            documents = reader.load_data()
            
            logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
    
    def create_text_splitter(self) -> SentenceSplitter:
        """创建文本分割器"""
        return SentenceSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separator="\n\n"  # 按段落分割
        )
    
    def create_vector_store(self):
        """创建FAISS向量存储"""
        try:
            logger.info("正在创建FAISS向量存储")
            
            # 创建FAISS索引
            dimension = 1024  # BGE-M3的嵌入维度
            faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 创建向量存储
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
            service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model,
                node_parser=self.create_text_splitter()
            )
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 构建索引
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                storage_context=storage_context
            )
            
            logger.info("向量索引构建成功")
            
        except Exception as e:
            logger.error(f"向量索引构建失败: {e}")
            raise
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            logger.info(f"正在保存索引到: {self.config.FAISS_INDEX_PATH}")
            
            # 确保目录存在
            os.makedirs(self.config.FAISS_INDEX_PATH, exist_ok=True)
            
            # 保存FAISS索引
            self.vector_store.persist(persist_dir=self.config.FAISS_INDEX_PATH)
            
            # 保存索引元数据
            index_metadata = {
                "embedding_model": self.config.EMBEDDING_MODEL,
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "similarity_top_k": self.config.SIMILARITY_TOP_K
            }
            
            with open(os.path.join(self.config.FAISS_INDEX_PATH, "metadata.pkl"), "wb") as f:
                pickle.dump(index_metadata, f)
            
            logger.info("索引保存成功")
            
        except Exception as e:
            logger.error(f"索引保存失败: {e}")
            raise
    
    def load_index(self):
        """从磁盘加载索引"""
        try:
            logger.info(f"正在从 {self.config.FAISS_INDEX_PATH} 加载索引")
            
            # 检查索引文件是否存在
            if not os.path.exists(self.config.FAISS_INDEX_PATH):
                raise FileNotFoundError("索引文件不存在，请先运行数据准备流程")
            
            # 加载FAISS索引
            self.vector_store = FaissVectorStore.from_persist_dir(
                persist_dir=self.config.FAISS_INDEX_PATH
            )
            
            # 创建服务上下文
            service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model
            )
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # 重建索引
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                service_context=service_context,
                storage_context=storage_context
            )
            
            logger.info("索引加载成功")
            
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            raise
    
    def get_retriever(self):
        """获取检索器"""
        if self.index is None:
            raise ValueError("索引未初始化，请先构建或加载索引")
        
        return self.index.as_retriever(similarity_top_k=self.config.SIMILARITY_TOP_K)
    
    def process_data(self, rebuild_index: bool = True):
        """完整的数据处理流程"""
        try:
            logger.info("开始数据处理流程")
            
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
                
                logger.info("数据处理流程完成，索引已构建并保存")
            else:
                # 加载现有索引
                self.load_index()
                logger.info("数据处理流程完成，索引已加载")
            
            # 清理临时文件
            if os.path.exists("temp_docs"):
                import shutil
                shutil.rmtree("temp_docs")
            
            return True
            
        except Exception as e:
            logger.error(f"数据处理流程失败: {e}")
            raise

def main():
    """主函数"""
    try:
        # 创建配置
        config = Config()
        
        # 创建数据处理器
        processor = MedicalRAGDataProcessor(config)
        
        # 执行数据处理
        processor.process_data(rebuild_index=True)
        
        # 测试检索功能
        logger.info("测试检索功能...")
        retriever = processor.get_retriever()
        
        # 测试查询
        test_query = "感冒和流感有什么区别？"
        logger.info(f"测试查询: {test_query}")
        
        retrieved_nodes = retriever.retrieve(test_query)
        logger.info(f"检索到 {len(retrieved_nodes)} 个相关文档片段")
        
        for i, node in enumerate(retrieved_nodes):
            logger.info(f"片段 {i+1} (相似度: {node.score:.4f}):")
            logger.info(f"内容: {node.text[:200]}...")
            logger.info("-" * 50)
        
        logger.info("阶段一完成！数据已成功处理并建立索引。")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
