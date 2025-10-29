# -*- coding: utf-8 -*-
"""
检索服务：加载索引、Top-K 召回、可选重排。

初学者理解：
- 这是RAG系统的核心检索模块
- 负责将用户问题转换为向量，然后在索引中搜索最相关的文档
- 可以理解为"智能搜索引擎"的核心部分
"""

import os
import sys
import pickle
import logging
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径（必须在导入其他模块之前）
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# 禁用Hugging Face符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)

"""
4️⃣ llama_index.core 的导入触发了一系列初始化
llama_index.core 这个包在导入时会：
- 导入 llama_index.core.base.response.schema
- 导入 llama_index.core.schema
- 导入 llama_index.core.utils
- 导入 nltk（自然语言处理库）
- 导入 scipy.stats（科学计算库）

问题出在这里：
这些库在初始化时会：
- 检查某些全局配置
- 初始化某些单例对象
- 可能会基于进程启动时的环境做一些决策
"""

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from scripts.config import Config
from components.types import CandidateResult
from components.rerankers import BaseReranker, CrossEncoderReranker


logger = logging.getLogger(__name__)


def _get_index_dir(cfg: Config) -> str:
    """根据配置生成索引目录路径
    
    为什么需要这样命名：
    - 不同模型、不同分块参数会产生不同的索引
    - 通过目录名区分，避免索引混乱
    """
    safe_model = str(cfg.EMBEDDING_MODEL).replace("/", "_")  # 将模型名中的斜杠替换为下划线
    sub_dir = f"{safe_model}_cs{cfg.CHUNK_SIZE}_co{cfg.CHUNK_OVERLAP}"  # 包含模型名和分块参数
    return os.path.join(cfg.FAISS_INDEX_PATH, sub_dir)


def _load_metadata(persist_dir: str) -> Dict[str, Any]:
    """加载索引元数据
    
    元数据包含：模型名、分块参数、构建时间等信息
    用于验证索引是否与当前配置匹配
    """
    meta_path = os.path.join(persist_dir, "metadata.pkl")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "rb") as f:
        return pickle.load(f)  # 反序列化pickle文件


def _format_source(meta: Dict[str, Any]) -> str:
    """格式化文档来源信息
    
    从元数据中提取最相关的来源信息，用于显示结果来源
    """
    if not meta:
        return "unknown"
    # 按优先级查找来源字段
    for k in ["file_name", "file_path", "doc_id", "source", "segment_id"]:
        if k in meta and meta[k]:
            return str(meta[k])
    # 如果没找到标准字段，取前3个键值对
    items = list(meta.items())[:3]
    return ", ".join([f"{k}={v}" for k, v in items]) if items else "unknown"


# 兼容旧名：语义（FAISS）检索器
class RetrievalService:
    """检索服务类 - RAG系统的核心检索组件
    
    功能：
    1. 加载预构建的向量索引
    2. 将用户问题转换为向量
    3. 在索引中搜索最相关的文档
    4. 可选：使用重排器提升结果质量
    """
    
    def __init__(self, cfg: Optional[Config] = None, reranker: Optional[BaseReranker] = None) -> None:
        """初始化检索服务
        
        Args:
            cfg: 配置对象，如果为None则使用默认配置
            reranker: 重排器，用于提升检索结果质量
        """
        self.cfg = cfg or Config()
        self.embed: Optional[HuggingFaceEmbedding] = None  # 嵌入模型
        self.index: Optional[VectorStoreIndex] = None      # 向量索引
        self.metadata: Dict[str, Any] = {}                 # 索引元数据
        self.reranker = reranker                           # 重排器

    def load(self) -> None:
        """加载索引和嵌入模型
        
        这是检索前的必要步骤，需要：
        1. 验证索引文件存在
        2. 加载嵌入模型（与构建时相同）
        3. 重建向量索引对象
        """
        # 设置Hugging Face镜像（加速模型下载）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 设置离线模式，优先使用本地缓存
        os.environ["HF_HUB_OFFLINE"] = "1"
        # 设置更长的超时时间
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        
        # 获取索引目录并验证文件存在
        persist_dir = _get_index_dir(self.cfg)
        assert os.path.isdir(persist_dir), f"索引目录不存在: {persist_dir}"
        assert os.path.exists(os.path.join(persist_dir, "docstore.json")), "缺少 docstore.json"

        # 加载索引元数据
        self.metadata = _load_metadata(persist_dir)
        logger.info(f"索引元数据: {self.metadata}")

        # 初始化嵌入模型（必须与构建时使用相同模型）
        self.embed = HuggingFaceEmbedding(
            model_name=self.cfg.EMBEDDING_MODEL,
            device=self.cfg.DEVICE,
        )

        # 配置LlamaIndex全局设置
        Settings.llm = None  # 检索阶段不需要LLM
        Settings.embed_model = self.embed  # 设置嵌入模型

        # 使用简化的方法，直接重建索引
        try:
            logger.info("使用正确的API加载索引...")
            
            # 使用 FaissVectorStore.from_persist_dir() 加载向量存储
            self.vector_store = FaissVectorStore.from_persist_dir(persist_dir)
            logger.info("FAISS向量存储加载成功")
            
            # 创建存储上下文
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=persist_dir  # 关键：指定持久化目录
            )
            logger.info("存储上下文创建成功")#storage_context.docstore 存不存文本（可查 len(docstore.docs)）
            
            # 使用 load_index_from_storage 而不是 from_vector_store
            self.index = load_index_from_storage(storage_context) 
            logger.info("索引重建成功")
            
            #StorageContext.from_defaults()创建的是新上下文，不是从持久化文件加载的，虽然传入了 vector_store，但​​文档存储(docstore)是空的
            #from_vector_store()需要访问文档存储来获取文本内容，from_vector_store()无法访问到已存储的文本内容
            #这就是为什么必须使用 load_index_from_storage()的原因 - 它专门设计用于从完整的持久化数据加载索引。
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            raise e

    def retrieve(self, query: str, top_k: Optional[int] = None, enable_rerank: bool = False) -> List[CandidateResult]:
        """执行检索查询
        
        Args:
            query: 用户问题
            top_k: 返回结果数量，默认使用配置中的值
            enable_rerank: 是否启用重排
            
        Returns:
            List[CandidateResult]: 检索结果列表，按相关性排序
        """
        # 验证索引已加载
        assert self.index is not None, "索引未加载，先调用 load()"
        
        # 确定返回结果数量
        k = int(top_k or self.cfg.SIMILARITY_TOP_K)
        
        # 核心检索步骤：将问题转换为向量，在索引中搜索最相似的文档
        nodes = self.index.as_retriever(similarity_top_k=k).retrieve(query)
        if not nodes:
            return []

        # 可选重排：使用交叉编码器提升结果质量
        if enable_rerank:
            reranker = self.reranker or CrossEncoderReranker()
            ranked = reranker.rerank(query, nodes)
        else:
            ranked = None

        # 格式化结果
        results: List[CandidateResult] = []
        if ranked is not None:
            # 使用重排后的结果
            for s, n in ranked:
                meta = getattr(n, "metadata", {}) or {}
                results.append(
                    CandidateResult(
                        text=(n.text or ""),
                        source=_format_source(meta),
                        sim_score=float(getattr(n, "score", 0.0)),  # 语义相似度分数
                        rerank_score=float(s),                      # 重排分数
                        metadata=meta,
                    )
                )
        else:
            # 使用原始检索结果
            for n in nodes:
                meta = getattr(n, "metadata", {}) or {}
                results.append(
                    CandidateResult(
                        text=(n.text or ""),
                        source=_format_source(meta),
                        sim_score=float(getattr(n, "score", 0.0)),  # 语义相似度分数
                        rerank_score=None,                          # 无重排分数
                        metadata=meta,
                    )
                )
        return results
