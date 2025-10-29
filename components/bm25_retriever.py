# -*- coding: utf-8 -*-
"""BM25检索器 - 基于关键词的传统检索

BM25是什么：
- BM25 (Best Matching 25) 是一种基于TF-IDF的排序算法
- 主要用于关键词精确匹配，与向量检索互补
- 优势：擅长处理专有名词、精确术语、短文本匹配
- 劣势：无法理解语义，对同义词、改写不敏感

为什么要BM25 + 向量检索混合：
- 向量检索：擅长语义理解，但可能漏掉关键词精确匹配
- BM25检索：擅长关键词匹配，但不理解语义
- 两者结合：既有精确匹配，又有语义理解，鲁棒性最强
"""

import logging
from typing import List, Optional, Dict, Any
import jieba
from rank_bm25 import BM25Okapi
from components.types import CandidateResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25关键词检索器
    
    功能：
    1. 对文档集合构建BM25索引
    2. 基于关键词相似度检索最相关文档
    3. 返回标准化的CandidateResult格式
    """
    
    def __init__(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """初始化BM25检索器
        
        Args:
            documents: 文档列表（文本内容）
            metadata: 文档元数据列表（可选）
        """
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        
        # 中文分词：将每个文档分词
        logger.info(f"正在对 {len(documents)} 个文档进行分词...")
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        
        # 构建BM25索引
        logger.info("构建BM25索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("✅ BM25索引构建完成")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[CandidateResult]:
        """执行BM25检索
        
        Args:
            query: 查询问题
            top_k: 返回结果数量
            
        Returns:
            List[CandidateResult]: 检索结果，按BM25分数降序排列
        """
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取Top-K索引（按分数降序）
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 构建结果
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            
            # 格式化来源信息
            meta = self.metadata[idx]
            source = self._format_source(meta)
            
            results.append(CandidateResult(
                text=self.documents[idx],
                sim_score=score,  # BM25分数作为相似度分数
                source=source,
                rerank_score=None,
                metadata=meta
            ))
        
        return results
    
    def _format_source(self, meta: Dict[str, Any]) -> str:
        """格式化来源信息"""
        if not meta:
            return "bm25_unknown"
        
        # 按优先级查找来源字段
        for k in ["file_name", "file_path", "doc_id", "source", "segment_id"]:
            if k in meta and meta[k]:
                return str(meta[k])
        
        return "bm25_doc"


class HybridBM25VectorRetriever:
    """BM25 + 向量检索混合器
    
    功能：
    1. 同时执行BM25检索和向量检索
    2. 融合两种检索结果（去重 + 分数归一化）
    3. 返回最终的Top-K结果
    
    融合策略：
    - 分数归一化：将BM25分数和向量相似度分数归一化到[0,1]
    - 加权融合：alpha * vector_score + (1-alpha) * bm25_score
    - 去重：使用文本指纹去除重复文档
    - 排序：按融合后的分数降序排列
    """
    
    def __init__(
        self, 
        vector_retriever,  # RetrievalService实例
        bm25_retriever: BM25Retriever,
        alpha: float = 0.7
    ):
        """初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器（RetrievalService）
            bm25_retriever: BM25检索器
            alpha: 向量检索权重（0~1），BM25权重为(1-alpha)
                  - alpha=1.0: 完全使用向量检索
                  - alpha=0.5: 向量和BM25平权
                  - alpha=0.3: BM25权重更高
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        
        logger.info(f"混合检索器初始化完成 (向量权重={alpha}, BM25权重={1-alpha})")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        enable_rerank: bool = False
    ) -> List[CandidateResult]:
        """执行混合检索
        
        Args:
            query: 查询问题
            top_k: 最终返回结果数量
            enable_rerank: 是否启用重排（仅对向量检索生效）
            
        Returns:
            List[CandidateResult]: 融合后的检索结果
        """
        # 1. 向量检索
        vector_results = self.vector_retriever.retrieve(
            query=query, 
            top_k=top_k * 2,  # 多取一些，后面融合时截断
            enable_rerank=enable_rerank
        )
        
        # 2. BM25检索
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k * 2
        )
        
        # 3. 分数归一化
        vector_results = self._normalize_scores(vector_results, "vector")
        bm25_results = self._normalize_scores(bm25_results, "bm25")
        
        # 4. 融合结果
        merged = self._merge_and_rerank(vector_results, bm25_results)
        
        # 5. 截断到Top-K
        return merged[:top_k]
    
    def _normalize_scores(
        self, 
        results: List[CandidateResult], 
        source_type: str
    ) -> List[CandidateResult]:
        """归一化分数到[0,1]区间
        
        使用Min-Max归一化：score_norm = (score - min) / (max - min)
        """
        if not results:
            return results
        
        # 提取分数
        scores = [r.sim_score for r in results]
        
        # Min-Max归一化
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # 所有分数相同，归一化为1.0
            for r in results:
                r.sim_score = 1.0
        else:
            for r in results:
                r.sim_score = (r.sim_score - min_score) / (max_score - min_score)
        
        return results
    
    def _merge_and_rerank(
        self, 
        vector_results: List[CandidateResult],
        bm25_results: List[CandidateResult]
    ) -> List[CandidateResult]:
        """融合并重新排序
        
        策略：
        1. 使用文本指纹去重
        2. 对同一文档，计算加权融合分数
        3. 按融合分数降序排列
        """
        import hashlib
        
        # 使用字典按文本指纹去重
        merged_dict: Dict[str, CandidateResult] = {}
        
        # 处理向量检索结果
        for result in vector_results:
            fingerprint = self._get_fingerprint(result.text)
            
            if fingerprint not in merged_dict:
                # 新文档：加权分数
                result.sim_score = self.alpha * result.sim_score
                result.metadata = result.metadata or {}
                result.metadata['source_type'] = 'vector'
                merged_dict[fingerprint] = result
            else:
                # 重复文档：累加向量部分的分数
                merged_dict[fingerprint].sim_score += self.alpha * result.sim_score
        
        # 处理BM25检索结果
        for result in bm25_results:
            fingerprint = self._get_fingerprint(result.text)
            
            if fingerprint not in merged_dict:
                # 新文档：加权分数
                result.sim_score = (1 - self.alpha) * result.sim_score
                result.metadata = result.metadata or {}
                result.metadata['source_type'] = 'bm25'
                merged_dict[fingerprint] = result
            else:
                # 重复文档：累加BM25部分的分数
                merged_dict[fingerprint].sim_score += (1 - self.alpha) * result.sim_score
                # 标记为混合来源
                merged_dict[fingerprint].metadata = merged_dict[fingerprint].metadata or {}
                merged_dict[fingerprint].metadata['source_type'] = 'hybrid'
        
        # 按融合后的分数降序排列
        merged_list = list(merged_dict.values())
        merged_list.sort(key=lambda x: float(x.sim_score), reverse=True)
        
        return merged_list
    
    def _get_fingerprint(self, text: str) -> str:
        """计算文本指纹（用于去重）"""
        # 使用前256字符计算MD5
        return hashlib.md5(text[:256].encode('utf-8')).hexdigest()

