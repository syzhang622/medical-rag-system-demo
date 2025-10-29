# -*- coding: utf-8 -*-
"""重排器接口与默认实现。

初学者理解：
- 重排器是检索系统的"二次筛选"组件
- 先用向量相似度快速找到候选文档（粗召回）
- 再用重排器精确计算相关性，重新排序（精排）
- 这样既保证了速度，又提升了精度
"""

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseReranker:
    """重排器基类 - 定义重排接口
    
    重排器的作用：
    - 接收查询和候选文档列表
    - 计算更精确的相关性分数
    - 返回重新排序的结果
    """
    
    def rerank(self, query: str, nodes: List[Any]) -> Optional[List[Tuple[float, Any]]]:
        """重排接口
        
        Args:
            query: 用户查询
            nodes: 候选文档节点列表
            
        Returns:
            Optional[List[Tuple[float, Any]]]: 重排后的结果，格式为[(分数, 节点), ...]
        """
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """交叉编码器重排器
    
    工作原理：
    1. 将查询和每个文档组成(查询, 文档)对
    2. 使用预训练的交叉编码器计算相关性分数
    3. 按分数重新排序
    
    优势：
    - 比向量相似度更精确
    - 能理解查询和文档的深层语义关系
    
    劣势：
    - 计算速度较慢（需要逐个计算）
    - 需要额外的模型资源
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """初始化交叉编码器重排器
        
        Args:
            model_name: 交叉编码器模型名称
        """
        self.model_name = model_name
        self.ce = None
        
        # 尝试加载交叉编码器模型
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self.ce = CrossEncoder(model_name)
            logger.info(f"成功加载交叉编码器: {model_name}")
        except Exception as e:  # 如果模型不可用，优雅降级
            logger.warning(f"CrossEncoder 不可用，跳过重排: {e}")

    def rerank(self, query: str, nodes: List[Any]) -> Optional[List[Tuple[float, Any]]]:
        """执行交叉编码器重排
        
        Args:
            query: 用户查询
            nodes: 候选文档节点列表
            
        Returns:
            Optional[List[Tuple[float, Any]]]: 重排后的结果，按相关性分数降序排列
        """
        # 如果模型未加载，返回None（不进行重排）
        if self.ce is None:
            return None
            
        try:
            # 步骤1：构建查询-文档对
            # 将查询与每个文档组成对，用于交叉编码器计算
            pairs = [(query, n.text) for n in nodes]
            
            # 步骤2：批量计算相关性分数
            # 交叉编码器会同时考虑查询和文档，计算更精确的相关性
            scores = self.ce.predict(pairs)
            
            # 步骤3：按分数重新排序
            # 将分数和节点配对，按分数降序排列
            ranking = sorted(zip(scores, nodes), key=lambda x: float(x[0]), reverse=True)
            
            # 步骤4：返回格式化的结果
            return [(float(s), n) for s, n in ranking]
            
        except Exception as e:
            # 如果重排失败，记录警告但不影响主流程
            logger.warning(f"CrossEncoder 重排失败，跳过: {e}")
            return None


