# -*- coding: utf-8 -*-
"""
RAG 问答服务模块

模块化设计：
- llm_client: LLM 客户端
- text_processing: 文本处理（上下文格式化、关键词提取、引用处理）
- evidence_quality: 证据质量（检查、弱证据处理）
- answer_service: 主问答服务（协调器）
"""

from .llm_client import LLMClient
from .text_processing import ContextFormatter, KeywordExtractor, CitationExtractor
from .evidence_quality import EvidenceQualityChecker, WeakEvidenceHandler
from .answer_service import AnswerService

__all__ = [
    'LLMClient',
    'ContextFormatter', 
    'KeywordExtractor',
    'CitationExtractor',
    'EvidenceQualityChecker',
    'WeakEvidenceHandler',
    'AnswerService'
]
