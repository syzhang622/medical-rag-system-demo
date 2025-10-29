# -*- coding: utf-8 -*-
"""
通用组件模块

包含可复用的组件：
- types: 类型定义
- rerankers: 重排器
- llm_client: LLM 客户端
- text_processing: 文本处理
- evidence_quality: 证据质量检查
"""

from .types import CandidateResult
from .rerankers import BaseReranker, CrossEncoderReranker
from .llm_client import LLMClient, DEFAULT_SYSTEM_PROMPT
from .text_processing import ContextFormatter, KeywordExtractor, CitationExtractor
from .evidence_quality import EvidenceQualityChecker, WeakEvidenceHandler

__all__ = [
    'CandidateResult',
    'BaseReranker',
    'CrossEncoderReranker', 
    'LLMClient',
    'DEFAULT_SYSTEM_PROMPT',
    'ContextFormatter',
    'KeywordExtractor',
    'CitationExtractor',
    'EvidenceQualityChecker',
    'WeakEvidenceHandler'
]

