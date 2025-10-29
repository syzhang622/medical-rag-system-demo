# -*- coding: utf-8 -*-
"""
主问答服务：协调各个组件完成端到端问答

设计要点：
- 作为协调器，依赖注入各个处理组件
- 实现证据质量检查与降速策略
- 处理 LLM 生成失败时的降级策略
"""

import os
import sys
import logging
from typing import List, Dict, Optional

# 添加项目根目录到Python路径（必须在导入其他模块之前）
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.config import Config
from core.retrieval import RetrievalService
from core.hyde import HyDERetriever
from components.llm_client import LLMClient, DEFAULT_SYSTEM_PROMPT
from components.text_processing import ContextFormatter, KeywordExtractor, CitationExtractor
from components.evidence_quality import EvidenceQualityChecker, WeakEvidenceHandler

logger = logging.getLogger(__name__)


class AnswerService:
    """问答服务：协调各个组件完成端到端问答"""

    def __init__(
        self, 
        cfg: Optional[Config] = None, 
        retrieval: Optional[RetrievalService] = None, 
        llm: Optional[LLMClient] = None, 
        hyde: Optional[HyDERetriever] = None,
        context_formatter: Optional[ContextFormatter] = None,
        keyword_extractor: Optional[KeywordExtractor] = None,
        citation_extractor: Optional[CitationExtractor] = None,
        quality_checker: Optional[EvidenceQualityChecker] = None,
        weak_evidence_handler: Optional[WeakEvidenceHandler] = None
    ) -> None:
        self.cfg = cfg or Config()
        self.retrieval = retrieval or RetrievalService(self.cfg)
        self.llm = llm or LLMClient(self.cfg)
        
        # 如果外部传入了 HyDERetriever 实例，则复用；否则创建新的
        if hyde is not None:
            self.hyde = hyde
        else:
            self.hyde = HyDERetriever(cfg=self.cfg, retrieval=self.retrieval, llm=self.llm)
        
        # 依赖注入各个处理组件
        self.context_formatter = context_formatter or ContextFormatter()
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.citation_extractor = citation_extractor or CitationExtractor()
        self.quality_checker = quality_checker or EvidenceQualityChecker(self.keyword_extractor)
        self.weak_evidence_handler = weak_evidence_handler or WeakEvidenceHandler(self.quality_checker)

    def answer(self, question: str, top_k: Optional[int] = None, enable_rerank: bool = False, mode: str = "base") -> Dict[str, object]:
        """执行端到端问答。

        mode:
            - "base": 直接使用原始问题进行向量检索
            - "hyde": 使用 HyDE 假设答案进行检索
            - "hybrid": 原始检索与 HyDE 检索融合
        发生异常时回退到纯检索结果。
        
        证据质量检查与降速策略：
        1. 检查关键词覆盖率、最佳排名、相似度分数
        2. 如果证据不足，尝试降速策略（hybrid模式、启用重排、提高top_k）
        3. 如果仍不足，返回拒答 + 证据清单
        """
        # 1) 检索（支持 HyDE / 混合）
        if mode == "hybrid":
            results = self.hyde.retrieve_hybrid(question=question, top_k=top_k, enable_rerank=enable_rerank)
        elif mode == "hyde":
            results = self.hyde.retrieve_with_hyde(question=question, top_k=top_k, enable_rerank=enable_rerank)
        else:
            results = self.retrieval.retrieve(question, top_k=top_k, enable_rerank=enable_rerank)
        texts = [r.text for r in results]
        sources = [r.source for r in results]
        k = int(top_k or self.cfg.SIMILARITY_TOP_K)

        # 如果没有检索到任何内容，直接返回提示
        if not texts:
            return {
                "answer": "未检索到相关内容，无法回答此问题。",
                "citations": "",
                "evidences": [],
                "mode": mode,
                "used_top_k": k,
                "rerank": bool(enable_rerank),
            }

        # 2) 检查证据质量，决定是否拒答或降速
        quality_info = self.quality_checker.check_quality(question, results, k)
        
        if quality_info["is_weak"]:
            # 尝试降速策略
            if mode == "base" and not enable_rerank:
                # 策略1：base -> hybrid + rerank
                logger.info(f"证据不足，尝试降速策略：base -> hybrid + rerank")
                hybrid_results = self.hyde.retrieve_hybrid(question=question, top_k=k, enable_rerank=True)
                hybrid_quality = self.quality_checker.check_quality(question, hybrid_results, k)
                if not hybrid_quality["is_weak"]:
                    # 降速成功，继续正常流程
                    results = hybrid_results
                    texts = [r.text for r in results]
                    sources = [r.source for r in results]
                    mode = "hybrid"
                    enable_rerank = True
                    logger.info("降速策略成功，使用 hybrid + rerank 模式")
                else:
                    # 降速失败，返回拒答
                    return self.weak_evidence_handler.handle_weak_evidence(question, results, mode, k, enable_rerank)
            elif not enable_rerank:
                # 策略2：启用重排
                logger.info(f"证据不足，尝试降速策略：启用重排")
                rerank_results = self.retrieval.retrieve(question, top_k=k, enable_rerank=True)
                rerank_quality = self.quality_checker.check_quality(question, rerank_results, k)
                if not rerank_quality["is_weak"]:
                    results = rerank_results
                    texts = [r.text for r in results]
                    sources = [r.source for r in results]
                    enable_rerank = True
                    logger.info("降速策略成功，启用重排")
                else:
                    return self.weak_evidence_handler.handle_weak_evidence(question, results, mode, k, enable_rerank)
            elif k < 10:  # 策略3：提高 top_k
                logger.info(f"证据不足，尝试降速策略：提高 top_k 从 {k} 到 {k*2}")
                higher_k_results = self.retrieval.retrieve(question, top_k=k*2, enable_rerank=enable_rerank)
                higher_k_quality = self.quality_checker.check_quality(question, higher_k_results, k*2)
                if not higher_k_quality["is_weak"]:
                    results = higher_k_results
                    texts = [r.text for r in results]
                    sources = [r.source for r in results]
                    k = k * 2
                    logger.info(f"降速策略成功，使用 top_k={k}")
                else:
                    return self.weak_evidence_handler.handle_weak_evidence(question, results, mode, k, enable_rerank)
            else:
                # 所有降速策略都失败，返回拒答
                return self.weak_evidence_handler.handle_weak_evidence(question, results, mode, k, enable_rerank)

        # 3) 组织上下文与提示词
        text_content = self.context_formatter.format_context(texts, sources)
        system_prompt = DEFAULT_SYSTEM_PROMPT
        user_prompt = self.context_formatter.build_user_prompt(question, text_content)

        # 4) 调用 LLM 生成（DeepSeek 优先，失败则回退 OpenAI；若都失败则降级）
        try:
            answer = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.cfg.MAX_TOKENS,
                temperature=self.cfg.TEMPERATURE,
            )
            citations = self.context_formatter.format_citations(sources)
            
            # 验证引用质量
            citation_validation = self.citation_extractor.validate_citations(answer, len(sources))
            
            evidences = [
                {
                    "text": r.text,
                    "source": r.source,
                    "sim_score": r.sim_score,
                    "rerank_score": r.rerank_score,
                    "rank": i + 1,
                }
                for i, r in enumerate(results)
            ]
            return {
                "answer": answer + citations,
                "citations": citations,
                "evidences": evidences,
                "mode": mode,
                "rerank": bool(enable_rerank),
                "evidence_quality": quality_info,
                "citation_validation": citation_validation,
            }
        except Exception as e:
            logger.warning(f"LLM 生成失败，降级返回检索片段: {e}")
            preview = "\n\n".join([(t or "").strip()[:200] for t in texts])
            citations = self.context_formatter.format_citations(sources)
            evidences = [
                {
                    "text": r.text,
                    "source": r.source,
                    "sim_score": r.sim_score,
                    "rerank_score": r.rerank_score,
                    "rank": i + 1,
                }
                for i, r in enumerate(results)
            ]
            return {
                "answer": f"根据提供的信息，无法调用生成模型，以下为相关片段摘要：\n\n{preview}" + citations,
                "citations": citations,
                "evidences": evidences,
                "mode": mode,
                "used_top_k": k,
                "rerank": bool(enable_rerank),
                "evidence_quality": quality_info,
                "fallback_reason": f"LLM生成失败: {str(e)}"
            }
