# -*- coding: utf-8 -*-
"""
HyDE（Hypothetical Document Embeddings）检索：

核心流程（为什么这么做）：
1) 用 LLM 先生成“假设答案”（更完整、更规范的表述）
   - 为什么：用户问题可能很短/口语化，直接检索召回弱；
     假设答案更像教科书句子，语义密度高，向量检索更容易命中要点。
2) 用“假设答案”作为查询进行向量检索
   - 怎么做：把假设答案当作新的 query 喂给检索器。
3) 可选：与原始问题的检索结果融合（混合检索）
   - 为什么：保留原始问题的直观匹配，又引入 HyDE 的语义扩展，通常更稳。

注意：
- 仅依赖现有的 LLMClient（DeepSeek）与 RetrievalService；
- 失败时降级：若假设答案生成失败，则自动回退为“原始检索”。
"""

from typing import List, Optional, Dict, Tuple
import hashlib

from scripts.config import Config
from core.retrieval import RetrievalService
from rag.types import CandidateResult


_HYDE_SYSTEM_PROMPT = (
    "你是医疗领域的专业助手。请根据用户问题生成一个专业且完整的‘假设答案’。\n"
    "要求：\n"
    "- 使用医学常用术语，条理清晰，包含要点；\n"
    "- 字数在200-400字之间；\n"
    "- 仅输出假设答案正文，不要解释你的行为。"
)


def _build_hyde_user_prompt(question: str) -> str:
    return (
        f"问题：{question}\n\n"
        f"请直接给出一个结构化、专业的假设答案，用于后续文档检索："
    )


def _fingerprint(text: str) -> str:
    # 用摘要（前256字）做 MD5 指纹：
    # - 为什么：混合检索时不同来源可能文本高度相似，指纹用于“去重合并”。
    # - 怎么做：稳定截断 + md5，既快速又足够区分常见重复。
    return hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()


class HyDERetriever:
    """提供 HyDE 检索与混合检索能力。"""

    def __init__(self, cfg: Optional[Config] = None, retrieval: Optional[RetrievalService] = None, llm=None) -> None:
        self.cfg = cfg or Config()
        self.retrieval = retrieval or RetrievalService(self.cfg)
        # 延迟导入LLMClient，避免循环导入
        if llm is None:
            from core.answering import LLMClient
            self.llm = LLMClient(self.cfg)
        else:
            self.llm = llm
        # 当需要作为统一检索接口被注入 AnswerService 时使用：
        # True 表示使用 hybrid（原始+HyDE 融合）模式；False 使用纯 HyDE。
        self.use_hybrid: bool = False

    # -----------------------------
    # 基础 HyDE：生成假设答案并用其检索
    # -----------------------------
    def generate_hypothetical_answer(self, question: str) -> str:
        user_prompt = _build_hyde_user_prompt(question)
        return self.llm.generate(
            system_prompt=_HYDE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self.cfg.MAX_TOKENS,
            # 关键点：适度降低温度，让“假设答案”更收敛、结构更稳定
            temperature=min(0.5, self.cfg.TEMPERATURE),
        )

    def retrieve_with_hyde(self, question: str, top_k: Optional[int] = None, enable_rerank: bool = False) -> List[CandidateResult]:
        try:
            hypo = self.generate_hypothetical_answer(question)
        except Exception:
            # 失败兜底：如果 LLM 生成失败，则回退为“原始问题”的常规检索
            return self.retrieval.retrieve(query=question, top_k=top_k, enable_rerank=enable_rerank)

        # 用“假设答案”替代原始问题做检索（HyDE 的核心）
        return self.retrieval.retrieve(query=hypo, top_k=top_k, enable_rerank=enable_rerank)

    # -----------------------------
    # 混合检索：原始检索 + HyDE检索 融合
    # -----------------------------
    def retrieve_hybrid(self, question: str, top_k: Optional[int] = None, enable_rerank: bool = False) -> List[CandidateResult]:
        k = top_k or self.cfg.SIMILARITY_TOP_K

        base_results = self.retrieval.retrieve(query=question, top_k=k, enable_rerank=enable_rerank)
        hyde_results = self.retrieve_with_hyde(question, top_k=k, enable_rerank=enable_rerank)

        # 融合策略：先合并，后按“重排分 + 相似度”统一排序，再截断 Top-K
        merged = self._merge_results(base_results, hyde_results)

        # 截断到 Top-K
        return merged[: int(k)]

    # 简单融合策略（为什么/怎么做）：
    # - 去重：以“文本摘要指纹”判重，避免同义/相似片段重复占位；
    # - 排序：若有重排分（cross-encoder 更准确），优先用其；否则退回相似度分。
    def _merge_results(self, base: List[CandidateResult], hyde: List[CandidateResult]) -> List[CandidateResult]:
        
        by_fp: Dict[str, CandidateResult] = {}

        def consider(item: CandidateResult) -> None:
            # 仅取片段前 256 字参与指纹，兼顾稳定性与性能
            fp = _fingerprint(item.text[:256])
            exists = by_fp.get(fp)
            if exists is None:
                by_fp[fp] = item
                return
            # 冲突解决：相同指纹保留“更强”的结果
            # - 指标含义：
            #   rerank_score -> 交叉编码器的重排分（更精确的相关性）
            #   sim_score    -> 语义相似度（向量检索分，CandidateResult.sim_score）
            # - 规则：先比较 rerank_score，再比较 sim_score
            def key(c: CandidateResult) -> Tuple[float, float]:
                return (float(c.rerank_score or 0.0), float(c.sim_score))
            if key(item) > key(exists):
                by_fp[fp] = item

        for it in base:
            consider(it)
        for it in hyde:
            consider(it)

        items = list(by_fp.values())
        # 统一排序键：先重排分，再相似度；从高到低
        items.sort(key=lambda c: (float(c.rerank_score or 0.0), float(c.sim_score)), reverse=True)
        return items

    # 统一检索接口（便于被 AnswerService 注入）：
    # - 默认走纯 HyDE；若 use_hybrid=True 则走融合模式。
    def retrieve(self, query: str, top_k: Optional[int] = None, enable_rerank: bool = False) -> List[CandidateResult]:
        if self.use_hybrid:
            return self.retrieve_hybrid(question=query, top_k=top_k, enable_rerank=enable_rerank)
        return self.retrieve_with_hyde(question=query, top_k=top_k, enable_rerank=enable_rerank)


