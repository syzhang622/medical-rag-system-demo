# -*- coding: utf-8 -*-
"""
RAG 问答服务：检索 + LLM 生成 + 引用。

设计要点（贴近工业化）：
- LLMClient 仅对接 DeepSeek（OpenAI 兼容协议），不再做 OpenAI 兜底；
  DeepSeek 不可用则直接降级为“检索摘要”。
- 明确的降级策略：LLM 失败时回退到纯检索结果
- 可配置的提示词、温度、最大输出长度
- 控制上下文长度，避免超过 LLM token 限制（此处用字符近似截断）
"""

from typing import List, Dict, Optional
import os
import sys
import logging

# 保证从项目根目录导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.config import Config
from core.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端封装：仅使用 DeepSeek（OpenAI 兼容 API）。

    - 通过 OpenAI 官方 SDK 指向 DeepSeek base_url
    - 不再使用 OpenAI 作为兜底；DeepSeek 不可用时由上层降级
    """

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self._ds_api_key = self.cfg.DEEPSEEK_API_KEY
        self._ds_base = self.cfg.DEEPSEEK_BASE_URL.rstrip('/')
        self._ds_model = self.cfg.DEEPSEEK_MODEL

        # 延迟导入，避免在无 openai 依赖场景报错
        self._openai_client = None
        if self._ds_api_key:
            try:
                from openai import OpenAI  # type: ignore
                base_url = self._ds_base
                # 兼容用户把 /v1 写进 base 的情况
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                logger.debug(f"DeepSeek OpenAI-client init with base_url={base_url}, model={self._ds_model}")
                self._openai_client = OpenAI(api_key=self._ds_api_key, base_url=base_url)
            except Exception as e:
                # 记录详细堆栈，便于调试
                logger.exception(f"DeepSeek 客户端创建失败: {e}")
                # 客户端创建失败时保持为 None，交由 generate 捕获并走降级
                self._openai_client = None
    def _generate_with_deepseek(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        if self._openai_client is None:
            raise RuntimeError("DeepSeek 客户端不可用或未正确配置")
        res = self._openai_client.chat.completions.create(
            model=self._ds_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (res.choices[0].message.content or "").strip()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        """仅调用 DeepSeek；失败让上层降级。

        要求环境变量提供 DEEPSEEK_API_KEY 与（可选）DEEPSEEK_BASE_URL、DEEPSEEK_MODEL。
        """
        if not self._ds_api_key:
            logger.error("未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek")
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        return self._generate_with_deepseek(system_prompt, user_prompt, max_tokens, temperature)


DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的医疗健康助手。请基于提供的医疗文档片段，用准确但通俗易懂的语言回答问题。\n\n"
    "要求：\n"
    "1) 严格基于提供的片段内容，不要编造。\n"
    "2) 语言准确专业，但表达通俗易懂，结构清晰。\n"
    "3) 如果片段中没有相关信息，请回答：根据提供的信息，无法回答此问题。\n"
    "4) 回答末尾用 [来源1] [来源2] ... 标识引用来源。"
)


class AnswerService:
    """问答服务：把检索到的片段喂给 LLM，得到答案与引用。"""

    def __init__(self, cfg: Optional[Config] = None, retrieval: Optional[RetrievalService] = None, llm: Optional[LLMClient] = None) -> None:
        self.cfg = cfg or Config()
        self.retrieval = retrieval or RetrievalService(self.cfg)
        self.llm = llm or LLMClient(self.cfg)

    def _format_context(self, texts: List[str], sources: List[str], max_chars: int = 6000) -> str:
        """将若干片段整理为上下文并进行长度截断（按字符近似）。"""
        parts: List[str] = []
        for i, (t, s) in enumerate(zip(texts, sources), start=1):
            clean = (t or "").strip().replace("\n", " ")
            parts.append(f"[片段{i}] 来源: {s}\n{clean}\n")
        joined = "\n".join(parts)
        if len(joined) <= max_chars:
            return joined
        # 简单截断：优先保留前若干高分片段
        return joined[:max_chars]

    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"问题：{question}\n\n"
            f"相关医疗文档片段：\n{context}\n\n"
            f"请基于以上片段回答问题："
        )

    def _format_citations(self, sources: List[str]) -> str:
        if not sources:
            return ""
        tags = [f"[来源{i}] {s}" for i, s in enumerate(sources, start=1)]
        return "\n" + "\n".join(tags)

    def answer(self, question: str, top_k: Optional[int] = None, enable_rerank: bool = False) -> Dict[str, str]:
        """执行端到端问答。发生异常时回退到纯检索结果。"""
        # 1) 检索
        results = self.retrieval.retrieve(question, top_k=top_k, enable_rerank=enable_rerank)
        texts = [r.text for r in results]
        sources = [r.source for r in results]

        # 如果没有检索到任何内容，直接返回提示
        if not texts:
            return {
                "answer": "未检索到相关内容，无法回答此问题。",
                "citations": "",
            }

        # 2) 组织上下文与提示词
        context = self._format_context(texts, sources)
        system_prompt = DEFAULT_SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(question, context)

        # 3) 调用 LLM 生成（DeepSeek 优先，失败则回退 OpenAI；若都失败则降级）
        try:
            answer = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.cfg.MAX_TOKENS,
                temperature=self.cfg.TEMPERATURE,
            )
            citations = self._format_citations(sources)
            return {
                "answer": answer + citations,
                "citations": citations,
            }
        except Exception as e:
            logger.warning(f"LLM 生成失败，降级返回检索片段: {e}")
            preview = "\n\n".join([(t or "").strip()[:200] for t in texts])
            citations = self._format_citations(sources)
            return {
                "answer": f"根据提供的信息，无法调用生成模型，以下为相关片段摘要：\n\n{preview}" + citations,
                "citations": citations,
            }


