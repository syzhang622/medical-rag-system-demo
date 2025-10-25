# -*- coding: utf-8 -*-
"""
LLM 客户端封装：仅使用 DeepSeek（OpenAI 兼容 API）。

设计要点：
- 通过 OpenAI 官方 SDK 指向 DeepSeek base_url
- 不再使用 OpenAI 作为兜底；DeepSeek 不可用时由上层降级
- 可配置的提示词、温度、最大输出长度
"""

import logging
from typing import Optional

from scripts.config import Config

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


# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的医疗健康助手。请基于提供的医疗文档片段，用准确但通俗易懂的语言回答问题。\n\n"
    "要求：\n"
    "1) 严格基于提供的片段内容，不要编造。\n"
    "2) 语言准确专业，但表达通俗易懂，结构清晰。\n"
    "3) 如果片段中没有相关信息，请回答：根据提供的信息，无法回答此问题。\n"
    "4) 在回答的关键句子后面直接插入引用编号，如：感冒的症状包括发热、咳嗽[来源1]。\n"
    "5) 回答末尾用 [来源1] [来源2] ... 标识所有引用来源。\n"
    "6) 引用编号必须与提供的片段编号对应，不要编造引用。"
)
