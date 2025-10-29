# -*- coding: utf-8 -*-
"""公共类型定义。"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CandidateResult:
    text: str
    source: str
    sim_score: float  # 语义相似度分数
    rerank_score: Optional[float]  # 重排分数（可选）
    metadata: Dict[str, Any]



