# -*- coding: utf-8 -*-
"""公共类型定义。"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CandidateResult:
    text: str
    source: str
    score: float
    rerank_score: Optional[float]
    metadata: Dict[str, Any]



