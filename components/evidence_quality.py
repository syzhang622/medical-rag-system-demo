# -*- coding: utf-8 -*-
"""
证据质量模块：证据质量检查、弱证据处理

包含功能：
- EvidenceQualityChecker: 证据质量检查
- WeakEvidenceHandler: 弱证据处理
"""

from typing import List, Dict, Optional
from components.text_processing import KeywordExtractor


class EvidenceQualityChecker:
    """证据质量检查器"""
    
    def __init__(self, keyword_extractor: Optional[KeywordExtractor] = None):
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
    
    def check_quality(self, question: str, results: List, top_k: int) -> Dict[str, any]:
        """检查证据质量，判断是否应该拒答或降速。
        
        判断标准（可配置阈值）：
        1. 关键词覆盖率 < 50%：从问题中提取关键词，检查检索片段覆盖情况
        2. citation_best_rank > top_k：第一个包含关键词的片段排名太靠后
        3. 相似度分数过低：sim_score 和 rerank_score 都低于阈值
        
        返回：
        - is_weak: bool - 是否证据不足
        - reason: str - 具体原因
        - coverage: float - 关键词覆盖率
        - best_rank: int - 最佳命中排名
        - avg_sim_score: float - 平均相似度分数
        """
        if not results:
            return {"is_weak": True, "reason": "无检索结果", "coverage": 0.0, "best_rank": 0, "avg_sim_score": 0.0}
        
        # 1. 提取问题关键词（工业级方法：jieba + TF-IDF）
        question_keywords = self.keyword_extractor.extract_keywords(question)
        
        if not question_keywords:
            return {"is_weak": False, "reason": "无法提取关键词", "coverage": 1.0, "best_rank": 1, "avg_sim_score": 0.0}
        
        # 2. 计算关键词覆盖率（全文搜索）
        all_text = "\n".join([r.text or "" for r in results])
        present_keywords = [kw for kw in question_keywords if kw in all_text]
        coverage = len(present_keywords) / len(question_keywords) if question_keywords else 0.0
        
        # 3. 找到第一个包含关键词的片段排名
        best_rank = 0
        for i, r in enumerate(results):
            if any(kw in (r.text or "") for kw in question_keywords):
                best_rank = i + 1
                break
        
        # 4. 计算平均相似度分数
        sim_scores = [r.sim_score for r in results if r.sim_score is not None]
        avg_sim_score = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
        
        # 5. 判断证据质量（可配置阈值）
        is_weak = False
        reasons = []
        
        if coverage < 0.5:  # 关键词覆盖率 < 50%
            is_weak = True
            reasons.append(f"关键词覆盖率低({coverage:.1%})")
        
        if best_rank > top_k or best_rank == 0:  # 最佳命中不在前K或未命中
            is_weak = True
            reasons.append(f"最佳证据排名靠后(rank={best_rank})")
        
        if avg_sim_score < 0.3:  # 平均相似度 < 0.3（可调整）
            is_weak = True
            reasons.append(f"相似度分数偏低({avg_sim_score:.3f})")
        
        return {
            "is_weak": is_weak,
            "reason": "; ".join(reasons) if reasons else "证据质量良好",
            "coverage": coverage,
            "best_rank": best_rank,
            "avg_sim_score": avg_sim_score
        }


class WeakEvidenceHandler:
    """弱证据处理器"""
    
    def __init__(self, quality_checker: Optional[EvidenceQualityChecker] = None):
        self.quality_checker = quality_checker or EvidenceQualityChecker()
    
    def handle_weak_evidence(self, question: str, results: List, mode: str, top_k: int, enable_rerank: bool) -> Dict[str, object]:
        """处理证据不足的情况：拒答 + 证据清单 + 可选降速重试。
        
        降速策略（按优先级）：
        1. 如果当前是 base 模式，尝试 hybrid 模式
        2. 如果未启用重排，尝试启用重排
        3. 如果 top_k 较小，尝试提高 top_k
        4. 如果都失败，返回拒答 + 证据清单
        """
        # 构建拒答文案
        quality_info = self.quality_checker.check_quality(question, results, top_k)
        refusal_text = f"根据提供的信息不足，无法给出可靠回答。\n\n"
        refusal_text += f"证据质量评估：{quality_info['reason']}\n"
        refusal_text += f"- 关键词覆盖率：{quality_info['coverage']:.1%}\n"
        refusal_text += f"- 最佳证据排名：{quality_info['best_rank']}\n"
        refusal_text += f"- 平均相似度：{quality_info['avg_sim_score']:.3f}\n\n"
        refusal_text += "以下为已检索到的相关片段，供参考：\n"
        
        # 构建证据清单（前3条）
        evidence_preview = []
        for i, r in enumerate(results[:3], 1):
            preview_text = (r.text or "").strip()[:150] + "..." if len(r.text or "") > 150 else (r.text or "")
            evidence_preview.append(f"[证据{i}] 来源: {r.source}\n相似度: {r.sim_score:.3f}\n内容: {preview_text}\n")
        
        refusal_text += "\n".join(evidence_preview)
        
        # 构建完整的 evidences 列表
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
            "answer": refusal_text,
            "citations": "",
            "evidences": evidences,
            "mode": mode,
            "used_top_k": top_k,
            "rerank": enable_rerank,
            "evidence_quality": quality_info,
            "fallback_reason": "证据不足，拒绝回答"
        }
