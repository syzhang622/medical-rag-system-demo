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
import re
from collections import Counter

# 中文分词和关键词提取
try:
    import jieba
    import jieba.analyse
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

# 保证从项目根目录导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.config import Config
from core.retrieval import RetrievalService


logger = logging.getLogger(__name__)

# 轻量依赖：优先使用 tiktoken 做 token 级截断；不可用时退回字符截断
try:
    import tiktoken  # type: ignore
    _TK_ENC = tiktoken.get_encoding("cl100k_base")  # DeepSeek/GPT 兼容编码
except Exception:  # pragma: no cover - 可选依赖
    tiktoken = None
    _TK_ENC = None


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
    "4) 在回答的关键句子后面直接插入引用编号，如：感冒的症状包括发热、咳嗽[来源1]。\n"
    "5) 回答末尾用 [来源1] [来源2] ... 标识所有引用来源。\n"
    "6) 引用编号必须与提供的片段编号对应，不要编造引用。"
)


class AnswerService:
    """问答服务：把检索到的片段喂给 LLM，得到答案与引用。"""

    def __init__(self, cfg: Optional[Config] = None, retrieval: Optional[RetrievalService] = None, llm: Optional[LLMClient] = None, hyde: Optional['HyDERetriever'] = None) -> None:
        self.cfg = cfg or Config()
        self.retrieval = retrieval or RetrievalService(self.cfg)
        self.llm = llm or LLMClient(self.cfg)
        # 如果外部传入了 HyDERetriever 实例，则复用；否则创建新的
        if hyde is not None:
            self.hyde = hyde
        else:
            from core.hyde import HyDERetriever
            self.hyde = HyDERetriever(cfg=self.cfg, retrieval=self.retrieval, llm=self.llm)

    def _format_context(self, texts: List[str], sources: List[str], max_chars: int = 6000, max_tokens: int = 3500) -> str:
        """将若干片段整理为上下文（优先按 token 截断，退化为字符截断）。

        实现细节与原理（简明注释）：
        - 首选 tiktoken：与 DeepSeek/GPT 使用相同的 BPE 编码，token 计算精确；
          这样能确保“我们数到的 token”和模型实际消耗一致，避免中文被误切断。
        - 退回字符截断：当 tiktoken 不可用时，使用安全的字符截断，不阻塞主流程。

        tiktoken 与 transformers tokenizer 的区别（为什么选前者）：
        - tiktoken：专为 GPT 系列，轻量、快速、精确（无需加载模型参数）。
        - transformers：为特定模型（BERT/T5 等）设计，体积大、速度慢，且与 GPT 的编码不同，
          用它来估算 GPT/DeepSeek 的 token 容量会有偏差。
        """
        if _TK_ENC is not None:
            # 按 token 截断
            parts: List[str] = []
            total_tokens = 0
            for i, (t, s) in enumerate(zip(texts, sources), start=1):
                clean = (t or "").strip().replace("\n", " ")
                header = f"[片段{i}] 来源: {s}\n"
                body_tokens = _TK_ENC.encode(clean)
                header_tokens = _TK_ENC.encode(header)

                need = len(header_tokens) + len(body_tokens) + 1  # +1 for newline
                if total_tokens + need <= max_tokens:
                    parts.append(header + clean + "\n")
                    total_tokens += need
                    continue

                # 仅截断当前片段的正文，保留 header 信息
                remain = max(0, max_tokens - total_tokens - len(header_tokens) - 1)
                if remain > 0:
                    truncated = _TK_ENC.decode(body_tokens[:remain])
                    parts.append(header + truncated + "...\n")
                break  # 达到 token 上限

            return "\n".join(parts)

        # 回退：按字符截断（保留原有逻辑）
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
    
    def _extract_citations_from_answer(self, answer: str) -> List[int]:
        """从答案中提取引用编号。
        
        解析答案中的多种引用格式，返回引用的编号列表。
        支持格式：[来源1], [1], (来源1), [ref1]
        用于验证LLM是否正确使用了提供的片段。
        """
        if not answer:
            return []
        
        # 工业界标准：支持多种引用格式
        patterns = [
            r'\[来源(\d+)\]',  # [来源1] - 当前格式
            r'\[(\d+)\]',      # [1] - 简化格式
            r'\(来源(\d+)\)',  # (来源1) - 括号格式
            r'\[ref(\d+)\]',   # [ref1] - 英文格式
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            citations.extend([int(m) for m in matches])
        
        # 去重 并 排序
        return sorted(list(set(citations)))
    
    def _validate_citations(self, answer: str, num_sources: int) -> Dict[str, any]:
        """验证答案中的引用是否有效。
        
        检查：
        1. 引用的编号是否在有效范围内（1 到 num_sources）
        2. 是否引用了不存在的来源
        3. 引用覆盖率（引用了多少个提供的片段）
        """
        cited_numbers = self._extract_citations_from_answer(answer)
        valid_citations = [n for n in cited_numbers if 1 <= n <= num_sources]
        invalid_citations = [n for n in cited_numbers if n < 1 or n > num_sources]
        
        coverage = len(set(valid_citations)) / num_sources if num_sources > 0 else 0.0
        
        return {
            "cited_numbers": cited_numbers,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "coverage": coverage,
            "is_valid": len(invalid_citations) == 0
        }

    # ------------------------
    # 轻量“扎根判定”实现（混合：关键词优先 + 相似度补充）
    # ------------------------
    def _extract_keywords(self, text: str) -> List[str]:
        """工业级中文关键词提取：使用jieba分词 + TF-IDF。

        工业界标准做法：
        1. jieba分词：中文分词的标准库，准确率高
        2. TF-IDF：统计方法，计算词的重要性
        3. 停用词过滤：去除无意义的词
        4. 词性过滤：保留名词、动词、形容词等有意义词性
        
        降级策略：如果jieba不可用，回退到简单正则方法
        """
        if not text:
            return []
            
        if _JIEBA_AVAILABLE:
            # 工业级方法：jieba + TF-IDF
            # 使用TF-IDF提取关键词，topK=10，只保留长度>=2的词
            keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False)
            # 过滤长度和停用词
            filtered_keywords = []
            for kw in keywords:
                if len(kw) >= 2 and kw not in ['什么', '怎么', '如何', '为什么', '是否', '可以', '应该', '需要']:
                    filtered_keywords.append(kw)
            return filtered_keywords
        else:
            # 降级方法：简单正则（保持向后兼容）
            words = re.findall(r"[\u4e00-\u9fff]+", text)
            return [w for w in words if len(w) >= 2]

    def _calculate_similarity(self, a: str, b: str) -> float:
        """极简相似度：Jaccard on character 3-grams（无需模型，足够鲁棒）。"""
        def shingles(s: str) -> set:
            s = (s or "").strip()
            return {s[i:i+3] for i in range(max(0, len(s)-2))}
        A, B = shingles(a), shingles(b)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def _grounding_check_by_keywords(self, answer: str, evidences: List[Dict]) -> Dict[str, any]:
        """关键词回查：快速、可解释，适合事实/对比类问题。

        流程：
        1) 答案分句；2) 句子提取关键词；3) 在证据文本中统计命中；
        4) 以句为单位计算覆盖率；5) 统计整体扎根比例。
        """
        sentences = [s for s in re.split(r'[。！？；：\n]+', answer or "") if s.strip()]
        evidence_text = "\n".join([(e.get("text") or "") for e in evidences])
        grounded, ungrounded = [], []
        for s in sentences:
            kws = self._extract_keywords(s)
            if not kws:
                continue
            found = [kw for kw in kws if kw in evidence_text]
            cov = (len(found) / max(1, len(kws)))
            (grounded if cov >= 0.5 else ungrounded).append({
                "sentence": s,
                "coverage": round(cov, 3),
                "found_keywords": found,
            })
        ratio = len(grounded) / max(1, len(grounded) + len(ungrounded))
        return {"grounded": grounded, "ungrounded": ungrounded, "ratio": round(ratio, 3)}

    def _grounding_check_by_similarity(self, answer: str, evidences: List[Dict]) -> Dict[str, any]:
        """相似度回查：覆盖推理/抽象场景，但成本更高，阈值需要调参。

        流程：
        1) 分句；2) 计算每句与各证据的相似度；3) 取最大值与阈值比较。
        """
        sentences = [s for s in re.split(r'[。！？；：\n]+', answer or "") if s.strip()]
        grounded, ungrounded = [], []
        for s in sentences:
            sims = [(self._calculate_similarity(s, e.get("text") or ""), e) for e in evidences]
            best_sim, best_e = (max(sims, key=lambda x: x[0]) if sims else (0.0, None))
            (grounded if best_sim >= 0.3 else ungrounded).append({
                "sentence": s,
                "similarity": round(best_sim, 3),
                "evidence": best_e,
            })
        ratio = len(grounded) / max(1, len(grounded) + len(ungrounded))
        return {"grounded": grounded, "ungrounded": ungrounded, "ratio": round(ratio, 3)}

    def _grounding_check_hybrid(self, answer: str, evidences: List[Dict]) -> Dict[str, any]:
        """混合扎根判定（推荐）：关键词优先，相似度兜底。

        决策逻辑（why / how）：
        - 医疗问答多为事实/对比/列举，关键词回查简单、快速、可解释 → 作为首选；
        - 当关键词覆盖不足（ratio < 0.6）时，说明涉及推理/改写/同义表达 → 启用相似度回查；
        - 若相似度回查也不足（ratio < 0.5），则标记为“证据不足”，可触发拒答或降速。
        """
        kw = self._grounding_check_by_keywords(answer, evidences)
        if kw.get("ratio", 0.0) >= 0.6:
            return {"method": "keywords", **kw}
        sim = self._grounding_check_by_similarity(answer, evidences)
        passed = sim.get("ratio", 0.0) >= 0.5
        return {"method": "hybrid", "passed": passed, "keywords": kw, "similarity": sim}

    def _check_evidence_quality(self, question: str, results: List, top_k: int) -> Dict[str, any]:
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
        
        # 1. 提取问题关键词（工业级方法：jieba + TF​ ​TF（词频）-​IDF（逆文档频率））
        question_keywords = self._extract_keywords(question)
        
        if not question_keywords:
            return {"is_weak": False, "reason": "无法提取关键词", "coverage": 1.0, "best_rank": 1, "avg_sim_score": 0.0}
        
        # 2. 计算关键词覆盖率 是全文搜索
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

    def _handle_weak_evidence(self, question: str, results: List, mode: str, top_k: int, enable_rerank: bool) -> Dict[str, object]:
        """处理证据不足的情况：拒答 + 证据清单 + 可选降速重试。
        
        降速策略（按优先级）：
        1. 如果当前是 base 模式，尝试 hybrid 模式
        2. 如果未启用重排，尝试启用重排
        3. 如果 top_k 较小，尝试提高 top_k
        4. 如果都失败，返回拒答 + 证据清单
        """
        # 构建拒答文案
        quality_info = self._check_evidence_quality(question, results, top_k)
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
        quality_info = self._check_evidence_quality(question, results, k)
        
        if quality_info["is_weak"]:
            # 尝试降速策略
            if mode == "base" and not enable_rerank:
                # 策略1：base -> hybrid + rerank
                logger.info(f"证据不足，尝试降速策略：base -> hybrid + rerank")
                hybrid_results = self.hyde.retrieve_hybrid(question=question, top_k=k, enable_rerank=True)
                hybrid_quality = self._check_evidence_quality(question, hybrid_results, k)
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
                    return self._handle_weak_evidence(question, results, mode, k, enable_rerank)
            elif not enable_rerank:
                # 策略2：启用重排
                logger.info(f"证据不足，尝试降速策略：启用重排")
                rerank_results = self.retrieval.retrieve(question, top_k=k, enable_rerank=True)
                rerank_quality = self._check_evidence_quality(question, rerank_results, k)
                if not rerank_quality["is_weak"]:
                    results = rerank_results
                    texts = [r.text for r in results]
                    sources = [r.source for r in results]
                    enable_rerank = True
                    logger.info("降速策略成功，启用重排")
                else:
                    return self._handle_weak_evidence(question, results, mode, k, enable_rerank)
            elif k < 10:  # 策略3：提高 top_k
                logger.info(f"证据不足，尝试降速策略：提高 top_k 从 {k} 到 {k*2}")
                higher_k_results = self.retrieval.retrieve(question, top_k=k*2, enable_rerank=enable_rerank)
                higher_k_quality = self._check_evidence_quality(question, higher_k_results, k*2)
                if not higher_k_quality["is_weak"]:
                    results = higher_k_results
                    texts = [r.text for r in results]
                    sources = [r.source for r in results]
                    k = k * 2
                    logger.info(f"降速策略成功，使用 top_k={k}")
                else:
                    return self._handle_weak_evidence(question, results, mode, k, enable_rerank)
            else:
                # 所有降速策略都失败，返回拒答
                return self._handle_weak_evidence(question, results, mode, k, enable_rerank)

        # 3) 组织上下文与提示词
        text_content = self._format_context(texts, sources)
        system_prompt = DEFAULT_SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(question, text_content)

        # 4) 调用 LLM 生成（DeepSeek 优先，失败则回退 OpenAI；若都失败则降级）
        try:
            answer = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.cfg.MAX_TOKENS,
                temperature=self.cfg.TEMPERATURE,
            )
            citations = self._format_citations(sources)
            
            # 验证引用质量
            citation_validation = self._validate_citations(answer, len(sources))
            
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
                "used_top_k": k,
                "rerank": bool(enable_rerank),
                "evidence_quality": quality_info,
                "citation_validation": citation_validation,
            }
        except Exception as e:
            logger.warning(f"LLM 生成失败，降级返回检索片段: {e}")
            preview = "\n\n".join([(t or "").strip()[:200] for t in texts])
            citations = self._format_citations(sources)
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


