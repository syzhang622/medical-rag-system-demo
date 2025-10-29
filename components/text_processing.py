# -*- coding: utf-8 -*-
"""
文本处理模块：上下文格式化、关键词提取、引用处理

包含功能：
- ContextFormatter: 上下文格式化
- KeywordExtractor: 关键词提取
- CitationExtractor: 引用提取和验证
"""

import re
from typing import List, Dict, Optional

# 中文分词和关键词提取
try:
    import jieba
    import jieba.analyse
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

# 轻量依赖：优先使用 tiktoken 做 token 级截断；不可用时退回字符截断
try:
    import tiktoken  # type: ignore
    _TK_ENC = tiktoken.get_encoding("cl100k_base")  # DeepSeek/GPT 兼容编码
except Exception:  # pragma: no cover - 可选依赖
    tiktoken = None
    _TK_ENC = None


class ContextFormatter:
    """上下文格式化器"""
    
    def format_context(self, texts: List[str], sources: List[str], max_chars: int = 6000, max_tokens: int = 3500) -> str:
        """将若干片段整理为上下文（优先按 token 截断，退化为字符截断）。

        实现细节与原理（简明注释）：
        - 首选 tiktoken：与 DeepSeek/GPT 使用相同的 BPE 编码，token 计算精确；
          这样能确保"我们数到的 token"和模型实际消耗一致，避免中文被误切断。
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

    def build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return (
            f"问题：{question}\n\n"
            f"相关医疗文档片段：\n{context}\n\n"
            f"请基于以上片段回答问题："
        )

    def format_citations(self, sources: List[str]) -> str:
        """格式化引用信息"""
        if not sources:
            return ""
        tags = [f"[来源{i}] {s}" for i, s in enumerate(sources, start=1)]
        return "\n" + "\n".join(tags)


class KeywordExtractor:
    """关键词提取器"""
    
    def extract_keywords(self, text: str) -> List[str]:
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

    def calculate_similarity(self, a: str, b: str) -> float:
        """极简相似度：Jaccard on character 3-grams（无需模型，足够鲁棒）。"""
        def shingles(s: str) -> set:
            s = (s or "").strip()
            return {s[i:i+3] for i in range(max(0, len(s)-2))}
        A, B = shingles(a), shingles(b)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)


class CitationExtractor:
    """引用提取和验证器"""
    
    def extract_citations_from_answer(self, answer: str) -> List[int]:
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
        
        # 去重并排序
        return sorted(list(set(citations)))
    
    def validate_citations(self, answer: str, num_sources: int) -> Dict[str, any]:
        """验证答案中的引用是否有效。
        
        检查：
        1. 引用的编号是否在有效范围内（1 到 num_sources）
        2. 是否引用了不存在的来源
        3. 引用覆盖率（引用了多少个提供的片段）
        """
        cited_numbers = self.extract_citations_from_answer(answer)
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
