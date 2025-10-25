# -*- coding: utf-8 -*-
"""检索 CLI：统一入口。

初学者理解：
- 这是RAG系统的命令行查询接口
- 用户可以通过命令行快速测试检索效果
- 支持自定义查询、结果数量、是否重排等参数

用法：
python -m apps.cli.retrieve --q "感冒和流感有什么区别？" --k 3 --rerank

参数说明：
--q, --query: 要查询的问题
--k: 返回结果数量（默认使用配置文件中的值）
--rerank: 是否启用交叉编码器重排（可选）
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.config import Config
from core.retrieval import RetrievalService  # 语义检索（旧名保留）
from core.hyde import HyDERetriever


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器
    
    定义CLI支持的所有参数和选项
    """
    p = argparse.ArgumentParser(description="RAG 检索/问答 CLI")
    
    # 必需参数：查询问题
    p.add_argument("--q", "--query", dest="query", type=str, required=True, 
                   help="要查询的问题（必需）")
    
    # 可选参数：返回结果数量
    p.add_argument("--k", dest="top_k", type=int, default=None, 
                   help="返回结果数量（默认使用配置文件中的SIMILARITY_TOP_K）")
    
    # 可选参数：是否启用重排
    p.add_argument("--rerank", action="store_true", 
                   help="是否启用交叉编码器重排（提升结果质量但速度较慢）")
    
    # 可选参数：是否直接生成答案（调用LLM）
    p.add_argument("--answer", action="store_true",
                   help="是否基于检索结果生成最终答案（调用LLM，含引用）")
    
    # HyDE/混合检索开关
    p.add_argument("--hyde", action="store_true",
                   help="使用HyDE：先生成假设答案再检索（提高对比类问题召回）")
    p.add_argument("--hybrid", action="store_true",
                   help="混合检索：原始检索 + HyDE 检索的融合结果")
    
    return p


def _decide_mode(args) -> str:
    """根据参数决定检索模式（供 --answer 使用）。"""
    if args.hybrid:
        return "hybrid"
    if args.hyde:
        return "hyde"
    return "base"


def main() -> None:
    """主函数：解析参数、执行检索、显示结果
    
    执行流程：
    1. 解析命令行参数
    2. 创建检索服务实例
    3. 加载索引和模型
    4. 执行检索查询
    5. 格式化并显示结果
    """
    # 步骤1：解析命令行参数
    args = build_argparser().parse_args()
    
    # 步骤2：加载底层向量检索器
    svc = RetrievalService()
    print("正在加载索引和模型...")
    svc.load()
    
    # 步骤3：创建 HyDERetriever（仅在需要时创建，避免重复实例化）
    hyde = None
    if args.hyde or args.hybrid or args.answer:
        hyde = HyDERetriever(retrieval=svc)
    
    # 步骤4：执行检索或问答
    print(f"正在检索: {args.query}")
    if args.answer:
        # 端到端问答：复用已创建的 HyDERetriever 实例
        from core.answering import AnswerService
        ans_svc = AnswerService(retrieval=svc, hyde=hyde)
        mode = _decide_mode(args)
        print("\n=== 生成答案（LLM） ===")
        out = ans_svc.answer(args.query, top_k=args.top_k, enable_rerank=args.rerank, mode=mode)
        print(out["answer"])  # 纯文本输出，末尾包含引用
    else:
        # 非 --answer：仅展示候选片段，便于直观看差异
        if args.hybrid:
            results = hyde.retrieve_hybrid(question=args.query, top_k=args.top_k, enable_rerank=args.rerank)
        elif args.hyde:
            results = hyde.retrieve_with_hyde(question=args.query, top_k=args.top_k, enable_rerank=args.rerank)
        else:
            results = svc.retrieve(query=args.query, top_k=args.top_k, enable_rerank=args.rerank)
    
        print("\n=== 候选片段 ===")
        if not results:
            print("未找到相关结果")
            return
        for i, r in enumerate(results, start=1):
            preview = r.text.strip().replace("\n", " ")[:200]
            if r.rerank_score is not None:
                print(f"Top{i}: rerank_score={r.rerank_score:.4f}, sim_score={r.sim_score:.4f}, source={r.source}")
            else:
                print(f"Top{i}: sim_score={r.sim_score:.4f}, source={r.source}")
            print(f"  {preview}...\n")


if __name__ == "__main__":
    main()


