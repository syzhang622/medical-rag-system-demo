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
from core.retrieval import RetrievalService


def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器
    
    定义CLI支持的所有参数和选项
    """
    p = argparse.ArgumentParser(description="RAG 检索 CLI")
    
    # 必需参数：查询问题
    p.add_argument("--q", "--query", dest="query", type=str, required=True, 
                   help="要查询的问题（必需）")
    
    # 可选参数：返回结果数量
    p.add_argument("--k", dest="top_k", type=int, default=None, 
                   help="返回结果数量（默认使用配置文件中的SIMILARITY_TOP_K）")
    
    # 可选参数：是否启用重排
    p.add_argument("--rerank", action="store_true", 
                   help="是否启用交叉编码器重排（提升结果质量但速度较慢）")
    
    return p


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
    
    # 步骤2：创建检索服务实例
    svc = RetrievalService()
    
    # 步骤3：加载索引和嵌入模型（这一步可能较慢，因为要加载模型）
    print("正在加载索引和模型...")
    svc.load()
    
    # 步骤4：执行检索查询
    print(f"正在检索: {args.query}")
    results = svc.retrieve(query=args.query, top_k=args.top_k, enable_rerank=args.rerank)

    # 步骤5：格式化并显示结果
    print("\n=== 候选片段 ===")
    if not results:
        print("未找到相关结果")
        return
        
    for i, r in enumerate(results, start=1):
        # 截取文本预览（前200字符）
        preview = r.text.strip().replace("\n", " ")[:200]
        
        # 根据是否有重排分数选择显示格式
        if r.rerank_score is not None:
            # 有重排分数：显示重排分数和原始相似度分数
            print(f"Top{i}: rerank_score={r.rerank_score:.4f}, base_sim={r.score:.4f}, source={r.source}")
        else:
            # 无重排分数：只显示相似度分数
            print(f"Top{i}: sim={r.score:.4f}, source={r.source}")
        
        print(f"  {preview}...\n")


if __name__ == "__main__":
    main()


