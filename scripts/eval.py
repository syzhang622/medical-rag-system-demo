# -*- coding: utf-8 -*-
"""
阶段二 - 第一步：检索侧评估脚本

🎯 这个脚本做什么？
- 测试我们的检索系统是否能把"正确答案"找出来
- 不调用LLM生成答案，只评估"检索"这一步的效果

📊 评估指标：
1. Hit@K：检索出的前K个片段中，是否包含我们期望的关键词？
   - 例如：问"感冒症状"，期望找到"发热、咳嗽"
   - 如果前3个片段中有任何一个包含这些词，就算命中
   
2. 关键词覆盖率：检索出的片段中，覆盖了多少个期望关键词？
   - 例如：期望4个关键词，找到了3个，覆盖率=75%

🔧 怎么用？
python scripts/eval.py --k 3 --export results.csv
python scripts/eval.py --k 3 --rerank --export results_rerank.csv

💡 为什么重要？
- 如果检索都找不到正确答案，LLM再强也没用
- 通过对比不同参数（K值、是否重排），找到最佳配置
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple, Optional
import statistics

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.config import Config
from core.retrieval import RetrievalService
from core.hyde import HyDERetriever


def normalize(s: str) -> str:
    """
    🔧 文本标准化函数
    作用：把文本转成小写，去掉首尾空格
    为什么需要：让关键词匹配不区分大小写，避免空格导致的匹配失败
    例如："症状" 能匹配 "主要症状包括..."
    """
    return (s or "").strip().lower()


def keyword_presence_in_text(text: str, keywords: List[str]) -> Dict[str, bool]:
    """
    🔍 检查关键词是否在文本中出现
    
    输入：
    - text: 要搜索的文本（比如检索到的文档片段）
    - keywords: 期望找到的关键词列表（比如["症状", "发热"]）
    
    输出：
    - 字典：{关键词: 是否出现}
    例如：{"症状": True, "发热": False, "并发症": True}
    
    工作原理：
    1. 把文本和关键词都转成小写（不区分大小写）
    2. 逐个检查每个关键词是否在文本中出现
    3. 返回每个关键词的匹配结果
    """
    text_n = normalize(text)  # 标准化文本
    presence: Dict[str, bool] = {}
    
    for kw in keywords:
        kwn = normalize(kw)  # 标准化关键词
        # 检查关键词是否在文本中（如果两者都不为空）
        presence[kw] = (kwn in text_n) if (kwn and text_n) else False
    
    return presence


def evaluate_single_question(
    svc: RetrievalService,
    hyde: Optional[HyDERetriever],
    question: str,
    expected_keywords: List[str],
    top_k: int,
    enable_rerank: bool,
    use_hyde: bool,
    use_hybrid: bool,
) -> Dict[str, object]:
    """
    📊 对单个问题进行完整的检索侧评估
    
    输入：
    - svc: 检索服务（已经加载好索引）
    - question: 要测试的问题（比如"感冒和流感有什么区别？"）
    - expected_keywords: 期望找到的关键词（比如["症状", "发热", "并发症"]）
    - top_k: 检索前K个片段（比如3）
    - enable_rerank: 是否启用重排
    
    输出：评估结果字典，包含两个核心指标
    
    🎯 两个核心指标：
    1. hit_at_k: 检索是否成功？
       - 检查前K个片段中，是否有任何一个包含期望的关键词
       - 1表示成功，0表示失败
       
    2. keyword_coverage: 关键词覆盖率？
       - 把所有检索到的片段合并，看覆盖了多少个期望关键词
       - 0.75表示覆盖了75%的关键词
    """
    # 步骤1：执行检索，获取前K个最相关的文档片段
    if use_hybrid and hyde is not None:
        results = hyde.retrieve_hybrid(question=question, top_k=top_k, enable_rerank=enable_rerank)
        mode = "hybrid"
    elif use_hyde and hyde is not None:
        results = hyde.retrieve_with_hyde(question=question, top_k=top_k, enable_rerank=enable_rerank)
        mode = "hyde"
    else:
        results = svc.retrieve(query=question, top_k=top_k, enable_rerank=enable_rerank)
        mode = "base"
    print(f"    检索到 {len(results)} 个片段")

    # 步骤2：计算关键词覆盖率
    # 把所有检索到的片段文本合并成一个长文本
    concatenated = "\n".join([r.text or "" for r in results])
    # 检查合并后的文本中包含了哪些期望关键词
    presence_union = keyword_presence_in_text(concatenated, expected_keywords)

    # 统计命中的关键词数量
    num_present = sum(1 for v in presence_union.values() if v)
    total = max(1, len(expected_keywords))  # 避免除零
    keyword_coverage = num_present / total

    # 步骤3：计算Hit@K 与 被引用片段（代理）名次
    # 检查前K个片段中，是否有任何一个片段包含期望关键词
    hit_at_k = False
    best_rank: int = 0  # 1-based，0 表示未找到
    for r in results:
        p = keyword_presence_in_text(r.text or "", expected_keywords)
        if any(p.values()):  # 如果这个片段包含任何期望关键词
            hit_at_k = True
            if best_rank == 0:
                best_rank = results.index(r) + 1

    # 步骤4：整理结果
    present_keywords = [kw for kw, ok in presence_union.items() if ok]

    return {
        "question": question,
        "top_k": top_k,
        "enable_rerank": enable_rerank,
        "mode": mode,
        "hit_at_k": int(hit_at_k),  # 1或0
        "keyword_coverage": round(keyword_coverage, 4),  # 0-1之间的小数
        "num_present": num_present,  # 命中的关键词数量
        "num_expected": total,  # 期望的关键词总数
        "present_keywords": ",".join(present_keywords),  # 命中的关键词列表
        "citation_in_topk": int(best_rank > 0),
        "citation_best_rank": (best_rank or None),
    }


def export_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    """
    💾 导出评估结果到CSV文件
    
    作用：把评估结果保存成表格，方便后续分析和对比
    文件内容：每行是一个问题的评估结果，包含所有指标
    """
    if not rows:
        return
    
    # 确保目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 获取表头（所有字段名）
    headers = list(rows[0].keys())
    
    # 写入CSV文件
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()  # 写入表头
        writer.writerows(rows)  # 写入所有数据行


def summarize(rows: List[Dict[str, object]]) -> Tuple[float, float]:
    """
    📈 计算整体评估指标的平均值
    
    输入：所有问题的评估结果列表
    输出：(平均Hit@K, 平均关键词覆盖率)
    
    作用：从多个问题的结果中，计算整体表现
    例如：3个问题，Hit@K分别是[1,0,1]，平均就是0.67
    """
    if not rows:
        return 0.0, 0.0
    
    # 计算平均Hit@K（成功的问题数 / 总问题数）
    avg_hit = sum(int(r["hit_at_k"]) for r in rows) / len(rows)
    
    # 计算平均关键词覆盖率
    avg_cov = sum(float(r["keyword_coverage"]) for r in rows) / len(rows)
    
    return avg_hit, avg_cov


def build_argparser() -> argparse.ArgumentParser:
    """
    🔧 构建命令行参数解析器
    
    支持的参数：
    - --k: 检索前K个片段（默认用配置文件中的值）
    - --rerank: 是否启用重排（提升精度但速度慢）
    - --export: 导出CSV文件路径（可选）
    """
    p = argparse.ArgumentParser(description="检索侧评估：Hit@K & 关键词覆盖率 & 引用可信度")
    p.add_argument("--k", dest="top_k", type=int, default=None, 
                   help="Top-K（默认取配置SIMILARITY_TOP_K）")
    p.add_argument("--rerank", action="store_true", 
                   help="是否启用交叉编码器重排（提升精度但速度慢）")
    p.add_argument("--export", type=str, default=None, 
                   help="导出CSV路径，如 data/eval_results.csv")
    p.add_argument("--repeats", type=int, default=1,
                   help="每个问题重复评估次数（默认1，用于稳定性快测）")
    # HyDE/混合检索
    p.add_argument("--hyde", action="store_true", help="使用HyDE假设答案进行检索")
    p.add_argument("--hybrid", action="store_true", help="原始+HyDE检索结果融合")
    return p


def main() -> None:
    """
    🚀 主函数：执行完整的评估流程
    
    执行步骤：
    1. 解析命令行参数
    2. 加载配置和测试问题
    3. 初始化检索服务并加载索引
    4. 对每个测试问题执行评估
    5. 计算整体指标并输出结果
    6. 可选：导出CSV文件
    """
    print("🚀 开始执行评估脚本...")
    
    # 步骤1：解析命令行参数
    args = build_argparser().parse_args()
    print(f"📋 解析参数: {args}")

    # 步骤2：加载配置
    cfg = Config()
    print(f"⚙️ 配置加载完成，测试问题数量: {len(cfg.TEST_QUESTIONS)}")
    
    # 步骤3：初始化检索服务
    svc = RetrievalService(cfg)
    hyde_svc = HyDERetriever(cfg=cfg, retrieval=svc)
    print("🔄 正在加载索引与嵌入模型...")
    svc.load()
    print("✅ 索引加载完成")

    # 步骤4：设置评估参数
    k = int(args.top_k or cfg.SIMILARITY_TOP_K)
    enable_rerank = bool(args.rerank)
    print(f"\n📊 评估参数: K={k}, rerank={enable_rerank}")

    # 步骤5：对每个测试问题执行评估
    rows: List[Dict[str, object]] = []
    print(f"\n🔍 开始评估 {len(cfg.TEST_QUESTIONS)} 个问题...")
    
    for i, item in enumerate(cfg.TEST_QUESTIONS):
        q = item.get("question", "").strip()
        expected_keywords = item.get("expected_keywords", []) or []
        
        print(f"\n📝 [{i+1}] 问题: {q}")
        print(f"🎯 期望关键词: {expected_keywords}")
        
        # 执行评估（支持重复以做稳定性快测）
        repeat = max(1, int(args.repeats))
        recs: List[Dict[str, object]] = []
        for _ in range(repeat):
            recs.append(
                evaluate_single_question(
                    svc=svc,
                    hyde=hyde_svc,
                    question=q,
                    expected_keywords=expected_keywords,
                    top_k=k,
                    enable_rerank=enable_rerank,
                    use_hyde=bool(args.hyde),
                    use_hybrid=bool(args.hybrid),
                )
            )

        # 聚合统计
        hits = [int(r["hit_at_k"]) for r in recs]
        covs = [float(r["keyword_coverage"]) for r in recs]
        cit_in = [int(r["citation_in_topk"]) for r in recs]
        cit_rank = [r["citation_best_rank"] or 0 for r in recs]

        agg = dict(recs[-1])
        agg.update({
            "hit_at_k_mean": round(sum(hits)/len(hits), 4),
            "hit_at_k_std": round(statistics.pstdev(hits) if len(hits) > 1 else 0.0, 4),
            "keyword_coverage_mean": round(sum(covs)/len(covs), 4),
            "keyword_coverage_std": round(statistics.pstdev(covs) if len(covs) > 1 else 0.0, 4),
            "citation_in_topk_mean": round(sum(cit_in)/len(cit_in), 4),
            "citation_best_rank_min": (min([c for c in cit_rank if c > 0]) if any(c > 0 for c in cit_rank) else None),
            "repeats": repeat,
        })
        rows.append(agg)

        # 显示结果（以聚合后的最后一次记录为代表）
        print(f"📈 结果: hit@{k}={agg['hit_at_k']} | 关键词覆盖率={agg['keyword_coverage']:.3f} ({agg['num_present']}/{agg['num_expected']})")
        if agg["present_keywords"]:
            print(f"✅ 命中的关键词: {agg['present_keywords']}")
        else:
            print("❌ 未命中任何关键词")

    # 步骤6：计算并显示整体指标
    avg_hit, avg_cov = summarize(rows)
    print("\n" + "="*50)
    print("📊 整体评估结果")
    print("="*50)
    print(f"🎯 平均 Hit@{k}: {avg_hit:.3f} ({avg_hit*100:.1f}%)")
    print(f"📈 平均关键词覆盖率: {avg_cov:.3f} ({avg_cov*100:.1f}%)")
    
    # 步骤7：导出CSV（如果指定了路径）
    if args.export:
        export_csv(rows, args.export)
        print(f"\n💾 已导出CSV: {args.export}")
        print("💡 可以用Excel或其他工具打开查看详细结果")


if __name__ == "__main__":
    main()


