#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键评估所有模式 - 自动运行完整评估

用法：
    python -m scripts.run_eval_all
    
功能：
1. 依次运行 base、hyde、hybrid 三种模式的评估
2. 可选：运行带重排的评估
3. 生成汇总对比报告
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_evaluation(mode: str, k: int = 3, rerank: bool = False) -> str:
    """运行单个评估任务
    
    Args:
        mode: 评估模式 (base/hyde/hybrid)
        k: Top-K值
        rerank: 是否启用重排
        
    Returns:
        输出文件路径
    """
    # 构建输出文件名
    suffix = f"_rerank" if rerank else ""
    output_file = f"data/eval_results_{mode}{suffix}.csv"
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "scripts.eval",
        "--k", str(k),
        "--export", output_file
    ]
    
    # 添加模式参数
    if mode == "hyde":
        cmd.append("--hyde")
    elif mode == "hybrid":
        cmd.append("--hybrid")
    
    # 添加重排参数
    if rerank:
        cmd.append("--rerank")
    
    # 执行评估
    mode_name = f"{mode.upper()}" + (" + Rerank" if rerank else "")
    print(f"\n{'='*60}")
    print(f"🔄 运行评估: {mode_name}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {mode_name} 评估完成")
        print(f"📄 结果保存至: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"❌ {mode_name} 评估失败: {e}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断了 {mode_name} 评估")
        raise


def load_and_summarize(file_path: str) -> dict:
    """加载评估结果并计算汇总指标
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        汇总指标字典
    """
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    
    return {
        "file": file_path,
        "hit_rate": df["hit_at_k"].mean() * 100,  # 转换为百分比
        "keyword_coverage": df["keyword_coverage"].mean() * 100,
        "best_rank_avg": df["citation_best_rank"].mean(),
        "num_questions": len(df)
    }


def print_comparison_table(results: dict):
    """打印对比表格
    
    Args:
        results: 各模式的评估结果
    """
    print(f"\n{'='*80}")
    print("📊 评估结果汇总对比")
    print(f"{'='*80}\n")
    
    # 表头
    print(f"{'模式':<20} {'Hit@K':<12} {'关键词覆盖':<12} {'平均最佳排名':<12} {'问题数':<8}")
    print("-" * 80)
    
    # 数据行
    for mode_name, summary in results.items():
        if summary:
            print(f"{mode_name:<20} "
                  f"{summary['hit_rate']:>8.1f}%    "
                  f"{summary['keyword_coverage']:>8.1f}%    "
                  f"{summary['best_rank_avg']:>11.2f}     "
                  f"{summary['num_questions']:>6}")
    
    print("-" * 80)
    
    # 找出最佳模式
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_hit = max(valid_results.items(), key=lambda x: x[1]['hit_rate'])
        best_coverage = max(valid_results.items(), key=lambda x: x[1]['keyword_coverage'])
        
        print(f"\n🏆 最佳命中率: {best_hit[0]} ({best_hit[1]['hit_rate']:.1f}%)")
        print(f"🏆 最佳关键词覆盖: {best_coverage[0]} ({best_coverage[1]['keyword_coverage']:.1f}%)")
    
    print()


def main():
    """主函数"""
    print("="*80)
    print("🚀 RAG系统完整评估")
    print("="*80)
    print()
    print("📋 评估计划:")
    print("  1. Base (基线检索)")
    print("  2. HyDE (假设答案扩展)")
    print("  3. Hybrid (混合检索)")
    print()
    
    # 询问是否包含重排评估
    include_rerank = input("是否包含重排(Rerank)评估? (y/n, 默认n): ").strip().lower()
    include_rerank = include_rerank == 'y'
    
    if include_rerank:
        print("✅ 将运行 6 个评估任务（3种模式 × 2种配置）")
    else:
        print("✅ 将运行 3 个评估任务")
    
    print()
    input("按 Enter 开始评估...")
    
    # 运行评估
    results = {}
    
    try:
        # 基础评估
        modes = ["base", "hyde", "hybrid"]
        for mode in modes:
            output_file = run_evaluation(mode, k=3, rerank=False)
            if output_file:
                results[mode.upper()] = load_and_summarize(output_file)
        
        # 重排评估（可选）
        if include_rerank:
            for mode in modes:
                output_file = run_evaluation(mode, k=3, rerank=True)
                if output_file:
                    results[f"{mode.upper()} + Rerank"] = load_and_summarize(output_file)
        
        # 打印对比表格
        print_comparison_table(results)
        
        # 保存汇总结果
        summary_file = "data/eval_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("RAG系统评估结果汇总\n")
            f.write("="*80 + "\n\n")
            for mode_name, summary in results.items():
                if summary:
                    f.write(f"{mode_name}:\n")
                    f.write(f"  - Hit@K: {summary['hit_rate']:.1f}%\n")
                    f.write(f"  - 关键词覆盖: {summary['keyword_coverage']:.1f}%\n")
                    f.write(f"  - 平均最佳排名: {summary['best_rank_avg']:.2f}\n")
                    f.write(f"  - 问题数: {summary['num_questions']}\n\n")
        
        print(f"📄 汇总结果已保存至: {summary_file}")
        print("\n✅ 所有评估任务完成！")
        print(f"\n💡 提示: 查看 docs/EVALUATION_REPORT.md 了解详细分析")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        print("已完成的评估结果已保存")
        sys.exit(1)


if __name__ == "__main__":
    main()

