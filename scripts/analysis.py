# -*- coding: utf-8 -*-
"""
RAG系统数据分析和可视化工具

WHY（为什么需要这个工具）：
- 在构建RAG系统之前，我们需要理解数据的分块效果和向量化质量
- 通过可视化分析，可以验证文本分块是否合理、嵌入模型是否有效
- 帮助调试和优化RAG系统的检索效果

HOW（如何工作）：
1) 读取医疗FAQ文本 → 获取原始数据
2) 按配置参数分块 → 模拟RAG系统的文本处理流程
3) 计算文本嵌入 → 将文本转换为向量表示
4) 保存分析结果 → 生成可查看的CSV和可视化数据
5) PCA降维可视化 → 将高维向量投影到2D平面
6) 相似度分析 → 展示文本片段之间的关联性

输出文件说明：
- chunks.csv: 分块详情（位置、长度、重叠等）
- chunks_annotated.txt: 带标记的分块文本，便于人工检查
- embeddings.npy: 文本嵌入向量（384维）
- projection.csv: 2D投影坐标，可用于绘图可视化
"""

import os
import csv
import math
import json
import numpy as np
from typing import List, Tuple

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config


def read_text(path: str) -> str:
    """
    读取文本文件
    
    WHY: 需要获取原始医疗FAQ数据进行分析
    HOW: 使用UTF-8编码读取文件内容
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text_with_spans(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    文本分块处理，返回带位置信息的文本片段
    
    WHY: 
    - RAG系统需要将长文档切分成小块进行向量化
    - 记录位置信息便于分析分块效果和重叠情况
    - 与主系统的分块逻辑保持一致
    
    HOW:
    - 按chunk_size大小切分文本
    - 相邻块之间有overlap个字符的重叠
    - 返回(start, end, chunk_text)元组列表
    
    参数:
    - chunk_size: 每个文本块的最大字符数
    - overlap: 相邻块之间的重叠字符数
    """
    spans: List[Tuple[int, int, str]] = []
    start: int = 0
    text_length: int = len(text)
    step: int = max(1, chunk_size - overlap)  # 计算步长，确保有重叠
    while start < text_length:
        end = min(text_length, start + chunk_size)
        spans.append((start, end, text[start:end]))
        if end == text_length:
            break
        start += step
    return spans


def ensure_output_dir(dir_path: str) -> None:
    """
    确保输出目录存在
    
    WHY: 保存分析结果前需要确保目录存在，避免文件写入失败
    HOW: 检查目录是否存在，不存在则创建
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def save_chunks_csv(spans: List[Tuple[int, int, str]], chunk_size: int, overlap: int, out_path: str) -> None:
    """
    保存分块信息到CSV文件
    
    WHY: 
    - 便于人工检查分块效果（长度、重叠、内容等）
    - 可以用Excel等工具进行数据分析和可视化
    - 为后续优化分块参数提供数据支持
    
    HOW:
    - 计算每个分块的长度和与前一个分块的重叠
    - 保存为CSV格式，包含分块ID、位置、长度、重叠、内容等字段
    """
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "start", "end", "length", "overlap_with_prev", "text"])
        prev_start, prev_end = None, None
        for i, (s, e, c) in enumerate(spans):
            length = e - s
            if prev_start is None:
                ov = 0  # 第一个分块没有重叠
            else:
                ov = max(0, prev_end - s)  # 计算与前一个分块的重叠长度
            writer.writerow([i, s, e, length, ov, c])
            prev_start, prev_end = s, e


def save_projection_csv(xy: np.ndarray, out_path: str) -> None:
    """
    保存PCA降维后的2D坐标到CSV文件
    
    WHY:
    - 高维向量（384维）无法直接可视化
    - 通过PCA降维到2D，可以用散点图展示文本片段的分布
    - 帮助理解文本片段的聚类和相似性关系
    
    HOW:
    - 将每个分块的2D坐标保存为CSV格式
    - 可用于matplotlib等工具绘制散点图
    """
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "x", "y"])
        for i, (x, y) in enumerate(xy.tolist()):
            writer.writerow([i, x, y])


def compute_embeddings(chunks: List[str]) -> np.ndarray:
    """
    计算文本嵌入向量
    
    WHY:
    - 将文本转换为数值向量，便于计算相似度
    - 验证嵌入模型的效果，确保相似文本有相近的向量表示
    - 为后续的相似度分析和可视化提供数据
    
    HOW:
    - 使用sentence-transformers模型将文本转换为384维向量
    - 对向量进行L2归一化，便于计算余弦相似度
    - 返回numpy数组格式的嵌入矩阵
    """
    # 直接使用 sentence-transformers，避免额外依赖初始化耗时
    from sentence_transformers import SentenceTransformer
    model_name = Config.EMBEDDING_MODEL
    model = SentenceTransformer(model_name)
    vecs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return vecs


def print_neighbors(chunks: List[str], embeddings: np.ndarray, k: int = 3, samples: int = 5) -> None:
    """
    打印文本片段的最近邻分析结果
    
    WHY:
    - 验证嵌入模型是否能够正确识别相似文本
    - 直观展示RAG系统的检索效果
    - 帮助理解文本片段之间的语义关联
    
    HOW:
    - 计算所有文本片段之间的余弦相似度矩阵
    - 对每个样本找到最相似的k个邻居
    - 打印相似度分数和文本内容预览
    """
    sim = cosine_similarity(embeddings, embeddings)
    total = len(chunks)
    samples = min(samples, total)
    print("\n=== 最近邻示例（基于余弦相似度）===")
    for i in range(samples):
        sims = sim[i]
        # 排除自身，取前 k 个邻居
        nn_idx = np.argsort(-sims)
        nn_idx = [idx for idx in nn_idx if idx != i][:k]
        base_text = chunks[i][:120].replace("\n", " ")
        print(f"\n[样本 {i}]\n文本片段: {base_text}...")
        for rank, j in enumerate(nn_idx, start=1):
            neighbor_text = chunks[j][:80].replace("\n", " ")
            print(f"  Top{rank} -> id={j}, sim={sims[j]:.4f}, 片段: {neighbor_text}...")


def main() -> None:
    """
    主函数：执行完整的RAG数据分析流程
    
    WHY: 提供一键式的数据分析工具，帮助开发者理解数据质量
    HOW: 按步骤执行文本读取、分块、向量化、可视化等操作
    """
    print("=== RAG数据分析工具：开始 ===")
    txt_path = Config.MEDICAL_FAQ_PATH
    assert os.path.exists(txt_path), f"找不到文件: {txt_path}"

    # 步骤1: 读取原始文本数据
    print("1) 读取文本...")
    text = read_text(txt_path)
    print(f"   文本长度: {len(text)} 字符")

    # 步骤2: 文本分块处理
    print("2) 分块...")
    spans = chunk_text_with_spans(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
    chunks = [c for (_, _, c) in spans]
    example_text = chunks[0][:80].replace("\n", " ")
    print(f"   分块数: {len(chunks)}, 示例片段: {example_text}...")

    # 步骤3: 保存分块分析结果
    out_dir = "data/analysis_results"
    ensure_output_dir(out_dir)
    save_chunks_csv(spans, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP, os.path.join(out_dir, "chunks.csv"))
    print(f"   已保存: {os.path.join(out_dir, 'chunks.csv')} (包含 start/end/重叠 等字段)")

    # 生成带边界标记的文本，便于直观看到每个分块的范围
    annotated_path = os.path.join(out_dir, "chunks_annotated.txt")
    with open(annotated_path, 'w', encoding='utf-8') as af:
        for i, (s, e, c) in enumerate(spans):
            header = f"\n===== CHUNK {i} | start={s} end={e} len={e - s} =====\n"
            af.write(header)
            af.write(c)
            af.write("\n")
    print(f"   已保存: {annotated_path}")

    # 步骤4: 计算文本嵌入向量
    print("3) 计算嵌入（这一步可能稍慢）...")
    embeddings = compute_embeddings(chunks)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    print(f"   向量形状: {embeddings.shape}, 已保存: {os.path.join(out_dir, 'embeddings.npy')}")

    # 步骤5: PCA降维可视化
    print("4) PCA 降维到 2D 并保存...")
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(embeddings)
    save_projection_csv(xy, os.path.join(out_dir, "projection.csv"))
    print(f"   已保存: {os.path.join(out_dir, 'projection.csv')}")

    # 步骤6: 相似度分析
    print("5) 打印最近邻示例...")
    print_neighbors(chunks, embeddings, k=3, samples=min(5, len(chunks)))

    print("\n=== RAG数据分析工具：完成 ===")
    print("分析结果已保存到 data/viz_artifacts/ 目录")
    print("- chunks.csv: 分块详情")
    print("- chunks_annotated.txt: 带标记的分块文本")
    print("- embeddings.npy: 文本嵌入向量")
    print("- projection.csv: 2D投影坐标（可用于绘图）")


if __name__ == "__main__":
    main()


