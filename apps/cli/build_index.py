# -*- coding: utf-8 -*-
"""
构建索引 CLI：一键构建FAISS向量索引

初学者理解：
- 这是RAG系统的索引构建CLI工具
- 调用核心索引构建服务
- 提供用户友好的命令行界面

用法：
python -m apps.cli.build_index

功能：
1. 解析命令行参数
2. 调用索引构建服务
3. 显示构建结果
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.config import Config
from core.indexing import IndexingService




def build_argparser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    p = argparse.ArgumentParser(description="构建/重建 FAISS 索引")
    p.add_argument("--rebuild", action="store_true", 
                   help="强制重建索引（默认重建）")
    return p


def main() -> None:
    """主函数：构建向量索引"""
    print("=== FAISS索引构建工具 ===")
    
    # 解析命令行参数
    args = build_argparser().parse_args()
    
    try:
        # 创建配置
        cfg = Config()
        print(f"使用配置: 模型={cfg.EMBEDDING_MODEL}, 分块大小={cfg.CHUNK_SIZE}, 重叠={cfg.CHUNK_OVERLAP}")
        
        # 创建索引构建服务
        indexing_service = IndexingService(cfg)
        
        # 执行索引构建
        print("开始构建索引...")
        indexing_service.build_index_pipeline(rebuild_index=True)
        
        print("✅ 索引构建完成！")
        print(f"索引保存位置: {cfg.FAISS_INDEX_PATH}")
        
    except Exception as e:
        print(f"❌ 索引构建失败: {e}")
        raise


if __name__ == "__main__":
    main()
