#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 docstore.json 文件内容
"""

import json
import os

def check_docstore():
    """检查 docstore.json 文件内容"""
    docstore_path = "data/faiss_index/sentence-transformers_all-MiniLM-L6-v2_cs512_co50/docstore.json"
    
    if not os.path.exists(docstore_path):
        print(f"❌ 文件不存在: {docstore_path}")
        return
    
    try:
        # 读取 docstore.json 文件
        with open(docstore_path, 'r', encoding='utf-8') as f:
            docstore_content = json.load(f)
        
        print("✅ 成功读取 docstore.json 文件")
        print(f"📁 文件大小: {os.path.getsize(docstore_path)} bytes")
        
        # 检查文档存储结构
        docstore_data = docstore_content.get('docstore', {})
        docs = docstore_data.get('docs', {})
        
        print(f"📊 文档存储中的节点数量: {len(docs)}")
        
        if len(docs) == 0:
            print("❌ 警告: 文档存储中没有节点!")
            return
        
        # 显示前几个节点的文本片段
        print("\n📝 前几个节点的文本片段:")
        for i, (node_id, node_data) in enumerate(list(docs.items())[:3]):
            text = node_data.get('_data_', {}).get('text', '')
            text_preview = text[:100] + '...' if len(text) > 100 else text
            print(f"节点{i+1} (ID: {node_id[:8]}...): {text_preview}")
        
        # 检查是否有文本内容
        has_text = any(
            node_data.get('_data_', {}).get('text', '') 
            for node_data in docs.values()
        )
        
        if has_text:
            print("✅ 文档存储包含文本内容")
        else:
            print("❌ 警告: 文档存储中没有文本内容!")
            
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")

if __name__ == "__main__":
    check_docstore()
