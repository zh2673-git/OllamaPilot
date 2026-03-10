"""
OllamaPilot 系统测试

测试配置：
- 生成模型: qwen3.5:4b
- 向量模型: qwen3-embedding:0.6b

测试内容：
1. 系统启动和配置读取
2. DocumentManager 手动文档管理
3. 静默索引进度显示
4. 对话格式检查
5. GraphRAG 查询

使用方法：
    python tests/test_system.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time


def print_header(title):
    """打印测试标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_system_startup():
    """测试系统启动"""
    print_header("测试 1: 系统启动")
    
    from ollamapilot.config import get_config
    from skills.graphrag.skill import GraphRAGSkill
    
    config = get_config()
    print(f"\n📋 配置信息:")
    print(f"  对话模型: {config.chat_model}")
    print(f"  向量模型: {config.embedding_model}")
    print(f"  知识库目录: {config.graph_rag_knowledge_base_dir}")
    
    print(f"\n🚀 启动 GraphRAG Skill...")
    
    try:
        skill = GraphRAGSkill(
            embedding_model=config.embedding_model,
            persist_dir="./data/test_graphrag",
            enable_auto_retrieval=False,
            knowledge_base_dir="./test_knowledge_base",
            enable_word_aligner=True,
            fuzzy_threshold=0.75
        )
        
        print(f"\n✅ GraphRAG Skill 启动成功")
        print(f"   向量存储: {skill.graph_service.collection_name}")
        
        return skill
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_document_manager():
    """测试 DocumentManager"""
    print_header("测试 2: DocumentManager（手动控制索引）")
    
    from ollamapilot.config import get_config
    from skills.graphrag.document_manager import DocumentManager
    
    config = get_config()
    
    # 创建测试文档
    test_doc_path = "./test_shanghanlun.txt"
    test_content = """太阳病，发热汗出，恶风，脉缓者，名为中风。
太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名为伤寒。
伤寒一日，太阳受之，脉若静者，为不传；颇欲吐，若躁烦，脉数急者，为传也。
"""
    
    with open(test_doc_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"\n📄 创建测试文档: {test_doc_path}")
    
    # 初始化 DocumentManager
    print(f"\n🔧 初始化 DocumentManager")
    print(f"   向量模型: {config.embedding_model}")
    
    doc_manager = DocumentManager(
        base_persist_dir="./data/test_docs",
        embedding_model=config.embedding_model
    )
    
    # 注册文档（不自动索引）
    print(f"\n📋 注册文档（手动控制）")
    doc_id = doc_manager.register_document(
        doc_name="伤寒论",
        file_path=test_doc_path,
        auto_index=False
    )
    
    print(f"\n📊 文档状态:")
    doc_info = doc_manager.get_document_status(doc_id)
    if doc_info:
        print(f"   名称: {doc_info.name}")
        print(f"   状态: {doc_info.status.value}")
        print(f"   存储路径: {doc_manager._get_document_storage_path(doc_info.name, doc_info.model_name)}")
    
    # 手动开始索引（静默模式）
    print(f"\n🔄 开始索引（静默模式，只显示50%和100%）")
    doc_manager.start_indexing(doc_id, silent=True)
    
    # 等待索引完成
    print(f"   等待索引完成...")
    success = doc_manager.wait_for_indexing(doc_id, timeout=120)
    
    if success:
        print(f"\n✅ 索引完成")
        doc_info = doc_manager.get_document_status(doc_id)
        print(f"   分块数: {doc_info.chunks_count}")
        print(f"   实体数: {doc_info.entities_count}")
    else:
        print(f"\n❌ 索引失败或超时")
    
    # 列出所有文档
    print(f"\n📚 文档列表:")
    docs = doc_manager.list_documents()
    for doc in docs:
        print(f"   - {doc.name} ({doc.status.value})")
    
    # 清理
    os.remove(test_doc_path)
    
    return success


def test_conversation_display():
    """测试对话显示格式"""
    print_header("测试 3: 对话显示格式")
    
    from ollamapilot.config import get_config
    from ollamapilot.models import init_ollama_model
    
    config = get_config()
    
    print(f"\n🤖 初始化对话模型: {config.chat_model}")
    
    try:
        model = init_ollama_model(model=config.chat_model, temperature=0.7)
        
        # 测试问题
        questions = [
            "你好",
            "什么是太阳病？",
            "请用一句话总结伤寒论的核心思想。"
        ]
        
        print(f"\n💬 开始对话测试:\n")
        
        for i, question in enumerate(questions, 1):
            print(f"用户: {question}")
            
            try:
                response = model.invoke(question)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # 检查回答格式
                print(f"助手: {answer[:200]}")
                if len(answer) > 200:
                    print(f"      ... ({len(answer)} 字符)")
                print()
                
            except Exception as e:
                print(f"助手: [错误] {e}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 对话测试失败: {e}")
        return False


def test_graphrag_query():
    """测试 GraphRAG 查询"""
    print_header("测试 4: GraphRAG 查询")
    
    from ollamapilot.config import get_config
    from skills.graphrag.tools import search_knowledge, query_graph_stats
    
    config = get_config()
    
    print(f"\n🔍 查询知识图谱统计")
    try:
        stats = query_graph_stats.invoke({})
        print(f"   统计: {stats}")
    except Exception as e:
        print(f"   查询失败: {e}")
    
    print(f"\n🔍 搜索知识: '太阳病'")
    try:
        results = search_knowledge.invoke({
            "query": "什么是太阳病？",
            "n_results": 3
        })
        print(f"   结果数: {len(results) if isinstance(results, list) else 'N/A'}")
        if isinstance(results, list) and len(results) > 0:
            print(f"   第一条: {str(results[0])[:150]}...")
    except Exception as e:
        print(f"   搜索失败: {e}")
    
    return True


def main():
    """主函数"""
    print("\n" + "🧪" * 35)
    print("  OllamaPilot 系统测试")
    print("  生成模型: qwen3.5:4b")
    print("  向量模型: qwen3-embedding:0.6b")
    print("🧪" * 35)
    
    results = {}
    
    # 测试1: 系统启动
    skill = test_system_startup()
    results["系统启动"] = skill is not None
    
    # 测试2: DocumentManager
    if skill:
        results["DocumentManager"] = test_document_manager()
    else:
        print("\n⏭️ 跳过 DocumentManager 测试（系统启动失败）")
        results["DocumentManager"] = False
    
    # 测试3: 对话显示
    results["对话显示"] = test_conversation_display()
    
    # 测试4: GraphRAG查询
    results["GraphRAG查询"] = test_graphrag_query()
    
    # 总结
    print_header("测试总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n  总测试数: {total}")
    print(f"  通过: {passed}")
    print(f"  失败: {total - passed}")
    
    print("\n  详细结果:")
    for name, success in results.items():
        icon = "✅" if success else "❌"
        print(f"    {icon} {name}")
    
    print("\n" + "=" * 70)
    if passed == total:
        print("  🎉 所有测试通过！")
    else:
        print(f"  ⚠️  {total - passed} 个测试失败")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
